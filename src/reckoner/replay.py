"""The Phase-2 replay ring: fixed capacity, `root_q` from field one, era-aware.

Memory arithmetic, shown before the capacity is declared
--------------------------------------------------------
Rider (c) applied to a resource: compute what the quantity can legitimately
reach *before* siting the alarm. Bytes per stored step, at the pinned shapes
(``seq_len 512``, ``max_sites 192``, ``gumbel_m 16``):

===================  ==========================================  =======
field                layout                                      bytes
===================  ==========================================  =======
``tokens``           512 × int16 (vocab 657 fits int16)             1024
``site_positions``   192 × int16                                     384
``length``           int16                                             2
``n_sites``          int16                                             2
``visit_actions``    16 × int32 — the top-m root actions, as dense     64
                     action indices (``rule * max_sites + site``;
                     F-41: format 1 stored root slot positions)
``visit_counts``     16 × int32                                       64
``root_q``           float32                                           4
``z``                int8                                              1
``par_source``       int8 (enum)                                       1
``par``              int16                                             2
``steps_remaining``  int16                                             2
``depth``            int8                                              1
``goal``             int8                                              1
``ring_format``      int16 — the format version this record was written under   2
``absent_mask``      uint32 — one bit per runtime-absent field           4
``improved_actions`` 160 × int32 — π′'s top-160 actions, dense      640
                     indices (F-42, format 3)
``improved_probs``   160 × float32 — their π′ mass                    640
``improved_tail``    float64 — π′ mass NOT stored, exact               8
**per record**                                                    **2846**
===================  ==========================================  =======

At ``train.replay_capacity`` = 500,000 that is **1.33 GiB** (0.72 GiB before
F-42's π′ fields), against ~45 GiB of realistically free host RAM since the
2026-08-16 rebalance (AGENTS.md §3). The capacity clears with more than an order
of margin, so it stays **decided at 500,000**. 200,000 would be 0.53 GiB and
100,000 would be 0.27 GiB, recorded so a later reduction is a choice with a
number attached.

The visit vector is stored **sparsely** — the top-``gumbel_m`` root actions and
their counts — not as a dense 1,344-wide distribution. The search visits at most
``m`` root actions by construction, so the visit layout is lossless.

**π′ is different (F-42):** it has mass on every legal action, so its top-160
store is lossless only up to 160 legal actions. Wider roots are written with the
dropped mass recorded exactly in ``improved_tail``, and when that mass exceeds
``IMPROVED_TAIL_TOL`` ``ring_batch`` withholds the row's POLICY target rather
than train on a cropped one — the row still trains the value and steps heads, so
the policy lever never changes who the value head learns from.

Two era systems exist. They are not the same one
-------------------------------------------------
* **``RING_FORMAT`` / ``_FIELDS_SINCE`` (here)** version the *memmap record
  layout*. A bump means a column was added to what a stored step contains — or
  that a column's stored **meaning** changed, in which case that column's
  ``since`` moves to the new format and records written under the old meaning
  read as era-absent rather than being reinterpreted. Format 2 is the first such
  bump: format-1 ``visit_actions`` hold root slot positions (F-41).
* **``logschema.SCHEMA_ERA`` (there)** versions *JSONL row columns*. A bump means
  a column was added to what an iteration reports.

Different substrates, different lifetimes, adjacent names, adjacent purposes —
which is the dedup-key hazard inverted and just as fatal to a grep. They are
deliberately spelled differently, and each module says the other exists so the
first person debugging an era mismatch learns there are **two** before learning
which one they are in.

One absence semantics
---------------------
A stored value is absent for exactly one of two reasons, and neither is ever a
raw zero:

1. **era absence** — the field did not exist when the record was written, or
   existed under a meaning since redefined (``visit_actions`` before format 2,
   F-41). Computed from ``record.ring_format`` against ``_FIELDS_SINCE[name]``,
   never stored, because a record written before a field existed could not have
   explained the absence of something nobody had named. (Same resolution as
   ``logschema``'s, for the same reason.)
2. **runtime absence** — the value genuinely was not available. One bit in
   ``absent_mask``; the *reason* is declared once on the field rather than stored
   per record, so absence is a bit and its meaning is a constant.

:meth:`ReplayRing.get` returns :class:`Absent` for either, carrying the reason.
It never returns the underlying zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reckoner.absence import AbsenceError, Absent
from reckoner.config import Config
from reckoner.dataset import sample_indices

#: Ring record-layout version. Bump when a FIELD IS ADDED to a stored step, or
#: when a field's stored MEANING changes (then move that field's ``since``).
#: Distinct from :data:`reckoner.logschema.SCHEMA_ERA`, which versions JSONL row
#: columns — see this module's docstring.
#:
#: 2 (F-41): ``visit_actions`` hold dense action indices. Format-1 records hold
#: root slot positions there, and the layout is otherwise identical.
#: 3 (F-42): adds ``improved_actions`` / ``improved_probs`` / ``improved_tail`` —
#: the Gumbel π′ over every legal root action, top ``IMPROVED_SLOTS`` by mass, with
#: the mass not stored recorded exactly. Appended fields; nothing redefined.
RING_FORMAT = 3

#: Visit slots the record layout provides. Part of RING_FORMAT: widening this is
#: a format bump, because every stored record's stride changes with it.
VISIT_SLOTS = 16

#: π′ slots per record (F-42). π′ has mass on EVERY legal action, so unlike the
#: visit share it is not bounded by ``gumbel_m``. Sized by width, not by how
#: peaked any network is: the smallest multiple of 32 at least 1.5 × the widest
#: root measured. First sized at 128 from uniform-stub episodes on the frozen
#: suites (widest 83); the F-42 review then measured 103 legal actions on rings a
#: random-init network fills from generator problems — the capped, wandering
#: episodes a weak policy produces — and the same rule gives 160. A record wider
#: than this is still written, its dropped mass recorded, and its POLICY target is
#: withheld if that mass exceeds ``IMPROVED_TAIL_TOL``. Part of the format:
#: widening it is a format bump.
IMPROVED_SLOTS = 160

#: π′ mass a record may drop and still train. Above it the row sits out, counted
#: — never cropped, never renormalised (the StateTooLarge precedent).
IMPROVED_TAIL_TOL = 1e-6

#: |Σ probs + tail − 1| allowed at the write and read boundaries: float32 rounding
#: over at most 128 entries, with two orders of margin.
IMPROVED_MASS_TOL = 1e-5

#: Known ``par_source`` values, as a stored enum. Order is part of the format:
#: appending is a format bump, reordering is a corruption.
PAR_SOURCES: tuple[str, ...] = ("unverified", "bfs", "scripted", "pool")


@dataclass(frozen=True, slots=True)
class RingField:
    """One stored column. ``absence`` is the reason it may be runtime-absent.

    ``since`` is the format from which the stored values mean what this module
    says they mean. ``since_reason``, when set, explains why older records cannot
    be read — for a column that existed before ``since`` under another meaning,
    "the field arrived later" would be false.
    """

    name: str
    since: int = 1
    absence: str | None = None
    since_reason: str | None = None


#: Every stored column and the format version it appeared in. The ring's era map.
_FIELDS_SINCE: tuple[RingField, ...] = (
    RingField("tokens"),
    RingField("site_positions"),
    RingField("length"),
    RingField("n_sites"),
    # Redefined in format 2 (F-41). The runner stored root SLOT positions here
    # while ring_batch read dense action indices, so a format-1 value is in the
    # wrong coordinates, not merely stale. Its counts are fine; without the
    # action each one belongs to they train nothing, and ring_batch sits the
    # whole record out.
    RingField(
        "visit_actions",
        since=2,
        since_reason=(
            "visit_actions held root SLOT positions (indices into legal_actions) "
            "before ring_format 2, not dense action indices — F-41; they cannot be "
            "read as actions"
        ),
    ),
    RingField("visit_counts"),
    # root_q is present from field one, not added later. The chess project
    # specified it and never stored it, and the gap was found only when the z/q
    # blend needed it — so it is in the first layout, with its sign proven by
    # test before it feeds anything (BRIEF-chunk9 §2).
    RingField("root_q"),
    RingField("z"),
    RingField("par_source"),
    RingField("par"),
    RingField("steps_remaining"),
    RingField("depth"),
    RingField("goal"),
    # --- format 3 (F-42). APPENDED, never inserted: a field's position is its
    # absent_mask bit, so an insertion would re-map every saved ring's bits.
    RingField("improved_actions", since=3),
    RingField("improved_probs", since=3),
    RingField("improved_tail", since=3),
)

_FIELD_INDEX = {f.name: i for i, f in enumerate(_FIELDS_SINCE)}
_FIELD_MAP = {f.name: f for f in _FIELDS_SINCE}


_MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def _mix64(index: int, seed: int) -> int:
    """splitmix64-style avalanche over ``(index, seed)``.

    The first version XOR'd the seed into a multiplied index, which changed the
    value by a bit or two and therefore almost never moved a row across the
    threshold — so different seeds produced the *same* split. The both-polarities
    test caught it: it asserted the seed matters, and it did not. A partitioner
    whose seed does nothing is a partitioner with one partition.
    """
    x = (index * 0x9E37_79B9_7F4A_7C15 + seed * 0xBF58_476D_1CE4_E5B9) & _MASK64
    x ^= x >> 30
    x = (x * 0xBF58_476D_1CE4_E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D0_49BB_1331_11EB) & _MASK64
    return x ^ (x >> 31)


class RingError(ValueError):
    """A malformed record or an impossible read."""


#: Re-exported so a reader of this module finds the absence type it returns.
#: One implementation, in `reckoner.absence` — a second copy of an absence type
#: is the two-implementations hazard the one-formatter law exists to prevent.
__all__ = [
    "IMPROVED_MASS_TOL",
    "IMPROVED_SLOTS",
    "IMPROVED_TAIL_TOL",
    "NO_IMPROVED_TARGET",
    "STALE_TARGETS",
    "Absent",
    "AbsenceError",
    "ReplayRing",
    "RingError",
    "bytes_per_record",
    "era_absent_on_disk",
]


class ReplayRing:
    """Fixed-capacity ring over episode steps. Overwrites oldest first."""

    def __init__(self, capacity: int, cfg: Config, *, ring_format: int = RING_FORMAT) -> None:
        if capacity < 1:
            raise RingError(f"capacity must be >= 1; got {capacity}")
        m = cfg.model
        self.capacity = int(capacity)
        self.cfg = cfg
        self.ring_format = ring_format
        self.cursor = 0
        self.count = 0

        self.tokens = np.zeros((capacity, m.seq_len), dtype=np.int16)
        self.site_positions = np.zeros((capacity, m.max_sites), dtype=np.int16)
        self.length = np.zeros(capacity, dtype=np.int16)
        self.n_sites = np.zeros(capacity, dtype=np.int16)
        width = cfg.search.gumbel_m
        # The layout is sized for gumbel_m visit slots, and m is CONFIG while the
        # layout is FORMAT — so load()'s layout refusal cannot see a config that
        # outgrew it. The m -> 32 lever is parked but live (chess parked the same
        # one); firing it against 16 slots would truncate visits silently, which
        # is a lossy ring wearing a lossless docstring. This is the assert that
        # makes that round trip loudly.
        if width > self.visit_actions_slots():
            raise RingError(
                f"search.gumbel_m = {width} exceeds the ring's {self.visit_actions_slots()} "
                "visit slots. The sparse visit layout is lossless only while m fits it. "
                "Raising m is a RING_FORMAT bump, not a config edit — bump the format "
                "and widen the layout together."
            )
        self.visit_actions = np.zeros((capacity, width), dtype=np.int32)
        self.visit_counts = np.zeros((capacity, width), dtype=np.int32)
        self.root_q = np.zeros(capacity, dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.int8)
        self.par_source = np.zeros(capacity, dtype=np.int8)
        self.par = np.zeros(capacity, dtype=np.int16)
        self.steps_remaining = np.zeros(capacity, dtype=np.int16)
        self.depth = np.zeros(capacity, dtype=np.int8)
        self.goal = np.zeros(capacity, dtype=np.int8)
        self.record_format = np.zeros(capacity, dtype=np.int16)
        self.absent_mask = np.zeros(capacity, dtype=np.uint32)
        self.improved_actions = np.zeros((capacity, IMPROVED_SLOTS), dtype=np.int32)
        self.improved_probs = np.zeros((capacity, IMPROVED_SLOTS), dtype=np.float32)
        # float64: the writer decides on this exact value and the reader must
        # decide on the same one, so it is stored at the precision it is summed.
        self.improved_tail = np.zeros(capacity, dtype=np.float64)

    # -- writing ----------------------------------------------------------

    @staticmethod
    def visit_actions_slots() -> int:
        """Visit slots the current RING_FORMAT layout provides."""
        return VISIT_SLOTS

    def append(
        self,
        *,
        tokens: np.ndarray,
        site_positions: np.ndarray,
        visit_actions: np.ndarray,
        visit_counts: np.ndarray,
        root_q: float,
        z: int,
        par_source: str,
        par: int,
        steps_remaining: int,
        depth: int,
        goal: int,
        improved_actions: np.ndarray,
        improved_probs: np.ndarray,
        improved_tail: float,
        absent: dict[str, str] | None = None,
    ) -> int:
        """Store one step. Returns its slot.

        ``absent`` names runtime-absent fields; each must be a field that
        *declares* it can be absent, because an undeclared absence is a hole
        nobody can interpret.
        """
        if z not in (-1, 0, 1):
            raise RingError(f"z must be -1, 0 or +1; got {z}")
        if par_source not in PAR_SOURCES:
            raise RingError(f"unknown par_source {par_source!r}; known: {list(PAR_SOURCES)}")
        if par_source == "bfs" and z > 0:
            raise RingError(
                "z = +1 against bfs par: beating exact par is impossible by "
                "construction. Third layer of the same tripwire (EpisodeResult, "
                "logschema, here) — a value this load-bearing is checked wherever "
                "it passes a boundary."
            )

        mask = 0
        for name, reason in (absent or {}).items():
            spec = _FIELD_MAP.get(name)
            if spec is None:
                raise RingError(f"unknown field {name!r} named absent")
            if spec.absence is None:
                raise RingError(
                    f"{name} does not declare that it can be absent; an undeclared "
                    "absence is a hole nobody can interpret. Give the field an "
                    "`absence` reason if it really can be missing."
                )
            if not str(reason).strip():
                raise RingError(f"{name}: absence needs a reason")
            mask |= 1 << _FIELD_INDEX[name]

        improved_actions = np.asarray(improved_actions)
        improved_probs = np.asarray(improved_probs)
        if improved_actions.shape != (IMPROVED_SLOTS,) or improved_probs.shape != (IMPROVED_SLOTS,):
            raise RingError(
                f"π′ fields must be {IMPROVED_SLOTS} wide; got actions "
                f"{improved_actions.shape}, probs {improved_probs.shape}"
            )
        if not (np.isfinite(improved_probs).all() and np.isfinite(improved_tail)):
            raise RingError("π′ carries a non-finite value")
        if (improved_probs < 0).any() or improved_tail < 0:
            raise RingError("π′ carries negative mass")
        total = float(improved_probs.astype(np.float64).sum()) + float(improved_tail)
        if abs(total - 1.0) > IMPROVED_MASS_TOL:
            raise RingError(
                f"π′ stored mass plus tail is {total!r}, not 1 within {IMPROVED_MASS_TOL}: "
                "a target that does not sum to one is not a distribution"
            )

        i = self.cursor
        n = int(len(tokens))
        s = int(len(site_positions))
        self.tokens[i] = 0
        self.tokens[i, :n] = tokens
        self.site_positions[i] = 0
        self.site_positions[i, :s] = site_positions
        self.length[i] = n
        self.n_sites[i] = s
        w = min(len(visit_actions), self.visit_actions.shape[1])
        self.visit_actions[i] = 0
        self.visit_counts[i] = 0
        self.visit_actions[i, :w] = visit_actions[:w]
        self.visit_counts[i, :w] = visit_counts[:w]
        self.root_q[i] = root_q
        self.z[i] = z
        self.par_source[i] = PAR_SOURCES.index(par_source)
        self.par[i] = par
        self.steps_remaining[i] = steps_remaining
        self.depth[i] = depth
        self.goal[i] = goal
        self.improved_actions[i] = improved_actions
        self.improved_probs[i] = improved_probs
        self.improved_tail[i] = improved_tail
        self.record_format[i] = self.ring_format
        self.absent_mask[i] = mask

        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        return i

    # -- reading ----------------------------------------------------------

    def _absence(self, slot: int, name: str) -> Absent | None:
        spec = _FIELD_MAP[name]
        record_format = int(self.record_format[slot])
        if record_format < spec.since:
            detail = spec.since_reason or f"{name} arrived in {spec.since}"
            return Absent(
                name,
                f"predates_field: record written under ring_format {record_format}, {detail}",
                "era",
            )
        if self.absent_mask[slot] & (1 << _FIELD_INDEX[name]):
            return Absent(name, spec.absence or "declared absent", "runtime")
        return None

    def get(self, slot: int) -> dict[str, object]:
        """One record, with absences reported rather than zeroed."""
        if not 0 <= slot < self.count:
            raise RingError(f"slot {slot} outside the {self.count} valid records")
        n, s = int(self.length[slot]), int(self.n_sites[slot])
        raw: dict[str, object] = {
            "tokens": self.tokens[slot, :n].copy(),
            "site_positions": self.site_positions[slot, :s].copy(),
            "length": n,
            "n_sites": s,
            "visit_actions": self.visit_actions[slot].copy(),
            "visit_counts": self.visit_counts[slot].copy(),
            "root_q": float(self.root_q[slot]),
            "z": int(self.z[slot]),
            "par_source": PAR_SOURCES[int(self.par_source[slot])],
            "par": int(self.par[slot]),
            "steps_remaining": int(self.steps_remaining[slot]),
            "depth": int(self.depth[slot]),
            "goal": int(self.goal[slot]),
            "improved_actions": self.improved_actions[slot].copy(),
            "improved_probs": self.improved_probs[slot].copy(),
            "improved_tail": float(self.improved_tail[slot]),
        }
        for name in raw:
            absent = self._absence(slot, name)
            if absent is not None:
                raw[name] = absent
        raw["ring_format"] = int(self.record_format[slot])
        return raw

    def era_absent(self, name: str) -> int:
        """How many valid records predate ``name``'s current meaning.

        A count, not a flag, so a caller refusing on it can say how much it
        refused. Read from the per-record format — never from the ring's own
        ``ring_format``, which only says what the next append will be stamped.
        """
        since = _FIELD_MAP[name].since
        return int(np.count_nonzero(self.record_format[: self.count] < since))

    def holdout(self, frac: float, seed: int = 0) -> set[int]:
        """Slots reserved for evaluation, never trained on.

        Deterministic in ``(slot, seed)`` rather than drawn per call, so the same
        row is on the same side of the split every time it is consulted — a
        holdout that re-rolls is a holdout the model has seen.
        """
        if not 0.0 <= frac < 1.0:
            raise RingError(f"holdout frac must be in [0, 1); got {frac}")
        if frac == 0.0:
            return set()
        cut = int(frac * 1_000_000)
        return {i for i in range(self.count) if _mix64(i, seed) % 1_000_000 < cut}

    def sample(self, k: int, seed: int) -> list[int]:
        """Slots sampled across the ring — never a prefix (the blessed sampler)."""
        return sample_indices(self.count, k, seed)

    def __len__(self) -> int:
        return self.count

    # -- persistence, for crash-resume ------------------------------------

    ARRAYS = (
        "tokens",
        "site_positions",
        "length",
        "n_sites",
        "visit_actions",
        "visit_counts",
        "root_q",
        "z",
        "par_source",
        "par",
        "steps_remaining",
        "depth",
        "goal",
        "record_format",
        "absent_mask",
        "improved_actions",
        "improved_probs",
        "improved_tail",
    )

    def save(self, path: Path) -> dict:
        """Arrays plus a meta sidecar. Meta is written LAST.

        Write ordering is the resume contract: a meta that exists means the
        arrays beside it are complete. A crash between the two leaves a ring
        without meta, which reads as "no ring" rather than as a truncated one.
        """
        path.mkdir(parents=True, exist_ok=True)
        for name in self.ARRAYS:
            # Write the layout of the format this ring DECLARES: a field added in
            # a later format is not part of it, and load() refuses a ring whose
            # files claim otherwise.
            spec = _FIELD_MAP.get(name)
            if spec is not None and spec.since > self.ring_format and spec.since_reason is None:
                continue
            getattr(self, name).tofile(path / f"{name}.bin")
        meta = {
            "ring_format": self.ring_format,
            "capacity": self.capacity,
            "cursor": self.cursor,
            "count": self.count,
            "seq_len": self.cfg.model.seq_len,
            "max_sites": self.cfg.model.max_sites,
            "gumbel_m": self.cfg.search.gumbel_m,
            "par_sources": list(PAR_SOURCES),
            "bytes_per_record": bytes_per_record(self.cfg, self.ring_format),
        }
        if self.ring_format >= 3:
            meta["improved_slots"] = IMPROVED_SLOTS
        (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return meta

    @classmethod
    def load(cls, path: Path, cfg: Config) -> ReplayRing:
        meta = json.loads((path / "meta.json").read_text())
        for key, got in (
            ("seq_len", cfg.model.seq_len),
            ("max_sites", cfg.model.max_sites),
            ("gumbel_m", cfg.search.gumbel_m),
        ):
            if meta[key] != got:
                raise RingError(
                    f"ring was written with {key}={meta[key]} but the config says {got}. "
                    "The record layout is denominated in these; loading anyway would "
                    "reinterpret every stored step."
                )
        if meta["par_sources"] != list(PAR_SOURCES):
            raise RingError(
                f"par_source enum changed: stored {meta['par_sources']}, now "
                f"{list(PAR_SOURCES)}. Order is part of the format — appending is a "
                "format bump, reordering is a corruption."
            )
        if meta["ring_format"] > RING_FORMAT:
            raise RingError(
                f"ring was written under ring_format {meta['ring_format']}, newer than this "
                f"code's {RING_FORMAT}. Code cannot vouch for a meaning it predates."
            )
        # Each record keeps the format it was WRITTEN under (``record_format``,
        # loaded below); the ring itself is rebuilt at the format THIS code
        # writes, because ``ring_format`` is what the next append is stamped
        # with. Rebuilding at the meta's format stamped every post-resume record
        # with the old format — inert while only format 1 existed, and at the
        # first bump (F-41) it would have made every new, correct record read as
        # era-absent. The layouts of formats 1 and 2 are identical; a future bump
        # that changes the layout fails the reshape below, loudly.
        if meta["ring_format"] >= 3 and meta.get("improved_slots") != IMPROVED_SLOTS:
            raise RingError(
                f"ring stores {meta.get('improved_slots')} π′ slots; this code's layout "
                f"is {IMPROVED_SLOTS}. Widening is a format bump, not a reinterpretation."
            )
        ring = cls(meta["capacity"], cfg)
        for name in cls.ARRAYS:
            target = getattr(ring, name)
            source = path / f"{name}.bin"
            spec = _FIELD_MAP.get(name)
            added_later = (
                spec is not None and spec.since > meta["ring_format"] and spec.since_reason is None
            )
            if added_later:
                # A field ADDED in a later format cannot be in an older ring. Its
                # array stays zero and every record reads it era-absent through
                # ``record_format`` — the absence is computed, never the zero. A
                # field REDEFINED later (it carries a ``since_reason``, like F-41's
                # visit_actions) did exist then, is loaded, and reads era-absent
                # the same way.
                if source.exists():
                    raise RingError(
                        f"a ring_format {meta['ring_format']} ring cannot contain "
                        f"{name}, which arrived in {spec.since}"
                    )
                continue
            if not source.exists():
                raise RingError(f"ring_format {meta['ring_format']} ring is missing {name}.bin")
            data = np.fromfile(source, dtype=target.dtype)
            target[...] = data.reshape(target.shape)
        ring.cursor = meta["cursor"]
        ring.count = meta["count"]
        return ring


#: Why a record whose ``visit_actions`` are era-absent is never trained on or
#: resumed into. One statement, shared by every door that refuses (F-41).
STALE_TARGETS = (
    "their visit_actions predate ring_format 2: they are root slot positions, not actions (F-41)"
)


#: Why a pre-format-3 record cannot train the policy under ``'improved'``, and
#: why a pre-format-3 ring is never resumed into (F-42).
NO_IMPROVED_TARGET = "they predate ring_format 3 and carry no π′ (F-42)"


def era_absent_on_disk(path: Path, name: str) -> tuple[int, int]:
    """``(records predating name's meaning, valid records)`` for a SAVED ring.

    Reads only ``meta.json`` and the per-record format column, so a door can
    refuse a stale ring before loading the whole of it — and before writing
    anything — rather than after.
    """
    meta = json.loads((path / "meta.json").read_text())
    count = int(meta["count"])
    formats = np.fromfile(path / "record_format.bin", dtype=np.int16, count=count)
    return int(np.count_nonzero(formats < _FIELD_MAP[name].since)), count


def bytes_per_record(cfg: Config, ring_format: int = RING_FORMAT) -> int:
    """Per-record footprint, computed rather than quoted from the docstring.

    ``ring_format`` prices an older layout: a ring saved at a declared format
    below 3 has no π′ fields, and its meta must not claim their bytes.
    """
    m = cfg.model
    improved = (IMPROVED_SLOTS * 4 * 2 + 8) if ring_format >= 3 else 0
    return improved + (
        m.seq_len * 2
        + m.max_sites * 2
        + 2  # length
        + 2  # n_sites
        + cfg.search.gumbel_m * 4  # visit_actions
        + cfg.search.gumbel_m * 4  # visit_counts
        + 4  # root_q
        + 1  # z
        + 1  # par_source
        + 2  # par
        + 2  # steps_remaining
        + 1  # depth
        + 1  # goal
        + 2  # record_format
        + 4  # absent_mask
        # + IMPROVED_SLOTS * 4 (improved_actions) + IMPROVED_SLOTS * 4
        # (improved_probs) + 8 (improved_tail, float64) from format 3 — above.
    )
