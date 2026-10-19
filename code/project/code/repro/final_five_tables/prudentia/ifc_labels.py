"""IFC label lattices for the Prudentia reproduction.

Paper anchors:
  - Prudentia section 2, "Security labels and how to propagate them":
    integrity lattice {T, U} with T <= U; confidentiality readers lattice
    (powerset of users) with join = set intersection.
  - Design ported from the public fides tutorial notebook
    (https://github.com/microsoft/fides, Tutorial.ipynb); no notebook code is
    imported, only the lattice semantics are reimplemented.

Convention:
  Integrity rank: TRUSTED(0) <= UNTRUSTED(1); join = max rank.
  Readers: a label describes the set of *authorized readers*; join is set
  intersection (fewer readers = higher confidentiality). Ordering is inverted:
  X <= Y iff X.readers is a superset of Y.readers. Readers(frozenset()) is the
  top (most confidential); None means "unspecified/top" (pi = (T, top)).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional


class Integrity(Enum):
    """Two-element integrity lattice from Prudentia section 2."""

    TRUSTED = "T"
    UNTRUSTED = "U"

    @property
    def rank(self) -> int:
        return 0 if self is Integrity.TRUSTED else 1

    def leq(self, other: "Integrity") -> bool:
        return self.rank <= other.rank

    def join(self, other: "Integrity") -> "Integrity":
        return self if self.rank >= other.rank else other


@dataclass(frozen=True)
class Readers:
    """Confidentiality label as a readers powerset (join = intersection)."""

    users: FrozenSet[str]

    def leq(self, other: "Readers") -> bool:
        return self.users >= other.users

    def join(self, other: "Readers") -> "Readers":
        return Readers(self.users & other.users)

    @classmethod
    def top(cls) -> "Readers":
        return cls(frozenset())


@dataclass(frozen=True)
class ContextLabel:
    """Product of the integrity lattice and the readers lattice."""

    integrity: Integrity = Integrity.TRUSTED
    readers: Optional[Readers] = None

    def is_trusted(self) -> bool:
        return self.integrity is Integrity.TRUSTED

    def join(self, other: "ContextLabel") -> "ContextLabel":
        readers = None
        if self.readers is not None and other.readers is not None:
            readers = self.readers.join(other.readers)
        return ContextLabel(self.integrity.join(other.integrity), readers)


def join_labels(labels) -> ContextLabel:
    """Join over an iterable of ContextLabel (empty -> trusted/top)."""

    result = ContextLabel()
    for label in labels:
        result = result.join(label)
    return result
