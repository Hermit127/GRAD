"""Schemas for the programmatic antibody state S=(A,T,O)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Antibody:
    pattern: list = field(default_factory=list)      # observable atom condition
    queries: list = field(default_factory=list)      # verification queries (reserved)
    exceptions: list = field(default_factory=list)   # tolerance conditions
    action: str = 'block'                            # allow|block|verify
    scope: dict = field(default_factory=dict)        # tools/validity (reserved)
    evidence: list = field(default_factory=list)     # source case ids / spans

    def to_dict(self):
        return {'pattern': self.pattern, 'queries': self.queries,
                'exceptions': self.exceptions, 'action': self.action,
                'scope': self.scope, 'evidence': self.evidence}


@dataclass
class Decision:
    scenario_id: str
    phase: str          # pre | post
    system: str
    decision: str       # allow|block
    label: int          # 1 attack (should block), 0 benign (should allow)
    fired: Any = None
    atoms: dict = field(default_factory=dict)
