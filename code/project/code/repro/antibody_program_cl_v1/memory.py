"""Case memory (N1 control): observable-feature 1-NN over revealed feedback."""
from __future__ import annotations

from .dsl import atoms_of

FEATURE_ATOMS = ['untrusted_read_sess', 'param_from_untrusted',
                 'read_but_not_param', 'span_from_untrusted',
                 'param_heuristic_guess', 'no_param_source']


def vector(trace, proposed):
    av = atoms_of(trace, proposed)
    return [1.0 if av[a] else 0.0 for a in FEATURE_ATOMS]


class CaseMemory:
    def __init__(self):
        self.cases = []

    def add(self, trace, proposed, label):
        self.cases.append((vector(trace, proposed), int(label)))

    def predict(self, trace, proposed, fallback='allow'):
        if not self.cases:
            return fallback
        v = vector(trace, proposed)
        best, bl = None, None
        for cv, lab in self.cases:
            d = sum((a - b) ** 2 for a, b in zip(v, cv))
            if best is None or d < best:
                best, bl = d, lab
        return 'block' if bl == 1 else 'allow'
