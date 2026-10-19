#!/usr/bin/env python3
"""GRAD-2 gated fusion (PI-approved 2026-09-08, method.md §2).

The static logaddexp-logit fusion is replaced by a small logistic gate learned
on the calibration split only:

  f = sigmoid( w·[logit_t, logit_s, |logit_t|, logit_t*logit_s, 1(|logit_t|<tau)] + b )

so the structural channel contributes when the text channel is uncertain.
Weights are frozen into the bundle's calibration.json; deployment never refits.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'gated_logit_v1'
FEATURES = ['logit_text', 'logit_struct', 'abs_logit_text',
            'logit_text_x_struct', 'text_uncertain_gate']
DEFAULT_TAU = 0.8
EPS = 1e-7


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def featurize(struct, text, tau=DEFAULT_TAU):
    lt = _logit(text)
    ls = _logit(struct)
    return np.column_stack([lt, ls, np.abs(lt), lt * ls,
                            (np.abs(lt) < tau).astype(float)])


def fit_gated(struct, text, y, tau=DEFAULT_TAU, C=1.0):
    from sklearn.linear_model import LogisticRegression
    X = featurize(struct, text, tau)
    m = LogisticRegression(C=C, max_iter=2000).fit(X, np.asarray(y, dtype=int))
    return {'version': VERSION, 'features': FEATURES, 'tau': tau, 'C': C,
            'coef': [float(w) for w in m.coef_[0]],
            'intercept': float(m.intercept_[0]),
            'n_fit': int(len(y)), 'n_pos': int(np.sum(y))}


def predict_gated(spec, struct, text):
    X = featurize(struct, text, spec.get('tau', DEFAULT_TAU))
    z = X @ np.asarray(spec['coef'], float) + float(spec['intercept'])
    return 1.0 / (1.0 + np.exp(-z))

def static_fuse(a, b, eps=EPS):
    a = min(max(float(a), eps), 1 - eps)
    b = min(max(float(b), eps), 1 - eps)
    la = math.log(a) - math.log1p(-a)
    lb = math.log(b) - math.log1p(-b)
    m = max(la, lb)
    return 1.0 / (1.0 + math.exp(-(m + math.log(math.exp(la - m)
                                                + math.exp(lb - m)))))


def apply(spec, struct, text):
    """Dispatch by spec; missing spec keeps the legacy static fusion."""
    if not spec:
        return static_fuse(struct, text)
    if spec.get('version') == VERSION:
        out = np.asarray(predict_gated(spec, struct, text)).reshape(-1)
        return float(out[0])
    if spec.get('version') == 'logaddexp_logit_v1':
        return static_fuse(struct, text)
    raise ValueError(f"unknown fusion spec version: {spec.get('version')!r}")
