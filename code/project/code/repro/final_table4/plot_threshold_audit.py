#!/usr/bin/env python
"""Plot the threshold audit distributions (run with the ipiguard venv).

Reads the audit JSONL files and writes distributions.png next to them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path('.')
AUDIT = REPO / 'LOGS/final_table4/threshold_audit_20260918'
FROZEN = 0.9999856948852539


def load(name):
    return [json.loads(l) for l in (AUDIT / name).read_text().splitlines()
            if l.strip()]


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) v2 dev/test by class
ax = axes[0][0]
for role, ls in (('dev', '-'), ('test', '--')):
    rows = load(f'scores_v2_{role}.jsonl')
    for lab, color in ((0, 'tab:blue'), (1, 'tab:red')):
        xs = [logit(r['p']) for r in rows if r['label'] == lab]
        if xs:
            ax.hist(xs, bins=20, alpha=0.35, linestyle=ls, histtype='step',
                    linewidth=2, color=color,
                    label=f"{role} {'mal' if lab else 'benign'}")
ax.axvline(logit(FROZEN), color='k', linewidth=1)
ax.set_title('GRAD v2 splits (old deepseek-chat traces)')
ax.set_xlabel('logit(p)')
ax.legend(fontsize=7)

# (b,c) Table 4 none arm per backbone
for idx, bb in enumerate(('deepseek-flash', 'glm-5.3-flash')):
    ax = axes[0][1] if idx == 0 else axes[1][0]
    rows = [r for r in load('scores_table4.jsonl')
            if r['backbone'] == bb and r['arm'] == 'none']
    groups = [
        ('harmful', [r['p'] for r in rows if r['condition'] ==
                     'attack_instruction' and r.get('harm_a1')], 'tab:red'),
        ('attack-safe', [r['p'] for r in rows if r['condition'] ==
                         'attack_instruction' and r.get('harm_a1') is False],
         'tab:orange'),
        ('benign', [r['p'] for r in rows if r['condition'] ==
                    'benign_instruction'], 'tab:blue'),
    ]
    for name, ps, color in groups:
        if ps:
            ax.scatter([logit(p) for p in ps],
                       np.random.default_rng(0).normal(0, 0.02, len(ps)) +
                       {'harmful': 3, 'attack-safe': 2, 'benign': 1}[name],
                       alpha=0.6, s=28, color=color, label=name)
    ax.axvline(logit(FROZEN), color='k', linewidth=1, label='frozen thr')
    ax.set_title(f'Table 4 None arm — {bb}')
    ax.set_xlabel('logit(p)')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['benign', 'attack-safe', 'harmful'])
    ax.legend(fontsize=7)

# (d) zoomed upper tail for both backbones (all arms)
ax = axes[1][1]
for bb, marker in (('deepseek-flash', 'o'), ('glm-5.3-flash', '^')):
    rows = load('scores_table4.jsonl')
    for label, cond, harm, color in (
            ('harm', 'attack_instruction', True, 'tab:red'),
            ('benign', 'benign_instruction', None, 'tab:blue')):
        xs = [1 - r['p'] for r in rows
              if r['backbone'] == bb and r['condition'] == cond
              and (harm is None or r.get('harm_a1') is harm)]
        if xs:
            ax.scatter(xs, np.random.default_rng(1).normal(
                0, 0.15, len(xs)) + (2 if bb == 'glm-5.3-flash' else 1),
                marker=marker, alpha=0.5, s=22, color=color,
                label=f'{bb} {label}')
ax.axvline(1 - FROZEN, color='k', linewidth=1)
ax.set_xscale('log')
ax.set_title('distance below 1 (log scale), all arms')
ax.set_yticks([1, 2])
ax.set_yticklabels(['DS', 'GLM'])
ax.legend(fontsize=7)

fig.tight_layout()
out = AUDIT / 'distributions.png'
fig.savefig(out, dpi=150)
print('wrote', out)
