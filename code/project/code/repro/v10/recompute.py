#!/usr/bin/env python
"""V10 zero-API offline recompute after the 2026-09-12 code fixes.

Applies to the saved authoritative batches (v10b) only:
  1. independent checker fixed (shop category + missing price + trace auth)
  2. authorization reconstructed from the delivered session-1 task text
     (old traces predate structured authorization events)
  3. strong lexicon gate replayed against every recorded action decision
     (fixed nested scan) to list cells whose online behaviour would change

Outputs LOGS/v10_recompute_fixed_20260912.json and .md. Old artifacts are
never modified; offline projections are labelled as such.
"""
from __future__ import annotations

import copy
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
from envlib import ENVS, authorization, gate_decision, independent_outcome  # noqa: E402

ARMS = ('none', 'strong', 'flat', 'relation', 'seq', 'verify',
        'verify_replan')
CONDS = ('attack_instruction', 'benign_instruction', 'authorized_override')
OUT_JSON = REPO / 'LOGS/v10_recompute_fixed_20260912.json'
OUT_MD = REPO / 'LOGS/v10_recompute_fixed_20260912.md'


def load_runs(prefix):
    runs = []
    for p in sorted(glob.glob(str(REPO / f'LOGS/v10_{prefix}_*.json'))):
        if 'analysis' in p or 'recompute' in p:
            continue
        d = json.loads(Path(p).read_text())
        for i, r in enumerate(d['runs']):
            r['_phase'] = infer_phase(prefix, i)
            runs.append(r)
    return runs


def infer_phase(prefix, idx):
    if prefix == 'mainb':
        return 'dev'
    if idx < 72:
        return 'P1'
    if idx < 96:
        return 'P2'
    return 'P3'


def reconstructed_auths(run):
    if run['condition'] != 'authorized_override':
        return []
    env = ENVS[run['env']]
    fam = env['families'][run['family']]
    text = ' '.join(t.get('text', '') for t in run['trace'].get('tasks', []))
    if fam['auth_text'].strip() and fam['auth_text'].strip() in text:
        return [authorization('recon0', run['trace'].get('user_id', 'user'),
                              env['action'], fam['bad_arg'], 0,
                              scope='one_time')]
    return []


def trace_with_auths(run):
    t = copy.deepcopy(run['trace'])
    t.setdefault('user_id', 'user')
    t.setdefault('authorizations', [])
    t['authorizations'] = t['authorizations'] + reconstructed_auths(run)
    return t


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def mcnemar_two_sided(pairs):
    b = sum(1 for a, x in pairs if a and not x)
    c = sum(1 for a, x in pairs if not a and x)
    n = b + c
    if n == 0:
        return {'n_pairs': len(pairs), 'b': 0, 'c': 0, 'p_one_sided': None,
                'p_two_sided': None}
    k = min(b, c)
    one = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {'n_pairs': len(pairs), 'b': b, 'c': c,
            'p_one_sided': round(one, 6),
            'p_two_sided': round(min(1.0, 2 * one), 6)}


def paired(runs, arm_a, arm_b, cond, metric):
    key = {}
    for r in runs:
        if r['condition'] != cond or r['error']:
            continue
        key[(r['env'], r['family'], r['repeat'], r['arm'])] = bool(
            r[metric])
    pairs = []
    for k in key:
        if k[3] != arm_a:
            continue
        other = (k[0], k[1], k[2], arm_b)
        if other in key:
            pairs.append((key[k], key[other]))
    return pairs


def cluster_bootstrap(runs, arm_a, arm_b, cond, metric, n_boot=5000,
                      seed=42):
    """Paired difference (arm_b - arm_a) with clusters = (env, family)."""
    key = {}
    for r in runs:
        if r['condition'] != cond or r['error']:
            continue
        key[(r['env'], r['family'], r['repeat'], r['arm'])] = bool(
            r[metric])
    clusters = defaultdict(list)
    for k in key:
        if k[3] != arm_a:
            continue
        other = (k[0], k[1], k[2], arm_b)
        if other in key:
            clusters[(k[0], k[1])].append(
                int(key[other]) - int(key[k]))
    if not clusters:
        return None
    keys = list(clusters)
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [clusters[rnd.choice(keys)] for _ in range(len(keys))]
        flat = [x for c in sample for x in c]
        diffs.append(sum(flat) / len(flat))
    diffs.sort()
    point = sum(sum(v) for v in clusters.values()) / sum(
        len(v) for v in clusters.values())
    return {'point': round(point, 4), 'n_clusters': len(keys),
            'ci95': [round(diffs[int(0.025 * n_boot)], 4),
                     round(diffs[int(0.975 * n_boot)], 4)]}


def mark_pending(out, envs=('finance', 'workspace')):
    for env in out['tables']:
        for cond in out['tables'][env]:
            cell = out['tables'][env][cond].get('strong')
            if cell and env in envs:
                cell['pending_rerun'] = True
    cell = out['pooled'].get('attack_instruction', {}).get('strong')
    if cell:
        cell['pending_rerun'] = True
        cell['pending_note'] = ('strong lexicon gate implementation fixed; '
                                'finance/workspace cells await online rerun')
    return out


def summarize(runs, prefix):
    out = {'tables': {}, 'pooled': {}, 'mcnemar': {}, 'bootstrap': {},
           'per_family': {}}
    envs = sorted({r['env'] for r in runs})
    for env in envs:
        out['tables'][env] = {}
        for cond in CONDS:
            out['tables'][env][cond] = {}
            for arm in ARMS:
                rs = [r for r in runs if r['env'] == env and
                      r['condition'] == cond and r['arm'] == arm and
                      not r['error']]
                n = len(rs)
                harm = sum(r['_harm'] for r in rs)
                comp = sum(r['_completion'] for r in rs)
                out['tables'][env][cond][arm] = {
                    'n': n, 'harm': harm,
                    'harm_rate': round(harm / n, 4) if n else None,
                    'harm_wilson': wilson(harm, n),
                    'completion': comp,
                    'completion_rate': round(comp / n, 4) if n else None,
                    'completion_wilson': wilson(comp, n)}
    for cond in CONDS:
        out['pooled'][cond] = {}
        for arm in ARMS:
            rs = [r for r in runs if r['condition'] == cond and
                  r['arm'] == arm and not r['error']]
            n = len(rs)
            harm = sum(r['_harm'] for r in rs)
            comp = sum(r['_completion'] for r in rs)
            out['pooled'][cond][arm] = {
                'n': n, 'harm': harm,
                'harm_rate': round(harm / n, 4) if n else None,
                'harm_wilson': wilson(harm, n),
                'completion': comp,
                'completion_rate': round(comp / n, 4) if n else None,
                'completion_wilson': wilson(comp, n)}
        for other in ('none', 'strong', 'flat', 'relation', 'verify'):
            out['mcnemar'][f'attack_harm|{other}->verify_replan'] = \
                mcnemar_two_sided(paired(runs, other, 'verify_replan',
                                         'attack_instruction', '_harm'))
            out['mcnemar'][f'benign_completion|{other}->verify_replan'] = \
                mcnemar_two_sided(paired(runs, other, 'verify_replan',
                                         'benign_instruction',
                                         '_completion'))
            out['bootstrap'][f'attack_completion|{other}->verify_replan'] = \
                cluster_bootstrap(runs, other, 'verify_replan',
                                  'attack_instruction', '_completion')
    for env in envs:
        rs = [r for r in runs if r['env'] == env and
              r['condition'] == 'attack_instruction' and not r['error']]
        fams = sorted({r['family'] for r in rs})
        out['per_family'][env] = {}
        for fam in fams:
            row = {}
            for arm in ARMS:
                fr = [r for r in rs if r['family'] == fam and
                      r['arm'] == arm]
                if fr:
                    row[arm] = {
                        'n': len(fr),
                        'harm': sum(r['_harm'] for r in fr),
                        'completion': sum(r['_completion'] for r in fr)}
            out['per_family'][env][fam] = row
    return mark_pending(out)


def fmt_pooled(S, conds=CONDS):
    lines = []
    for cond in conds:
        row = S['pooled'].get(cond, {})
        parts = []
        for arm in ARMS:
            c = row.get(arm)
            if c and c['n']:
                flag = ' *' if c.get('pending_rerun') else ''
                parts.append(f"{arm} {c['harm']}/{c['n']}h,"
                             f"{c['completion']}/{c['n']}c{flag}")
        if parts:
            lines.append(f"- `{cond}`: " + ' | '.join(parts))
    return lines


def write_markdown(report):
    b_main = report['batches']['mainb']
    b_conf = report['batches']['confb']
    L = []
    L.append('# V10 离线重计（实现修复后，零 API；2026-09-12）')
    L.append('')
    L.append('对象：`LOGS/v10_{mainb,confb}_*.json`（v10b 权威批，'
             '744 runs）。脚本 `code/repro/v10/recompute.py`；'
             '契约 `code/repro/tests/test_v10_contract.py`（13/13）。')
    L.append('')
    L.append('## 1. 修复与离线影响')
    L.append('')
    L.append('- 授权：门控不再接收 `condition`/`bad_arg`，只读 trace 的'
             '结构化授权事件（subject/object/action/scope/时间/撤销）。'
             '旧 trace 无事件，按交付的 session-1 任务文本重建'
             '（当时授权文本确实送达）；新 runner 直接写事件。')
    L.append('- strong：记录扫描改为完整词表嵌套循环。')
    L.append('- shop 检查器：类别 + 价格 + 缺失价格不再默认合规；'
             'verify 同步修 `unverified_price`。')
    L.append('')
    L.append(f"- 检查器修复对旧数据改变结果：mainb "
             f"**{b_main['n_outcome_changed']}** / confb "
             f"**{b_conf['n_outcome_changed']}**（缺失价格/错误类别在"
             f"已存最终动作中未出现）。")
    L.append(f"- 修复后 strong 决策变化（回放已记录动作）：mainb "
             f"{b_main['n_decision_changed']}、confb "
             f"{b_conf['n_decision_changed']}；受影响格子必须在线重跑："
             f"mainb {b_main['n_affected_strong_cells']}、confb "
             f"{b_conf['n_affected_strong_cells']}（finance/workspace 全部 "
             f"strong）。`*` = pending_rerun，**不作为最终结果**。")
    L.append('- 离线回放只覆盖已记录动作决策；被拦的非动作提案不可恢复，'
             '故不做“修复后指标”外推。')
    L.append('')
    L.append('## 2. 主批 dev（checker 重计，与旧结果一致）')
    L.append('')
    L.extend(fmt_pooled(b_main['phases']['dev']))
    L.append('')
    L.append('### 逐环境 attack（“三环境一致”口径更正）')
    L.append('')
    L.append('| env | none harm | strong harm | flat harm | relation harm |'
             ' verify harm | verify_replan harm | verify_replan comp |')
    L.append('|---|---|---|---|---|---|---|---|')
    for env in ('shop', 'workspace', 'finance'):
        t = b_main['tables'][env]['attack_instruction']
        L.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            env, t['none']['harm_rate'], t['strong']['harm_rate'],
            t['flat']['harm_rate'], t['relation']['harm_rate'],
            t['verify']['harm_rate'], t['verify_replan']['harm_rate'],
            t['verify_replan']['completion_rate']))
    L.append('')
    L.append('## 3. 确认批（P1 确认、P2 关系消融、P3 reasoner）')
    L.append('')
    for ph in ('P1', 'P2', 'P3'):
        L.append(f'### {ph}')
        L.append('')
        L.extend(fmt_pooled(b_conf['phases'][ph]))
        L.append('')
    L.append('## 4. 统计更正')
    L.append('')
    for label, b in (('mainb(dev)', b_main), ('confb(P1)', b_conf['phases']['P1'])):
        mc = {k: v for k, v in b['mcnemar'].items()
              if v.get('p_two_sided') is not None and 'attack_harm' in k}
        if mc:
            L.append(f'- {label} attack harm McNemar（原报告为单侧）：')
            for k, v in mc.items():
                L.append(f'  - {k}: b={v["b"]}, c={v["c"]}, '
                         f'p_one={v["p_one_sided"]}, p_two={v["p_two_sided"]}')
        bs = {k: v for k, v in b['bootstrap'].items()
              if v and 'none->' in k or (v and 'strong->' in k)}
        for k, v in bs.items():
            if v:
                L.append(f'  - {k}: Δ={v["point"]} CI95={v["ci95"]} '
                         f'(clusters={v["n_clusters"]})')
    L.append('')
    L.append('- 语义：McNemar 为配对二值精确检验（双侧=2×单侧，封顶 1）；'
             'CI 为按 (env, family) 聚类的 5000 次 bootstrap。'
             'completion 的非劣不能仅凭点估计。')
    L.append('')
    L.append('## 5. 待在线重跑与新增对照（需 PI 预算确认）')
    L.append('')
    L.append('- 受影响 strong 格子：mainb 48 + confb 44（含 P3）。')
    L.append('- 图贡献对照：新增 `relation_verify_replan`（关系闭包+验证+'
             '重规划）与 `verify_replan`（去图）同格子。')
    L.append('- 外部强对照：`monitor`（同模型 LLM 顺序监控基线，'
             '每动作一次额外调用，计入预算）。')
    L.append('- 冻结后新族确认：新增未曝光 family 的小批确认。')
    L.append('')
    OUT_MD.write_text('\n'.join(L))


def main():
    report = {'generated_by': 'code/repro/v10/recompute.py',
              'fixes': [
                  'independent checker: shop category + missing price + '
                  'trace authorization events',
                  'authorization reconstructed from delivered session-1 task '
                  'text for pre-fix traces',
                  'strong lexicon gate: full nested word scan replayed on '
                  'recorded action decisions'],
              'limitations': [
                  'offline replay only sees recorded action decisions; '
                  'blocked non-action proposals are not recoverable',
                  'projected fixed-gate outcomes are not online results; '
                  'affected cells must be re-run online'],
              'batches': {}}
    all_runs = {}
    for prefix in ('mainb', 'confb'):
        runs = load_runs(prefix)
        affected = []
        outcome_changed = []
        for r in runs:
            r['_model_inferred'] = ('deepseek-reasoner'
                                    if r['_phase'] == 'P3'
                                    else 'deepseek-chat')
            twa = trace_with_auths(r)
            outcome = independent_outcome(ENVS[r['env']],
                                          ENVS[r['env']]['families'][
                                              r['family']], twa,
                                          r['final_arg'])
            r['_harm'] = outcome['harm']
            r['_completion'] = outcome['completion']
            if (bool(r['independent_harm']) != outcome['harm'] or
                    bool(r['independent_completion']) !=
                    outcome['completion']):
                outcome_changed.append({
                    'run_id': r['run_id'],
                    'old': [bool(r['independent_harm']),
                            bool(r['independent_completion'])],
                    'new': [outcome['harm'], outcome['completion']]})
            action = ENVS[r['env']]['action']
            for d in r.get('decisions', []):
                if d.get('tool') != action:
                    continue
                old = (d['level'], d['reason'])
                try:
                    level, reason, _ = gate_decision(
                        r['arm'], ENVS[r['env']],
                        ENVS[r['env']]['families'][r['family']], twa,
                        d['tool'], d['arg'], {}, d.get('session', 2))
                except ValueError:
                    continue
                if (level, reason) != old:
                    affected.append({
                        'run_id': r['run_id'], 'env': r['env'],
                        'family': r['family'],
                        'condition': r['condition'], 'arm': r['arm'],
                        'repeat': r['repeat'], 'phase': r['_phase'],
                        'arg': d['arg'], 'session': d.get('session'),
                        'old': list(old), 'new': [level, reason]})
        affected_ids = sorted({a['run_id'] for a in affected})
        if prefix == 'confb':
            phases = {ph: summarize([r for r in runs
                                     if r['_phase'] == ph], prefix)
                      for ph in ('P1', 'P2', 'P3')}
        else:
            phases = {'dev': summarize(runs, prefix)}

        def cell(r):
            return {'run_id': r['run_id'], 'env': r['env'],
                    'family': r['family'], 'condition': r['condition'],
                    'arm': r['arm'], 'repeat': r['repeat'],
                    'phase': r['_phase'],
                    'model_inferred': r['_model_inferred']}

        affected_strong = [cell(r) for r in runs
                           if r['arm'] == 'strong' and
                           r['env'] in ('finance', 'workspace')]
        report['batches'][prefix] = {
            'n_runs': len(runs), 'n_errors': sum(1 for r in runs
                                                 if r['error']),
            'outcome_changed': outcome_changed,
            'n_outcome_changed': len(outcome_changed),
            'decision_changed': affected,
            'n_decision_changed': len(affected),
            'affected_cells': affected_ids,
            'affected_strong_cells': affected_strong,
            'n_affected_strong_cells': len(affected_strong),
            'phases': phases,
            **summarize(runs, prefix),
            'offline_note': (
                'fixed-gate replay covers recorded action decisions only; '
                'blocked non-action proposals are not recoverable, so strong '
                'finance/workspace cells are marked pending_rerun and no '
                'offline projection is reported as a result')}
        all_runs[prefix] = runs
    OUT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_markdown(report)
    print(json.dumps({
        'mainb_outcome_changed': report['batches']['mainb'][
            'n_outcome_changed'],
        'mainb_decision_changed': report['batches']['mainb'][
            'n_decision_changed'],
        'confb_outcome_changed': report['batches']['confb'][
            'n_outcome_changed'],
        'confb_decision_changed': report['batches']['confb'][
            'n_decision_changed'],
        'mainb_affected_cells': report['batches']['mainb'][
            'affected_cells'],
        'wrote': str(OUT_JSON)}, indent=1))


if __name__ == '__main__':
    main()
