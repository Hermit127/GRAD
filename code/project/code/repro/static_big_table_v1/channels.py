"""Channel training/scoring for the static table (frozen GRAD-2 components).

All research-code dependencies are imported from the immutable snapshot.
Channels:
  text   -> BoW([a-z0-9]+, min_count>=3) + XGBoost(200, depth 6, lr 0.1)
  struct -> 2-layer GINEConv (hidden 64) on FD=56 full-observable features
  sentry -> Agent-Sentry L1 local repro (17-dim per-action XGBoost, max-pool)
  binary -> AgentTrust / NeuroTaint / L1-regex / AgentArmor / 1-NN / node-count
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

from common import REPO, SNAPSHOT

if str(SNAPSHOT) not in sys.path:
    sys.path.insert(0, str(SNAPSHOT))


def enforce_snapshot() -> None:
    """Repo code paths are prepended by legacy modules (memorygraft_to_graph /
    train_gnn / mg_cv_grouped). Remove them so every research dependency
    resolves to the immutable snapshot (spec §1.3)."""
    bad = {str(REPO / 'code'), str(REPO / 'code/repro'),
           'code', 'code/repro', './code', './code/repro'}
    forbidden_resolved = {(REPO / 'code').resolve(),
                          (REPO / 'code/repro').resolve()}
    for p in list(sys.path):
        if p in bad:
            while p in sys.path:
                sys.path.remove(p)
            continue
        try:
            if Path(p or '.').resolve() in forbidden_resolved:
                while p in sys.path:
                    sys.path.remove(p)
        except OSError:
            pass
    while str(SNAPSHOT) in sys.path:
        sys.path.remove(str(SNAPSHOT))
    sys.path.insert(0, str(SNAPSHOT))


enforce_snapshot()

SNAPSHOT_MODULES = ('graph_featurizer', 'mg_cv_grouped', 'gated_fusion',
                    'agent_sentry_repro', 'agentarmor_repro',
                    'compare_all_baselines', 'a3_feature_bench', 'train_gnn',
                    'memorygraft_to_graph', 'permission_spec')


def verify_snapshot_modules() -> dict:
    """Assert every loaded research module resolves to the snapshot copy."""
    enforce_snapshot()
    out = {}
    for name in SNAPSHOT_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            out[name] = 'not_loaded'
            continue
        path = Path(getattr(mod, '__file__', ''))
        out[name] = str(path)
        if path.parent.resolve() != SNAPSHOT.resolve():
            raise RuntimeError(
                f'{name} loaded from {path}, not the frozen snapshot {SNAPSHOT}')
    return out

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from graph_featurizer import to_data  # noqa: E402
from mg_cv_grouped import Net as StructNet  # noqa: E402

enforce_snapshot()
from gated_fusion import fit_gated, predict_gated, static_fuse  # noqa: E402

TOK = re.compile(r'[a-z0-9]+')
TEXT_CONFIG = {'tokenizer': '[a-z0-9]+', 'min_count': 3,
               'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
               'scale_pos_weight': 'n_neg/n_pos'}
STRUCT_CONFIG = {'hidden': 64, 'layers': 2, 'edge_dim': 8, 'epochs': 100,
                 'batch_size': 32, 'lr': 1e-3, 'weight_decay': 5e-4,
                 'checkpoint_selection': 'dev_auroc'}
VIEW = 'full_observable'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def text_of(g: dict) -> str:
    return ' '.join(str((n.get('features') or {}).get('subtask', ''))
                    for n in g.get('nodes', []))


class TextChannel:
    config = TEXT_CONFIG
    DETERMINISTIC = True
    RANDOM_STATE = 42

    @staticmethod
    def fit(graphs: list[dict], seed: int) -> dict:
        vocab = Counter()
        for g in graphs:
            vocab.update(TOK.findall(text_of(g).lower()))
        keep = sorted(w for w, c in vocab.items() if c >= TEXT_CONFIG['min_count'])
        idx = {w: i for i, w in enumerate(keep)}

        def fe(g):
            v = [0] * len(idx)
            for w in TOK.findall(text_of(g).lower()):
                if w in idx:
                    v[idx[w]] += 1
            return v
        ys = [int(g['label']) for g in graphs]
        na = sum(ys)
        nb = len(ys) - na
        model = xgb.XGBClassifier(
            n_estimators=TEXT_CONFIG['n_estimators'],
            max_depth=TEXT_CONFIG['max_depth'],
            learning_rate=TEXT_CONFIG['learning_rate'],
            random_state=42,
            scale_pos_weight=nb / max(na, 1),
            eval_metric='logloss')
        model.fit([fe(g) for g in graphs], ys)
        return {'vocab': keep, 'idx': idx, 'model': model, 'seed': seed,
                'n_train': len(graphs), 'n_pos': na, 'n_neg': nb,
                'random_state': TextChannel.RANDOM_STATE,
                'deterministic': True}

    @staticmethod
    def score(channel: dict, graphs: list[dict]) -> list[float]:
        idx = channel['idx']
        if not idx:
            return [0.5] * len(graphs)

        def fe(g):
            v = [0] * len(idx)
            for w in TOK.findall(text_of(g).lower()):
                if w in idx:
                    v[idx[w]] += 1
            return v
        return [float(x) for x in
                channel['model'].predict_proba([fe(g) for g in graphs])[:, 1]]


class StructChannel:
    config = STRUCT_CONFIG

    @staticmethod
    def _data(graphs: list[dict]):
        out, status = [], []
        for g in graphs:
            d = to_data(g, drop=(), mode='full')
            if d is None:
                status.append('no_nodes_after_featurize')
            else:
                out.append(d)
                status.append('ok')
        return out, status

    @staticmethod
    def fit(train_graphs: list[dict], dev_graphs: list[dict], seed: int) -> dict:
        set_seed(seed)
        trd, tr_status = StructChannel._data(train_graphs)
        vad, va_status = StructChannel._data(dev_graphs)
        tr_ys = [int(d.y.item()) for d in trd]
        va_ys = [int(d.y.item()) for d in vad]
        model = StructNet()
        opt = torch.optim.Adam(model.parameters(), lr=STRUCT_CONFIG['lr'],
                               weight_decay=STRUCT_CONFIG['weight_decay'])
        best = {'dev_auroc': -1.0, 'epoch': -1, 'state': None}
        for ep in range(STRUCT_CONFIG['epochs']):
            model.train()
            perm = list(range(len(trd)))
            random.shuffle(perm)
            for i in range(0, len(perm), STRUCT_CONFIG['batch_size']):
                batch = Batch.from_data_list(
                    [trd[j] for j in perm[i:i + STRUCT_CONFIG['batch_size']]])
                opt.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.edge_attr,
                               batch.batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch.y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                ps = []
                for i in range(0, len(vad), 64):
                    b = Batch.from_data_list(vad[i:i + 64])
                    ps.extend(torch.sigmoid(model(
                        b.x, b.edge_index, b.edge_attr, b.batch)).tolist())
            if len(set(va_ys)) > 1:
                auc = roc_auc_score(va_ys, ps)
            else:
                auc = float('nan')
            if not np.isnan(auc) and auc > best['dev_auroc']:
                best = {'dev_auroc': float(auc), 'epoch': ep,
                        'state': {k: v.clone() for k, v in model.state_dict().items()}}
        if best['state'] is None:
            best['state'] = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best['state'])
        model.eval()
        return {'state': best['state'], 'dev_auroc': best['dev_auroc'],
                'epoch': best['epoch'], 'seed': seed,
                'n_train': len(trd), 'n_pos': sum(tr_ys),
                'n_neg': len(tr_ys) - sum(tr_ys),
                'train_status': tr_status, 'dev_status': va_status}

    @staticmethod
    def score(channel: dict, graphs: list[dict]) -> tuple[list[float], list[str]]:
        model = StructNet()
        model.load_state_dict(channel['state'])
        model.eval()
        ds, status = StructChannel._data(graphs)
        scores = []
        di = 0
        for st in status:
            if st == 'ok':
                d = ds[di]
                di += 1
                with torch.no_grad():
                    scores.append(float(torch.sigmoid(model(
                        d.x, d.edge_index, d.edge_attr,
                        torch.zeros(d.x.size(0), dtype=torch.long))).item()))
            else:
                scores.append(float('nan'))
        return scores, status


def native_model_hash(baseline: str, train_graphs: list[dict] | None = None) -> str:
    """Stable implementation identity for binary/native baselines (S4)."""
    from common import sha256_file, sha256_json
    files = {
        'agenttrust': [SNAPSHOT / 'compare_all_baselines.py',
                       REPO / 'ref/AgentTrust/src/agent_trust/core/interceptor.py',
                       REPO / 'ref/AgentTrust/src/agent_trust/core/riskchain.py'],
        'neurotaint': [SNAPSHOT / 'compare_all_baselines.py'],
        'l1_regex': [SNAPSHOT / 'compare_all_baselines.py'],
        'aegis22': [SNAPSHOT / 'a3_feature_bench.py'],
        'agentarmor': [SNAPSHOT / 'agentarmor_repro.py'],
        'agentarmor_drop': [SNAPSHOT / 'agentarmor_repro.py'],
        'knn_tool_1nn': [REPO / 'code/repro/static_big_table_v1/channels.py'],
        'node_count': [REPO / 'code/repro/static_big_table_v1/channels.py'],
        'agent_sentry_l1': [SNAPSHOT / 'agent_sentry_repro.py'],
    }
    payload = {'baseline': baseline, 'files': {}}
    for p in files.get(baseline, []):
        if Path(p).exists():
            payload['files'][str(Path(p).name)] = sha256_file(p)
    if baseline == 'agent_sentry_l1' and train_graphs is not None:
        payload['train_input_sha256'] = sha256_json(
            sorted(observable_key(g) for g in train_graphs))
        payload['config'] = {'n_estimators': 100, 'max_depth': 4,
                             'learning_rate': 0.1, 'random_state': 42,
                             'theta_H': 0.66}
    if baseline in ('knn_tool_1nn', 'node_count'):
        payload['deterministic'] = True
    return sha256_json(payload)


def observable_key(g: dict) -> str:
    from common import sha256_json
    obs = {'nodes': [{'type': n.get('type'),
                      'subtask': str((n.get('features') or {}).get('subtask', ''))}
                     for n in g.get('nodes', [])],
           'edges': [{'s': e.get('source'), 't': e.get('target'), 'type': e.get('type')}
                     for e in g.get('edges', [])]}
    return sha256_json(obs)


def sentry_scores(train_graphs: list[dict], test_graphs: list[dict]):
    """Agent-Sentry L1 local repro; deterministic (random_state=42 inside)."""
    ys = {int(g['label']) for g in train_graphs}
    if len(ys) < 2:
        return ([float('nan')] * len(test_graphs),
                ['sentry_single_class_train'] * len(test_graphs))
    enforce_snapshot()
    from agent_sentry_repro import fit_predict
    res = fit_predict(train_graphs, test_graphs)
    if res is None:
        return [float('nan')] * len(test_graphs), ['sentry_no_features'] * len(test_graphs)
    ys, ps, gpred = res
    status = ['ok'] * len(ps)
    return [float(p) for p in ps], status


def _enable_reference_baselines() -> None:
    """External baseline code lives in ref/AgentTrust (official source, not our
    research code); add its paths so the snapshot copy of
    compare_all_baselines.py resolves imports without falling back to repo code."""
    for rel in ('ref/AgentTrust/src', 'ref/AgentTrust'):
        p = str(REPO / rel)
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)


def native_provenance() -> dict:
    enforce_snapshot()
    out = {}
    for name in ('compare_all_baselines', 'agentarmor_repro', 'a3_feature_bench',
                 'agent_sentry_repro'):
        mod = sys.modules.get(name)
        out[name] = getattr(mod, '__file__', 'not_loaded') if mod else 'not_loaded'
    mod = sys.modules.get('agent_trust.core.interceptor')
    out['agent_trust.interceptor'] = getattr(mod, '__file__', 'not_loaded') if mod else 'not_loaded'
    return out


def binary_baselines(test_graphs: list[dict]) -> dict[str, list[int]]:
    """Native binary baselines on one split (no training)."""
    enforce_snapshot()
    _enable_reference_baselines()
    from compare_all_baselines import (baseline_agenttrust,
                                       baseline_neurotaint,
                                       baseline_l1_regex)
    from agentarmor_repro import agentarmor_graph, NO_DANGER_SET
    from a3_feature_bench import AEGIS_PATTERNS
    import re as _re

    def aegis22(g):
        for n in g.get('nodes', []):
            text = str((n.get('features') or {}).get('subtask', ''))
            for _, pat in AEGIS_PATTERNS:
                if _re.search(pat, text):
                    return 1
        return 0

    return {
        'agenttrust': [int(x) for x in baseline_agenttrust(test_graphs)],
        'neurotaint': [int(x) for x in baseline_neurotaint(test_graphs)],
        'l1_regex': [int(x) for x in baseline_l1_regex(test_graphs)],
        'aegis22': [aegis22(g) for g in test_graphs],
        'agentarmor': [int(x) for x in
                       (agentarmor_graph(g) for g in test_graphs)],
        'agentarmor_drop': [int(x) for x in
                            (agentarmor_graph(g, NO_DANGER_SET) for g in test_graphs)],
    }


def knn_tool_1nn(train_graphs: list[dict], test_graphs: list[dict]) -> list[int]:
    """Historical q5b tool-type-set 1-NN (binary, native)."""
    cnt = Counter((tuple(sorted(n['type'] for n in g['nodes'])), int(g['label']))
                  for g in train_graphs)

    def sim(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    preds = []
    for g in test_graphs:
        ts = tuple(sorted(n['type'] for n in g['nodes']))
        best_t, best_s = 0, -1.0
        for (tt, lab), _ in cnt.items():
            s = sim(ts, tt)
            if s > best_s:
                best_s, best_t = s, lab
        preds.append(int(best_t))
    return preds


def node_count_pred(train_graphs: list[dict], test_graphs: list[dict]) -> list[int]:
    """Historical q5b node-count threshold selected on train max F1 (binary)."""
    from sklearn.metrics import f1_score
    ys = [int(g['label']) for g in train_graphs]
    best = (0.0, 2)
    for t in range(2, 30):
        f = f1_score(ys, [1 if len(g['nodes']) > t else 0 for g in train_graphs])
        if f > best[0]:
            best = (f, t)
    return [1 if len(g['nodes']) > best[1] else 0 for g in test_graphs], best[1]


def position_shortcut(test_graphs: list[dict]) -> list[float]:
    """A3 diagnostic: max(source_position + target_position) over edges into
    memory_write, using the model's normalized position feature."""
    out = []
    for g in test_graphs:
        nodes = g.get('nodes', [])
        n = len(nodes)
        id2i = {nd['id']: i for i, nd in enumerate(nodes)}
        best = -1.0
        for e in g.get('edges', []):
            if e.get('type') not in ('data_flow', 'writes_to', 'reads_from'):
                continue
            t = nodes[id2i[e['target']]] if e['target'] in id2i else None
            s = nodes[id2i[e['source']]] if e['source'] in id2i else None
            if t is None or s is None or t['type'] != 'memory_write':
                continue
            sp = id2i[e['source']] / max(n, 1)
            tp = id2i[e['target']] / max(n, 1)
            best = max(best, sp + tp)
        out.append(best)
    return out
