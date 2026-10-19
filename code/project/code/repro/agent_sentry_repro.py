#!/usr/bin/env python3
"""Agent-Sentry Layer 1 (XGBoost, 17-dim provenance features) — local faithful
reproduction on behavior-graph data.

Follows arXiv:2603.22868 Table 9 / Appendix A.2 (features f1-f17), θL=0.33,
θH=0.66, XGBoost 100 trees / depth 4 / lr 0.1 / seed 42 / w+ = n_benign/n_attack.

Per-argument provenance metadata does not exist in our graph schema, so it is
approximated from graph structure: an action node's provenance sources are the
retrieval-type nodes connected to it via data_flow / writes_to edges; untrusted
retrievals T_U = {web_search, web_download, information_search, memory_read,
file_read, database_query} (external / persistent content); action nodes T_A =
{tool_execute, shell_exec, code_exec, email_send, message_send, social_media,
document_create, agent_spawn, version_control, system_config, prompt_inject}.

Trace-level decision: attack iff any action node is Block (p >= 0.66).
Graph score for AUC: max over action nodes of p.
"""
import json, re, sys
from collections import Counter

import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

RETRIEVAL = {"web_search", "web_download", "information_search", "file_read",
             "memory_read", "database_query", "data_aggregate", "account_access"}
UNTRUSTED = {"web_search", "web_download", "information_search", "memory_read",
             "file_read", "database_query"}
ACTION = {"tool_execute", "shell_exec", "code_exec", "email_send", "message_send",
          "social_media", "document_create", "agent_spawn", "version_control",
          "system_config", "prompt_inject", "memory_write", "file_write",
          "file_delete", "web_request"}

SENSITIVE_ARG_NAMES = ["recipient", "recipients", "password", "otp", "ssh_key",
                       "collaborator_email", "new_owner_email", "new_owner_username",
                       "amount", "product_id", "product_ids", "discount_code",
                       "url", "body", "content", "file_id", "account", "iban",
                       "user", "user_email", "participants"]
SENSITIVE_RE = re.compile(r"\b(iban|account|password|email|recipient|amount|"
                          r"credential|secret|token|transfer|send|upload)\b", re.I)


def words(text):
    return set(re.findall(r"[a-z0-9@._-]+", str(text).lower()))


def token_overlap(v, doc):
    return bool(words(v) & words(doc)) if len(str(v)) >= 3 else False


def per_action_features(g, idx):
    """17-dim feature vector for the action node at index idx of graph g."""
    nodes = g["nodes"]
    n = nodes[idx]
    text = str(n.get("features", {}).get("subtask", ""))
    nid = n["id"]
    args = [text] if text else []
    m = max(len(args), 1)

    # control flow over action sequence
    n_action_sofar = sum(1 for j in range(idx) if nodes[j]["type"] in ACTION)
    f1 = n_action_sofar  # ordinal position among action tools
    f2 = sum(1 for j in range(idx) if nodes[j]["type"] in RETRIEVAL)  # preceding retrievals
    f3 = n_action_sofar  # preceding actions
    f4 = 0
    for j in range(idx - 1, -1, -1):
        if nodes[j]["type"] in ACTION:
            break
        if nodes[j]["type"] in RETRIEVAL:
            f4 += 1
    chain = [nodes[j]["type"] for j in range(idx)]
    f5 = len(set(chain))

    # data flow: provenance via edges from retrieval nodes to this action
    sources = []
    for e in g.get("edges", []):
        if e["target"] == nid and e["type"] in ("data_flow", "writes_to", "reads_from"):
            for j in range(idx):
                if nodes[j]["id"] == e["source"] and nodes[j]["type"] in RETRIEVAL:
                    sources.append(nodes[j])
    usrc = [s for s in sources if s["type"] in UNTRUSTED]
    f6 = len(usrc) / max(len(sources), 1)
    f7 = 0.0  # no user-prompt provenance in graph data; approx 0
    # max untrusted hop depth: longest consecutive untrusted retrieval chain
    # reachable from this action backwards along edges
    f8 = 0
    visited = set()
    frontier = [nid]
    depth = {nid: 0}
    while frontier:
        cur = frontier.pop()
        d = depth[cur]
        for e in g.get("edges", []):
            if e["target"] == cur and e["type"] in ("data_flow", "writes_to", "reads_from"):
                s = e["source"]
                if s in visited:
                    continue
                visited.add(s)
                for j in range(idx):
                    if nodes[j]["id"] == s and nodes[j]["type"] in UNTRUSTED:
                        depth[s] = d + 1
                        f8 = max(f8, d + 1)
                        frontier.append(s)
    f9 = len({s["type"] for s in sources})
    f10 = len({s["type"] for s in usrc})

    # information flow over sensitive args (approximated from text)
    sens = [a for a in args if any(k in str(a).lower() for k in SENSITIVE_ARG_NAMES)]
    f11 = len(sens)
    f12 = len([a for a in sens if a in [str(s.get("features", {}).get("subtask", "")) for s in usrc]])
    f13 = 0.0  # user prompt unavailable
    f14 = f11
    f15 = len([a for a in args if SENSITIVE_RE.search(str(a))])
    f16 = 0  # derivation type unavailable in graph schema

    # value tracing f17: arg value verbatim in any preceding untrusted retrieval output
    f17 = 0
    for a in args:
        for j in range(idx):
            if nodes[j]["type"] in UNTRUSTED:
                rt = str(nodes[j].get("features", {}).get("subtask", ""))
                if token_overlap(a, rt):
                    f17 += 1
                    break
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17]


def graph_features(g):
    """Per-action features for every ACTION node; graph score = max p."""
    feats = []
    for i, nd in enumerate(g["nodes"]):
        if nd["type"] in ACTION:
            feats.append(per_action_features(g, i))
    return feats


def fit_predict(train_g, test_g):
    X_tr, y_tr = [], []
    for g in train_g:
        for f in graph_features(g):
            X_tr.append(f)
            y_tr.append(g["label"])
    X_te, groups = [], []
    for g in test_g:
        fs = graph_features(g)
        X_te.extend(fs)
        groups.append((g, len(fs)))
    if not X_tr or not X_te:
        return None
    n_benign = sum(1 for y in y_tr if y == 0)
    n_attack = sum(1 for y in y_tr if y == 1)
    w = n_benign / max(n_attack, 1)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              random_state=42, scale_pos_weight=w,
                              eval_metric="logloss")
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = [1 if p >= 0.66 else 0 for p in probs]
    ys, ps, gpred = [], [], []
    k = 0
    for g, nf in groups:
        ys.append(g["label"])
        blk = [preds[k + j] for j in range(nf)]
        gpred.append(1 if any(blk) else 0)
        ps.append(max(probs[k:k + nf]) if nf else 0.0)
        k += nf
    return ys, ps, gpred


def report(name, ys, ps, gpred):
    auc = roc_auc_score(ys, ps)
    f1 = f1_score(ys, gpred)
    return f"{name}: AUC={auc:.4f} F1={f1:.4f} P={precision_score(ys, gpred):.3f} " \
           f"R={recall_score(ys, gpred):.3f}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mg"
    if mode == "mg":
        graphs = [json.loads(l) for l in open(
            "./ref/dataset/memorygraft_augmented.jsonl") if l.strip()]
        from collections import defaultdict
        def seed_id(t):
            return t.split("_v")[0]
        att, ben = defaultdict(list), defaultdict(list)
        for g in graphs:
            (att if g["label"] else ben)[seed_id(g["task_id"])].append(g)
        as_, bs = sorted(att), sorted(ben)
        import random
        random.seed(42)
        random.shuffle(as_)
        random.shuffle(bs)
        train_g, test_g = [], []
        for s in as_[:8]:
            train_g += att[s]
        for s in as_[8:]:
            test_g += att[s]
        for s in bs[:80]:
            train_g += ben[s]
        for s in bs[80:]:
            test_g += ben[s]
        print(f"MG: train {len(train_g)} test {len(test_g)} (unseen {as_[8:]})")
        ys, ps, gpred = fit_predict(train_g, test_g)
        print(report("Agent-Sentry L1 (local repro, MG)", ys, ps, gpred))
    elif mode == "decomp":
        test_g = [json.loads(l) for l in open(
            "./ref/dataset/decomp_graphs_test_mixed.jsonl") if l.strip()]
        train_g = [json.loads(l) for l in open(
            "./ref/dataset/decomp_graphs_train_mixed.jsonl") if l.strip()]
        print(f"Decomp: train {len(train_g)} test {len(test_g)}")
        ys, ps, gpred = fit_predict(train_g, test_g)
        print(report("Agent-Sentry L1 (local repro, Decomp)", ys, ps, gpred))


if __name__ == "__main__":
    main()
