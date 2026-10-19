#!/usr/bin/env python3
"""Sliding-window GNN for harm subgraph detection.
Node features: 12-dim security capability encoding + position = 13 dim.
Edge: 7-dim onehot edge type.
Model: GINEConv → 2× TransformerConv (multi-head) → attention pool → MLP.
"""
from __future__ import annotations

import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TransformerConv, global_add_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ===== Config =====
DATA_DIR = Path(__file__).resolve().parent.parent / "ref" / "dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
HIDDEN_DIM = 256
DROPOUT = 0.4
HEADS = 4
SEED = 42
WINDOW_SIZE = 7
TRAIN_STRIDE = 3
PATIENCE = 25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===== 12-dim Security Capability Encoding =====
from permission_spec import PERMISSION  # noqa: E402  (R7 single source)

TOOL_TO_PERM = {
    "shell_exec":       [1,0,0,0,0,0,0,0,0,0,0,0],
    "code_exec":        [1,0,0,0,0,0,0,0,0,0,0,0],
    "tool_execute":     [1,0,0,0,0,0,0,0,0,0,0,0],
    "file_read":        [0,1,0,0,0,0,0,0,0,0,0,0],
    "database_query":   [0,1,0,0,0,0,0,0,0,0,0,0],
    "information_search": [0,1,0,0,0,0,0,0,0,0,0,0],
    "data_aggregate":   [0,1,0,0,0,0,0,0,0,0,0,0],
    "file_write":       [0,0,1,0,0,0,0,0,0,0,0,0],
    "file_delete":      [0,0,0,1,0,0,0,0,0,0,0,0],
    "web_request":      [0,0,0,0,1,0,0,0,0,0,0,0],
    "web_search":       [0,0,0,0,1,0,0,0,0,0,0,0],
    "account_access":   [0,0,0,0,1,0,0,0,0,0,0,0],
    "web_download":     [0,0,0,0,0,1,0,0,0,0,0,0],
    "memory_write":     [0,0,0,0,0,0,1,0,0,0,0,0],
    "version_control":  [0,0,0,0,0,0,1,0,0,0,0,0],
    "memory_read":      [0,0,0,0,0,0,0,1,0,0,0,0],
    "memory_reflect":   [0,0,0,0,0,0,0,1,0,0,0,0],
    "agent_spawn":      [0,0,0,0,0,0,0,0,1,0,0,0],
    "agent_message":    [0,0,0,0,0,0,0,0,0,1,0,0],
    "email_send":       [0,0,0,0,0,0,0,0,0,0,1,0],
    "message_send":     [0,0,0,0,0,0,0,0,0,0,1,0],
    "social_media":     [0,0,0,0,0,0,0,0,0,0,1,0],
    "document_create":  [0,0,0,0,0,0,0,0,0,0,1,0],
    "prompt_inject":    [0,0,0,0,0,0,0,0,0,0,1,0],
    "system_config":    [0,0,0,0,0,0,0,0,0,0,0,1],
}
DEFAULT_PERM = [0] * len(PERMISSION)

EDGE_TYPES = [
    "data_flow", "causal_derivation", "reads_from", "writes_to",
    "cross_turn", "spawns", "triggers",
]
EDGE2IDX = {t: i for i, t in enumerate(EDGE_TYPES)}


# ===== Content keyword features (6 dims, regex-free, fast) =====
CONTENT_KEYWORDS = [
    # [0] phishing/deception
    ["phish", "email", "invoice", "fake", "template", "impersonat", "pretend", "disguis", "forg", "counterfeit", "spoof"],
    # [1] financial
    ["payment", "$", "bank", "transfer", "money", "credit", "debit", "financial", "salary", "deposit"],
    # [2] malicious intent  
    ["hack", "exploit", "backdoor", "malware", "steal", "trojan", "ransom", "virus", "worm", "payload"],
    # [3] dangerous shell commands
    ["curl ", "wget ", "rm -rf", "| bash", "| sh", "eval(", "exec(", "os.system", "subprocess", "chmod 777"],
    # [4] data exfiltration
    ["upload", "send to", "download", "external", "remote", "post.*data", "exfiltrat", "leak"],
    # [5] credential/access
    ["account", "login", "username", "passwd", "secret", "credential", "token", "auth", "sudo", "root", "admin"],
    # [6] social engineering urgency
    ["update", "verify", "confirm", "alert", "urgent", "immediat", "asap", "deadline", "action require", "important notice", "secur update"],
    # [7] attachment/link
    ["attachment", "click here", "open this", "download", ".pdf", ".doc", "file attach", "link in"],
    # [8] impersonation / legitimate context
    ["IT department", "HR ", "administrator", "support team", "service desk", "security team", "official", "company polic"],
]

def content_features(subtask: str) -> list:
    t = subtask.lower()
    return [1 if any(k in t for k in kw_list) else 0 for kw_list in CONTENT_KEYWORDS]

# ===== Graph conversion =====
def _graph_to_full_data(g: dict) -> Data:
    n_nodes = len(g["nodes"])
    # 12 perm + 1 position + 6 content = 19 dim
    x = torch.zeros(n_nodes, len(PERMISSION) + 1 + len(CONTENT_KEYWORDS))
    y_node = torch.zeros(n_nodes, dtype=torch.float32)
    id_to_idx = {}
    for i, n in enumerate(g["nodes"]):
        perm = TOOL_TO_PERM.get(n["type"], DEFAULT_PERM)
        x[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
        x[i, len(PERMISSION)] = n.get("turn", i) / max(n_nodes, 1)
        subtask = n.get("features", {}).get("subtask", "")
        cf = content_features(subtask)
        x[i, len(PERMISSION)+1:] = torch.tensor(cf, dtype=torch.float32)
        y_node[i] = float(n.get("features", {}).get("node_label", 0))
        id_to_idx[n["id"]] = i

    edge_index, edge_type = [], []
    for e in g.get("edges", []):
        src = id_to_idx.get(e["source"]); tgt = id_to_idx.get(e["target"])
        if src is not None and tgt is not None:
            edge_index.append([src, tgt])
            edge_type.append(EDGE2IDX.get(e["type"], 0))

    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(len(edge_type), len(EDGE_TYPES)) if edge_type else torch.zeros(0, len(EDGE_TYPES))
    for i, eidx in enumerate(edge_type):
        ea[i, eidx] = 1.0
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y_node)


# ===== Window extraction =====
def extract_window(data: Data, start: int, k: int) -> Data | None:
    n = data.x.size(0); end = min(start + k, n)
    if end - start < 2:
        return None
    node_idx = list(range(start, end))
    old2new = {old: new for new, old in enumerate(node_idx)}
    sub_x = data.x[node_idx]

    sub_edges, sub_ea = [], []
    for j in range(data.edge_index.size(1)):
        s, t = data.edge_index[0, j].item(), data.edge_index[1, j].item()
        if s in old2new and t in old2new:
            sub_edges.append([old2new[s], old2new[t]])
            if data.edge_attr is not None:
                sub_ea.append(data.edge_attr[j])

    sub_ei = torch.tensor(sub_edges, dtype=torch.long).t().contiguous() if sub_edges else torch.zeros(2, 0, dtype=torch.long)
    sub_ea = torch.stack(sub_ea) if sub_ea else torch.zeros(sub_ei.size(1), len(EDGE_TYPES))

    sub_y_nodes = data.y[node_idx] if hasattr(data, 'y') and data.y is not None else torch.zeros(end - start)
    win_label = 0.95 if (sub_y_nodes > 0).sum() >= (end - start) * 0.3 else 0.05
    return Data(x=sub_x, edge_index=sub_ei, edge_attr=sub_ea,
                y=torch.tensor([win_label], dtype=torch.float32))


# ===== Window Dataset =====
class WindowDataset(Dataset):
    def __init__(self, jsonl_path: str, k: int = WINDOW_SIZE,
                 train: bool = True, stride: int = 1):
        super().__init__()
        self.k = k; self.train = train
        self.stride = stride if train else 1

        self.graphs = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.graphs.append(json.loads(line))
        self.full_data = [_graph_to_full_data(g) for g in self.graphs]

        self._graph_nw = [max(0, g["num_nodes"] - 1) for g in self.graphs]
        self._cum_nw = [0]
        for nw in self._graph_nw:
            self._cum_nw.append(self._cum_nw[-1] + nw)

        self.window_index = []
        if train:
            for gi, g in enumerate(self.graphs):
                fd = self.full_data[gi]
                n = fd.x.size(0)
                for start in range(0, n - 1, stride):
                    end = min(start + k, n)
                    if end - start >= 2:
                        win_harm = (fd.y[start:end] > 0).sum().item()
                        win_label = 0.95 if win_harm >= (end - start) * 0.3 else 0.05
                        self.window_index.append((gi, start, win_label))

            pos = [(g, s, l) for g, s, l in self.window_index if l > 0.5]
            neg = [(g, s, l) for g, s, l in self.window_index if l <= 0.5]
            self.window_index = pos + neg
            random.shuffle(self.window_index)

    def len(self) -> int:
        return len(self.window_index) if self.train else self._cum_nw[-1]

    def get(self, idx: int) -> Data:
        if self.train:
            gi, start, _ = self.window_index[idx]
        else:
            gi = 0
            for lo, hi in zip(self._cum_nw[:-1], self._cum_nw[1:]):
                if lo <= idx < hi:
                    start = idx - lo; break
                gi += 1
            else:
                gi, start = 0, 0
        return extract_window(self.full_data[gi], start, self.k)


# ===== Model =====
class WindowGINE(torch.nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int,
                 dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, hidden_dim)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim)), edge_dim=edge_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.tf1 = TransformerConv(hidden_dim, hidden_dim // heads,
                                   heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.tf2 = TransformerConv(hidden_dim, hidden_dim // heads,
                                   heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.attn = torch.nn.Linear(hidden_dim, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_proj(x)
        # GINEConv for edge-type-aware structural encoding
        residual = x
        x = self.gine(x, edge_index, edge_attr)
        x = self.bn1(x); x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        # TransformerConv for multi-head graph attention
        residual = x
        x = self.tf1(x, edge_index, edge_attr)
        x = self.bn2(x); x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        residual = x
        x = self.tf2(x, edge_index, edge_attr)
        x = self.bn3(x); x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        # Attention pooling
        a = self.attn(x).sigmoid()
        wx = global_add_pool(x * a, batch)
        wa = global_add_pool(a, batch).clamp(min=1e-8)
        x = wx / wa
        return self.classifier(x).squeeze(-1)


# ===== Training =====
POS_WEIGHT = 1.0

def compute_pos_weight(ds: WindowDataset) -> float:
    n_pos = sum(1 for _, _, lb in ds.window_index if lb > 0.5)
    n_neg = sum(1 for _, _, lb in ds.window_index if lb <= 0.5)
    pw = n_neg / max(n_pos, 1)
    print(f"  pos_weight = neg/pos = {n_neg}/{n_pos} = {pw:.3f}", flush=True)
    return pw


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, n_batches = 0, 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        w = torch.tensor(POS_WEIGHT, device=DEVICE)
        loss = F.binary_cross_entropy_with_logits(logits, data.y, pos_weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_window(model, loader):
    model.eval()
    all_pred_bin, all_label_bin = [], []
    total_bce, n = 0, 0
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_bce += F.binary_cross_entropy_with_logits(logits, data.y, reduction='sum').item()
        probs = torch.sigmoid(logits)
        n += data.num_graphs
        all_pred_bin.extend((probs > 0.5).long().cpu().tolist())
        all_label_bin.extend((data.y > 0.5).long().cpu().tolist())
    return {
        "bce": total_bce / max(n, 1),
        "accuracy": accuracy_score(all_label_bin, all_pred_bin),
        "precision": precision_score(all_label_bin, all_pred_bin, zero_division=0),
        "recall": recall_score(all_label_bin, all_pred_bin, zero_division=0),
        "f1": f1_score(all_label_bin, all_pred_bin, zero_division=0),
    }


@torch.no_grad()
def evaluate_graph_level(model, ds: WindowDataset) -> dict:
    model.eval()
    loader = DataLoader(ds, batch_size=64)
    all_probs = []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())

    all_pred, all_label, all_scores = [], [], []
    for gi, g in enumerate(ds.graphs):
        s, e = ds._cum_nw[gi], ds._cum_nw[gi + 1]
        scores = all_probs[s:e]
        gs = max(scores) if scores else 0.0
        all_pred.append(1 if gs > 0.5 else 0)
        all_label.append(g["label"])
        all_scores.append(gs)

    return {
        "accuracy": accuracy_score(all_label, all_pred),
        "precision": precision_score(all_label, all_pred, zero_division=0),
        "recall": recall_score(all_label, all_pred, zero_division=0),
        "f1": f1_score(all_label, all_pred, zero_division=0),
        "auroc": roc_auc_score(all_label, all_scores) if len(set(all_label)) > 1 else 0.5,
    }


# ===== Main =====
def main():
    global POS_WEIGHT
    print("=== START TRAINING ===", flush=True)
    print(f"Device: {DEVICE}  |  Window: {WINDOW_SIZE}  |  Hidden: {HIDDEN_DIM}  |  Perm: {len(PERMISSION)}d", flush=True)

    train_path = str(DATA_DIR / "unified_train.jsonl")
    val_path = str(DATA_DIR / "unified_val.jsonl")
    test_path = str(DATA_DIR / "unified_test.jsonl")

    t0 = time.time()
    print("Loading training windows...", end=" ", flush=True)
    train_ds = WindowDataset(train_path, k=WINDOW_SIZE, train=True, stride=TRAIN_STRIDE)
    print(f"{len(train_ds)} windows "
          f"({sum(1 for _,_,lb in train_ds.window_index if lb>0.5)}+/"
          f"{sum(1 for _,_,lb in train_ds.window_index if lb<=0.5)}-) "
          f"in {time.time()-t0:.1f}s", flush=True)
    POS_WEIGHT = compute_pos_weight(train_ds)

    t0 = time.time()
    print("Loading val+test windows...", end=" ", flush=True)
    val_ds = WindowDataset(val_path, k=WINDOW_SIZE, train=False)
    test_ds = WindowDataset(test_path, k=WINDOW_SIZE, train=False)
    print(f"val={val_ds.len()} test={test_ds.len()} in {time.time()-t0:.1f}s", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    in_dim = len(PERMISSION) + 1 + len(CONTENT_KEYWORDS)  # dynamic (was hardcoded 19)
    model = WindowGINE(in_dim=in_dim, edge_dim=len(EDGE_TYPES), hidden_dim=HIDDEN_DIM,
                       dropout=DROPOUT, heads=HEADS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    best_val_bce = float("inf"); best_state = None; best_ep = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_win = evaluate_window(model, val_loader)
        scheduler.step(val_win["bce"])
        history.append({"epoch": epoch, "train_loss": train_loss, **val_win})

        if val_win["bce"] < best_val_bce:
            best_val_bce = val_win["bce"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = epoch

        print(f"E {epoch:3d} | loss={train_loss:.4f}  v_bce={val_win['bce']:.4f}  "
              f"v_f1={val_win['f1']:.3f}", flush=True)

        if epoch - best_ep >= PATIENCE:
            print(f"Early stop at {epoch} (best={best_ep} bce={best_val_bce:.4f})", flush=True)
            break

    model.load_state_dict(best_state)
    test_win = evaluate_window(model, test_loader)
    test_graph = evaluate_graph_level(model, test_ds)

    print(f"\nWINDOW-LEVEL TEST (best bce={best_val_bce:.4f})", flush=True)
    for k, v in test_win.items():
        print(f"  {k}: {v:.4f}", flush=True)
    print(f"GRAPH-LEVEL TEST (slide window)", flush=True)
    for k, v in test_graph.items():
        print(f"  {k}: {v:.4f}", flush=True)

    out_dir = Path(__file__).resolve().parent.parent / "code" / "models"
    out_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state, "history": history,
        "test_window": test_win, "test_graph": test_graph,
        "config": {"in_dim": in_dim, "edge_dim": len(EDGE_TYPES),
                   "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
                   "window_size": WINDOW_SIZE, "perm_dims": len(PERMISSION)},
    }, out_dir / "gnn_window_classifier.pt")

    results = Path(__file__).resolve().parent.parent / "LOGS" / "gnn_window_results.json"
    with open(results, "w") as f:
        json.dump({"test_window": test_win, "test_graph": test_graph, "history": history}, f, indent=2)
    print(f"\nSaved: gnn_window_classifier.pt + {results}", flush=True)


if __name__ == "__main__":
    main()
