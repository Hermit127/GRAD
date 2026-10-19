"""GNN Monitor: sliding window inference on live graph."""

import torch
from pathlib import Path
import sys

# Import from parent code
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from train_gnn import (
    WindowGINE, extract_window, WINDOW_SIZE, DEVICE,
    PERMISSION, EDGE_TYPES, TOOL_TO_PERM, DEFAULT_PERM, EDGE2IDX,
)
from torch_geometric.data import Data


class GNNMonitor:
    """Pre-loaded GNN for real-time graph scoring."""

    def __init__(self, model_path: str):
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["config"]
        self.model = WindowGINE(
            in_dim=cfg["in_dim"], edge_dim=cfg["edge_dim"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"], heads=4,
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self._warm()

    def _warm(self):
        """Warm-up inference."""
        x = torch.randn(3, len(PERMISSION) + 1)
        ei = torch.tensor([[0, 1], [1, 2]])
        ea = torch.randn(2, len(EDGE_TYPES))
        batch = torch.zeros(3, dtype=torch.long)
        with torch.no_grad():
            self.model(x, ei, ea, batch)

    def score(self, graph_dict: dict, window_size: int = WINDOW_SIZE) -> float:
        """Run sliding window GNN. Returns graph-level risk score [0, 1]."""
        # Build Data from dict
        nodes = graph_dict["nodes"]
        n_nodes = len(nodes)
        if n_nodes < 2:
            return 0.0

        x = torch.zeros(n_nodes, len(PERMISSION) + 1)
        id_to_idx = {}
        for i, n in enumerate(nodes):
            perm = TOOL_TO_PERM.get(n["type"], DEFAULT_PERM)
            x[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            x[i, len(PERMISSION)] = n.get("turn", i) / max(n_nodes, 1)
            id_to_idx[n["id"]] = i

        ei_list, ea_list = [], []
        for e in graph_dict.get("edges", []):
            s = id_to_idx.get(e["source"])
            t = id_to_idx.get(e["target"])
            if s is not None and t is not None:
                ei_list.append([s, t])
                ea_list.append(EDGE2IDX.get(e["type"], 0))

        ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous() if ei_list else torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(len(ea_list), len(EDGE_TYPES))
        for i, eidx in enumerate(ea_list):
            ea[i, eidx] = 1.0

        fd = Data(x=x, edge_index=ei, edge_attr=ea)

        probs = []
        for start in range(max(0, n_nodes - 1)):
            win = extract_window(fd, start, window_size)
            if win is None:
                continue
            with torch.no_grad():
                logit = self.model(win.x, win.edge_index, win.edge_attr,
                                   torch.zeros(win.x.size(0), dtype=torch.long))
                probs.append(torch.sigmoid(logit).item())

        return max(probs) if probs else 0.0
