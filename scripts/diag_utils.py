"""
Shared utilities for the belief and counterfactual diagnostics.

These helpers are intentionally local to the diagnostic scripts. They do
not change the default training/evaluation paths used by the paper scripts.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORIES = [
    "attend_class",
    "get_food",
    "study",
    "socialize",
    "exercise",
    "rest_home",
]

_BBOX = {
    "lat_min": 32.867,
    "lat_max": 32.895,
    "lon_min": -117.253,
    "lon_max": -117.220,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_force_math_sdp() -> bool:
    flag = os.environ.get("BDI_FORCE_MATH_SDP", "").lower()
    if flag not in {"1", "true", "yes", "on"}:
        return False
    if not torch.cuda.is_available():
        return False

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return False

    for name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),
        ("enable_math_sdp", True),
    ):
        setter = getattr(cuda_backend, name, None)
        if setter is not None:
            setter(enabled)
    return True


def adamw_kwargs() -> dict[str, bool]:
    flag = os.environ.get("BDI_DISABLE_FOREACH_OPT", "").lower()
    if flag in {"1", "true", "yes", "on"}:
        return {"foreach": False}
    return {}


def get_device(no_cuda: bool = False) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_json(path: str | Path) -> Any:
    with open(resolve(path)) as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_text(path: str | Path, text: str) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with open(resolve(path), newline="") as f:
        return list(csv.DictReader(f))


def load_records(path: str | Path):
    from simulation.episode import EpisodeRecord

    records = []
    with open(resolve(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


def load_split_records(splits_dir: str | Path, splits: list[str], smoke: bool = False, smoke_n: int = 96):
    splits_dir = resolve(splits_dir)
    out = {}
    for split in splits:
        path = splits_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        records = load_records(path)
        out[split] = records[:smoke_n] if smoke else records
    return out


def poi_metadata(poi_cfg_path: str | Path = "config/poi_categories.json") -> dict[str, Any]:
    poi_cfg = load_json(poi_cfg_path)
    poi_order = [
        poi["id"]
        for cat_data in poi_cfg["categories"].values()
        for poi in cat_data["pois"]
    ]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}
    idx_to_poi = {i: pid for pid, i in poi_to_idx.items()}
    poi_to_category = {
        poi["id"]: cat
        for cat, cat_data in poi_cfg["categories"].items()
        for poi in cat_data["pois"]
    }
    idx_to_category = {poi_to_idx[pid]: cat for pid, cat in poi_to_category.items()}
    category_to_indices = {
        cat: [poi_to_idx[p["id"]] for p in cat_data["pois"]]
        for cat, cat_data in poi_cfg["categories"].items()
    }
    poi_to_node = {
        poi["id"]: int(poi["node_id"])
        for cat_data in poi_cfg["categories"].values()
        for poi in cat_data["pois"]
    }
    return {
        "poi_cfg": poi_cfg,
        "poi_order": poi_order,
        "poi_to_idx": poi_to_idx,
        "idx_to_poi": idx_to_poi,
        "poi_to_category": poi_to_category,
        "idx_to_category": idx_to_category,
        "category_to_indices": category_to_indices,
        "poi_to_node": poi_to_node,
        "num_pois": len(poi_order),
    }


def node_coords_from_graph(graph_path: str | Path = "data/ucsd_walk.graphml") -> dict[int, tuple[float, float]]:
    from model.dataset import node_coords_from_graph as _node_coords_from_graph

    try:
        import osmnx as ox

        G = ox.load_graphml(resolve(graph_path))
    except Exception:
        import networkx as nx

        G = nx.read_graphml(resolve(graph_path))
    return _node_coords_from_graph(G)


def load_bdi_bottleneck(ckpt_path: str | Path, device: torch.device):
    from model.bdi_bottleneck import BDIBottleneckPredictor

    ckpt = torch.load(resolve(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state"]
    mkw = ckpt.get("model_kwargs")
    if mkw is None:
        d_model = int(state["coord_proj.weight"].shape[0])
        num_pois = int(state["desire_head.weight"].shape[0])
        first = state["goal_head.0.weight"]
        dim_ff = int(first.shape[0])
        model = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_model, dim_ff=dim_ff)
    else:
        model = BDIBottleneckPredictor(**mkw)
    model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def load_aux_context(ckpt_path: str | Path, device: torch.device):
    from model.context_goal_predictor import AuxSupervisedContextGoalPredictor

    ckpt = torch.load(resolve(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state"]
    mkw = ckpt.get("model_kwargs")
    if mkw is None:
        d_model = int(state["coord_proj.weight"].shape[0])
        num_pois = int(state["desire_head.weight"].shape[0])
        dim_ff = int(state["goal_head.0.weight"].shape[0])
        model = AuxSupervisedContextGoalPredictor(
            num_pois=num_pois,
            d_model=d_model,
            dim_ff=dim_ff,
        )
    else:
        model = AuxSupervisedContextGoalPredictor(**mkw)
    model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _record_to_coords(record, node_coords: dict[int, tuple[float, float]]) -> torch.Tensor:
    def norm(lat: float, lon: float) -> tuple[float, float]:
        lat_n = (lat - _BBOX["lat_min"]) / (_BBOX["lat_max"] - _BBOX["lat_min"])
        lon_n = (lon - _BBOX["lon_min"]) / (_BBOX["lon_max"] - _BBOX["lon_min"])
        return lat_n, lon_n

    pts = [norm(*node_coords.get(node_id, (0.0, 0.0))) for node_id in record.path]
    return torch.tensor(pts, dtype=torch.float32)


def _build_context_tensor(records_ctx: list, node_coords: dict[int, tuple[float, float]], k: int):
    tensors = [_record_to_coords(r, node_coords) for r in records_ctx]
    while len(tensors) < k:
        tensors.append(tensors[-1].clone() if tensors else torch.zeros(1, 2))
    tensors = tensors[:k]
    t_max = max(t.size(0) for t in tensors)
    coords = torch.zeros(k, t_max, 2)
    mask = torch.zeros(k, t_max, dtype=torch.long)
    for i, tensor in enumerate(tensors):
        t_len = tensor.size(0)
        coords[i, :t_len] = tensor
        mask[i, :t_len] = 1
    return coords, mask


def _build_agent_index(records: list) -> dict[str, list[int]]:
    agent_idx: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        agent_idx[record.agent_id].append(i)
    return agent_idx


def _stack_context(ctx_list: list[tuple[torch.Tensor, torch.Tensor]]):
    k = ctx_list[0][0].size(0)
    t_max = max(coords.size(1) for coords, _ in ctx_list)
    bsz = len(ctx_list)
    coords_out = torch.zeros(bsz, k, t_max, 2)
    mask_out = torch.zeros(bsz, k, t_max, dtype=torch.long)
    for i, (coords, mask) in enumerate(ctx_list):
        t_len = coords.size(1)
        coords_out[i, :, :t_len, :] = coords
        mask_out[i, :, :t_len] = mask
    return coords_out, mask_out


def _stack_current(records: list, node_coords: dict[int, tuple[float, float]], fraction: float):
    curr = []
    for record in records:
        t_len = len(record.path)
        k = max(1, round(float(fraction) * t_len))
        coords = _record_to_coords(record, node_coords)[:k]
        curr.append(coords)
    t_max = max(c.size(0) for c in curr)
    bsz = len(curr)
    coords_out = torch.zeros(bsz, t_max, 2)
    mask_out = torch.zeros(bsz, t_max, dtype=torch.long)
    for i, coords in enumerate(curr):
        t_len = coords.size(0)
        coords_out[i, :t_len] = coords
        mask_out[i, :t_len] = 1
    return coords_out, mask_out


def deterministic_context_indices(records: list, k: int, k_recent: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    agent_idx = _build_agent_index(records)
    flat_pos = {
        agent_id: {flat_idx: pos for pos, flat_idx in enumerate(indices)}
        for agent_id, indices in agent_idx.items()
    }
    contexts: list[tuple[list[int], list[int]]] = []
    for flat_idx, record in enumerate(records):
        indices = agent_idx[record.agent_id]
        pos = flat_pos[record.agent_id][flat_idx]
        others = [i for i in indices if i != flat_idx]
        if len(others) == 0:
            desire_src = [flat_idx] * k
        elif len(others) < k:
            desire_src = rng.choice(others, size=k, replace=True).tolist()
        else:
            desire_src = rng.choice(others, size=k, replace=False).tolist()

        preceding = indices[:pos]
        if len(preceding) == 0:
            fallback = indices[pos + 1 :] or [flat_idx]
            recent_src = (fallback * ((k_recent // len(fallback)) + 1))[:k_recent]
        elif len(preceding) < k_recent:
            recent_src = (preceding * ((k_recent // len(preceding)) + 1))[:k_recent]
        else:
            recent_src = preceding[-k_recent:]
        contexts.append((desire_src, recent_src))
    return contexts


@torch.no_grad()
def collect_bdi_components(
    model,
    records: list,
    node_coords: dict[int, tuple[float, float]],
    poi_to_idx: dict[str, int],
    device: torch.device,
    fraction: float = 1.0,
    k: int = 10,
    k_recent: int = 5,
    batch_size: int = 64,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Encode desire_probs, belief_probs, z_tau, and normal goal logits."""
    model.eval()
    contexts = deterministic_context_indices(records, k, k_recent, seed)
    out: list[dict[str, Any]] = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_contexts = contexts[start : start + batch_size]
        desire_ctx = [
            _build_context_tensor([records[i] for i in d_src], node_coords, k)
            for d_src, _ in batch_contexts
        ]
        recent_ctx = [
            _build_context_tensor([records[i] for i in r_src], node_coords, k_recent)
            for _, r_src in batch_contexts
        ]
        dc, dm = _stack_context(desire_ctx)
        rc, rm = _stack_context(recent_ctx)
        cc, cm = _stack_current(batch_records, node_coords, fraction)

        desire_probs, belief_probs, z_tau = model.encode_components(
            dc.to(device),
            dm.to(device),
            rc.to(device),
            rm.to(device),
            cc.to(device),
            cm.to(device),
        )
        goal_logits = model.predict_goal(desire_probs, belief_probs, z_tau)

        for local_i, record in enumerate(batch_records):
            idx = start + local_i
            goal_idx = poi_to_idx[record.goal_poi]
            out.append(
                {
                    "record_index": idx,
                    "agent_id": record.agent_id,
                    "character": record.character,
                    "split": record.split,
                    "config": record.belief_config,
                    "desire": record.desire,
                    "goal_poi": record.goal_poi,
                    "goal_idx": goal_idx,
                    "belief_target": torch.tensor(record.belief_vector, dtype=torch.float32),
                    "desire_target": torch.tensor(record.desire_vector, dtype=torch.float32),
                    "desire_probs": desire_probs[local_i].detach().cpu(),
                    "belief_probs": belief_probs[local_i].detach().cpu(),
                    "z_tau": z_tau[local_i].detach().cpu(),
                    "goal_logits": goal_logits[local_i].detach().cpu(),
                    "path_len": len(record.path),
                    "closed_count": int(sum(1 for v in record.belief_vector if v == 0)),
                    "false_beliefs": list(record.false_beliefs),
                }
            )
    return out


def topk_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = F.softmax(logits, dim=-1)
    top5 = probs.topk(min(5, probs.size(1)), dim=1).indices
    top1 = top5[:, 0]
    n = labels.numel()
    true_prob = probs[torch.arange(n), labels]
    return {
        "top1": float((top1 == labels).float().mean().item()),
        "top5": float((top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "true_goal_prob": float(true_prob.mean().item()),
    }


def binary_metrics_for_closed(y_open_true: np.ndarray, p_open_pred: np.ndarray, threshold: float = 0.5):
    y_closed = (y_open_true == 0).astype(np.int32)
    p_closed = 1.0 - p_open_pred.astype(np.float64)
    pred_closed = (p_closed >= threshold).astype(np.int32)
    pred_open = 1 - pred_closed
    y_open = 1 - y_closed

    tp = int(((pred_closed == 1) & (y_closed == 1)).sum())
    fp = int(((pred_closed == 1) & (y_closed == 0)).sum())
    fn = int(((pred_closed == 0) & (y_closed == 1)).sum())
    tn = int(((pred_closed == 0) & (y_closed == 0)).sum())

    closed_precision = tp / (tp + fp) if (tp + fp) else 0.0
    closed_recall = tp / (tp + fn) if (tp + fn) else 0.0
    closed_f1 = (
        2.0 * closed_precision * closed_recall / (closed_precision + closed_recall)
        if (closed_precision + closed_recall)
        else 0.0
    )

    open_tp = int(((pred_open == 1) & (y_open == 1)).sum())
    open_fp = int(((pred_open == 1) & (y_open == 0)).sum())
    open_fn = int(((pred_open == 0) & (y_open == 1)).sum())
    open_precision = open_tp / (open_tp + open_fp) if (open_tp + open_fp) else 0.0
    open_recall = open_tp / (open_tp + open_fn) if (open_tp + open_fn) else 0.0
    open_f1 = (
        2.0 * open_precision * open_recall / (open_precision + open_recall)
        if (open_precision + open_recall)
        else 0.0
    )

    tpr = closed_recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    overall_acc = (tp + tn) / max(1, tp + tn + fp + fn)

    roc_auc = float("nan")
    pr_auc = float("nan")
    if y_closed.min(initial=0) != y_closed.max(initial=0):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            roc_auc = float(roc_auc_score(y_closed, p_closed))
            pr_auc = float(average_precision_score(y_closed, p_closed))
        except Exception:
            roc_auc = float("nan")
            pr_auc = float("nan")

    return {
        "n_entries": int(y_open_true.size),
        "n_open": int((y_open_true == 1).sum()),
        "n_closed": int((y_open_true == 0).sum()),
        "closed_prop": float((y_open_true == 0).mean()) if y_open_true.size else 0.0,
        "overall_accuracy": float(overall_acc),
        "closed_precision": float(closed_precision),
        "closed_recall": float(closed_recall),
        "closed_f1": float(closed_f1),
        "open_precision": float(open_precision),
        "open_recall": float(open_recall),
        "open_f1": float(open_f1),
        "macro_f1": float((closed_f1 + open_f1) / 2.0),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "roc_auc_closed": roc_auc,
        "pr_auc_closed": pr_auc,
        "closed_tp": tp,
        "closed_fp": fp,
        "closed_fn": fn,
        "closed_tn": tn,
    }


def calibration_bins(y_open_true: np.ndarray, p_open_pred: np.ndarray, n_bins: int = 10):
    y_closed = (y_open_true == 0).astype(np.float64)
    p_closed = 1.0 - p_open_pred.astype(np.float64)
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p_closed >= lo) & (p_closed <= hi)
        else:
            mask = (p_closed >= lo) & (p_closed < hi)
        n = int(mask.sum())
        rows.append(
            {
                "bin": i,
                "p_closed_min": float(lo),
                "p_closed_max": float(hi),
                "n_entries": n,
                "mean_pred_closed": float(p_closed[mask].mean()) if n else float("nan"),
                "empirical_closed_rate": float(y_closed[mask].mean()) if n else float("nan"),
            }
        )
    return rows


def preference_rank(desire_vector: list[float] | torch.Tensor | np.ndarray, poi_idx: int) -> int:
    values = np.asarray(desire_vector, dtype=np.float64)
    ranked = np.argsort(values)[::-1]
    return int(np.where(ranked == poi_idx)[0][0]) + 1


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_se(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=0) / math.sqrt(max(1, arr.size)))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def index_result_files(base_dir: str | Path) -> dict[str, list[str]]:
    base = resolve(base_dir)
    index: dict[str, list[str]] = defaultdict(list)
    if not base.exists():
        return {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(base))
            index[path.suffix.lstrip(".") or "no_extension"].append(rel)
    return dict(index)
