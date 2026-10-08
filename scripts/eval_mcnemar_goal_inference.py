"""
scripts/eval_mcnemar_goal_inference.py

McNemar's paired significance tests: does BDIBottleneck significantly
outperform GoalPredictor and ContextGoalPredictor on top-1 goal accuracy?

For each combination of
  split  ∈ {test, test_new_agent}
  fraction ∈ {0.10, 0.20, ..., 1.00}

we collect per-episode binary top-1 correctness for all three models, build
the 2×2 contingency tables (BDI correct / baseline wrong, etc.) and apply
McNemar's exact binomial test (two-sided) with Holm–Bonferroni correction
across all 40 comparisons.

Note on dependence: episodes are grouped by agent (15 agents × ~32 episodes
each).  McNemar's test treats episodes as exchangeable, which slightly inflates
type-I error; the within-agent clustering is noted in the appendix caveat.

Output
------
  results/goal_inference_significance/mcnemar_results.json
  results/goal_inference_significance/mcnemar_table.tex
  results/goal_inference_significance/mcnemar_table_full.tex

Usage
-----
  .venv/bin/python scripts/eval_mcnemar_goal_inference.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import binomtest  # exact binomial (McNemar's exact form)

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.context_goal_predictor import ContextGoalPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.goal_predictor import GoalPredictor
from model.meta_dataset import _norm
from simulation.episode import EpisodeRecord

# ── Evaluation fractions (10% grid matching the main paper curves) ────
EVAL_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ── Checkpoint paths ──────────────────────────────────────────────────
GOAL_CKPT = "outputs/goal_predictor/best.pt"
CTX_CKPT  = "outputs/context_goal/best.pt"
BNK_CKPT  = "outputs/bdi_bottleneck/best.pt"

# ── Splits to evaluate ────────────────────────────────────────────────
SPLITS = ["test", "test_new_agent"]

# ── Context sizes (must match training) ───────────────────────────────
K        = 10
K_RECENT = 5
BATCH    = 64


# ── Helpers ───────────────────────────────────────────────────────────

def _load_records(path: Path) -> list[EpisodeRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


def _build_agent_index(records: list[EpisodeRecord]) -> dict[str, list[int]]:
    idx_map: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        idx_map[r.agent_id].append(i)
    return idx_map


def _build_context_tensor(
    records_ctx: list[EpisodeRecord],
    node_coords: dict,
    K_ctx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [
        torch.tensor(
            [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path],
            dtype=torch.float32,
        )
        for r in records_ctx
    ]
    while len(tensors) < K_ctx:
        tensors.append(tensors[-1].clone() if tensors else torch.zeros(1, 2))
    tensors = tensors[:K_ctx]
    T_max = max(t.size(0) for t in tensors)
    coords = torch.zeros(K_ctx, T_max, 2)
    mask   = torch.zeros(K_ctx, T_max, dtype=torch.long)
    for i, t in enumerate(tensors):
        coords[i, :t.size(0)] = t
        mask[i,   :t.size(0)] = 1
    return coords, mask


def _stack_context(ctx_list):
    K_  = ctx_list[0][0].size(0)
    T_  = max(c.size(1) for c, _ in ctx_list)
    B_  = len(ctx_list)
    co  = torch.zeros(B_, K_, T_, 2)
    ma  = torch.zeros(B_, K_, T_, dtype=torch.long)
    for i, (c, m) in enumerate(ctx_list):
        t = c.size(1)
        co[i, :, :t, :] = c
        ma[i, :, :t]    = m
    return co, ma


# ── Per-episode correctness collection ───────────────────────────────

def collect_correctness(
    gp_model,
    ctx_model,
    bnk_model,
    records: list[EpisodeRecord],
    node_coords: dict,
    poi_to_idx: dict[str, int],
    device,
    frac: float,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For one path fraction, run all three models over all episodes and return
    three boolean arrays (gp_correct, ctx_correct, bnk_correct), one entry
    per episode, True = top-1 prediction was correct.
    """
    rng       = np.random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)

    gp_correct  = []
    ctx_correct = []
    bnk_correct = []

    for start in range(0, len(records), BATCH):
        batch_recs = records[start : start + BATCH]
        B = len(batch_recs)

        desire_ctx_list = []
        recent_ctx_list = []
        curr_list       = []
        gp_curr_list    = []  # GoalPredictor needs flat coords
        labels          = []

        for r in batch_recs:
            flat_idx   = records.index(r)
            agent_idxs = agent_idx[r.agent_id]
            pos        = agent_idxs.index(flat_idx)
            others     = [i for i in agent_idxs if i != flat_idx]

            # Desire context
            if len(others) == 0:
                d_src = [flat_idx] * K
            elif len(others) < K:
                d_src = rng.choice(others, size=K, replace=True).tolist()
            else:
                d_src = rng.choice(others, size=K, replace=False).tolist()

            # Recent context
            preceding = agent_idxs[:pos]
            if len(preceding) == 0:
                fallback = agent_idxs[pos + 1:] or [flat_idx]
                r_src = (fallback * (K_RECENT // len(fallback) + 1))[:K_RECENT]
            elif len(preceding) < K_RECENT:
                r_src = (preceding * (K_RECENT // len(preceding) + 1))[:K_RECENT]
            else:
                r_src = preceding[-K_RECENT:]

            desire_ctx_list.append(
                _build_context_tensor([records[i] for i in d_src], node_coords, K)
            )
            recent_ctx_list.append(
                _build_context_tensor([records[i] for i in r_src], node_coords, K_RECENT)
            )

            # Current trajectory prefix
            T = len(r.path)
            k = max(1, round(frac * T))
            pts = [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path[:k]]
            t   = torch.tensor(pts, dtype=torch.float32)
            curr_list.append(t)
            gp_curr_list.append(t)

            labels.append(poi_to_idx[r.goal_poi])

        # Build tensors for BDI / Context models
        dc, dm = _stack_context(desire_ctx_list)
        rc, rm = _stack_context(recent_ctx_list)

        T_c = max(c.size(0) for c in curr_list)
        cc  = torch.zeros(B, T_c, 2)
        cm  = torch.zeros(B, T_c, dtype=torch.long)
        for i, c in enumerate(curr_list):
            cc[i, :c.size(0)] = c
            cm[i, :c.size(0)] = 1

        # GoalPredictor: flat coords only
        T_gp = max(c.size(0) for c in gp_curr_list)
        gp_cc = torch.zeros(B, T_gp, 2)
        gp_cm = torch.zeros(B, T_gp, dtype=torch.long)
        for i, c in enumerate(gp_curr_list):
            gp_cc[i, :c.size(0)] = c
            gp_cm[i, :c.size(0)] = 1

        labels_t = torch.tensor(labels)

        with torch.no_grad():
            gp_logits = gp_model(gp_cc.to(device), gp_cm.to(device))
            # ContextGoalPredictor returns a single logit tensor
            ctx_logits = ctx_model(
                dc.to(device), dm.to(device),
                rc.to(device), rm.to(device),
                cc.to(device), cm.to(device),
            )
            # BDIBottleneck returns (goal_logits, desire_logits, belief_logits)
            bnk_logits, _, _ = bnk_model(
                dc.to(device), dm.to(device),
                rc.to(device), rm.to(device),
                cc.to(device), cm.to(device),
            )

        gp_pred  = gp_logits.argmax(1).cpu()
        ctx_pred = ctx_logits.argmax(1).cpu()
        bnk_pred = bnk_logits.argmax(1).cpu()

        gp_correct.extend((gp_pred  == labels_t).numpy().tolist())
        ctx_correct.extend((ctx_pred == labels_t).numpy().tolist())
        bnk_correct.extend((bnk_pred == labels_t).numpy().tolist())

    return (
        np.array(gp_correct,  dtype=bool),
        np.array(ctx_correct, dtype=bool),
        np.array(bnk_correct, dtype=bool),
    )


# ── McNemar's exact test ──────────────────────────────────────────────

def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """
    McNemar's exact binomial test (two-sided).
    Contingency table:
        n11 = both correct    n10 = A correct, B wrong
        n01 = A wrong, B correct   n00 = both wrong
    The discordant counts are n10 and n01.
    Returns dict with n, n10, n01, stat, p_value, direction.
    """
    n11 = int(( a_correct &  b_correct).sum())
    n10 = int(( a_correct & ~b_correct).sum())
    n01 = int((~a_correct &  b_correct).sum())
    n00 = int((~a_correct & ~b_correct).sum())
    n   = n11 + n10 + n01 + n00

    discordant = n10 + n01
    if discordant == 0:
        p_value = 1.0
    else:
        # Exact: P(X >= max(n10,n01)) under Binomial(discordant, 0.5), two-sided
        p_value = float(binomtest(max(n10, n01), discordant, 0.5).pvalue)

    direction = "BDI_better" if n01 > n10 else ("baseline_better" if n10 > n01 else "tie")
    return {
        "n": n,
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "discordant": discordant,
        "p_value": p_value,
        "direction": direction,
    }


# ── Holm–Bonferroni correction ────────────────────────────────────────

def holm_bonferroni(results: list[dict], p_key: str = "p_value") -> list[dict]:
    """Add 'p_adjusted' and 'significant' (alpha=0.05) to each result dict."""
    n = len(results)
    indexed = sorted(enumerate(results), key=lambda x: x[1][p_key])
    for rank, (orig_idx, r) in enumerate(indexed):
        threshold = 0.05 / (n - rank)
        results[orig_idx]["p_adjusted"] = min(r[p_key] * (n - rank), 1.0)
        results[orig_idx]["holm_threshold"] = threshold
        results[orig_idx]["significant"]  = r[p_key] <= threshold
    # Once any H_0 is not rejected, all remaining are not rejected
    rejected = True
    for _, (orig_idx, r) in enumerate(indexed):
        if not results[orig_idx]["significant"]:
            rejected = False
        if not rejected:
            results[orig_idx]["significant"] = False
    return results


# ── LaTeX table builder ───────────────────────────────────────────────

def _pval_str(p: float, sig: bool) -> str:
    if p < 0.001:
        s = "< 0.001"
    elif p < 0.01:
        s = f"{p:.3f}"
    else:
        s = f"{p:.3f}"
    return f"\\textbf{{{s}}}" if sig else s


def build_latex_table(all_results: dict) -> str:
    """
    Summary table with one row per (split, comparison), 10 fraction columns.
    Printed as two sub-tables (10--50% and 60--100%) stacked vertically so
    the result fits on a NeurIPS text column.
    """
    split_labels = {
        "test":           "\\texttt{test}",
        "test_new_agent": "\\texttt{test\\_new\\_agent}",
    }
    comp_labels = {
        "BDI_vs_GP":  r"BDI vs.\ GP",
        "BDI_vs_CTX": r"BDI vs.\ CTX",
    }
    fracs_lo = [0.1, 0.2, 0.3, 0.4, 0.5]
    fracs_hi = [0.6, 0.7, 0.8, 0.9, 1.0]

    def _block(fracs):
        hdr = " & ".join(f"{int(f*100)}\\%" for f in fracs)
        col_spec = "ll" + "c" * len(fracs)
        rows = []
        rows.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        rows.append(r"    \toprule")
        rows.append(f"    Split & Comparison & {hdr} \\\\")
        rows.append(r"    \midrule")
        for split in SPLITS:
            sl = split_labels.get(split, split)
            first = True
            for comp in ["BDI_vs_GP", "BDI_vs_CTX"]:
                cells = []
                for frac in fracs:
                    r = all_results[split][frac][comp]
                    cells.append(_pval_str(r["p_adjusted"], r["significant"]))
                sc = f"\\multirow{{2}}{{*}}{{{sl}}}" if first else ""
                rows.append(f"    {sc} & {comp_labels[comp]} & " + " & ".join(cells) + r" \\")
                first = False
            rows.append(r"    \midrule")
        rows[-1] = r"    \bottomrule"
        rows.append(r"  \end{tabular}")
        return rows

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(
        r"  \caption{McNemar's exact test (BDIBottleneck vs.\ baselines) at 10\% "
        r"path-fraction increments.  Holm--Bonferroni adjusted $p$-values across "
        r"40 tests.  \textbf{Bold} = significant at $\alpha=0.05$.  "
        r"Top panel: 10--50\%; bottom panel: 60--100\%.}"
    )
    lines.append(r"  \label{tab:mcnemar_results}")
    lines.extend(_block(fracs_lo))
    lines.append(r"  \vspace{4pt}")
    lines.extend(_block(fracs_hi))
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_latex_table_full(all_results: dict) -> str:
    """
    Full table: one row per (split, comparison, fraction) with raw p, adjusted
    p, discordant counts n01/n10, and significance marker.
    """
    comp_labels = {
        "BDI_vs_GP":  "BDI vs.\\ GoalPredictor",
        "BDI_vs_CTX": "BDI vs.\\ ContextGoalPredictor",
    }
    split_labels = {
        "test":           "\\texttt{test}",
        "test_new_agent": "\\texttt{test\\_new\\_agent}",
    }
    n_fracs = len(EVAL_FRACTIONS)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(
        r"  \caption{Full McNemar's exact test results at 10\% path-fraction "
        r"increments.  $n_{01}$: BDI correct \& baseline wrong; "
        r"$n_{10}$: baseline correct \& BDI wrong; "
        r"$p$: raw two-sided $p$-value; $p^*$: Holm--Bonferroni adjusted "
        r"(40 tests total).  \textbf{Bold} = significant at $\alpha=0.05$.}"
    )
    lines.append(r"  \label{tab:mcnemar_full}")
    lines.append(r"  \begin{tabular}{llcrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Split & Comparison & Frac & $n$ & $n_{01}$ & $n_{10}$ & $p$ & $p^*$ \\"
    )
    lines.append(r"    \midrule")

    for split in SPLITS:
        split_label = split_labels.get(split, split)
        for comp in ["BDI_vs_GP", "BDI_vs_CTX"]:
            first_row = True
            for frac in EVAL_FRACTIONS:
                r = all_results[split][frac][comp]
                sl = (f"\\multirow{{{n_fracs * 2}}}{{*}}{{{split_label}}}"
                      if (first_row and comp == "BDI_vs_GP") else "")
                cl = (f"\\multirow{{{n_fracs}}}{{*}}{{{comp_labels[comp]}}}"
                      if first_row else "")
                p_raw_s = "< 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
                p_adj_s = "< 0.001" if r["p_adjusted"] < 0.001 else f"{r['p_adjusted']:.3f}"
                if r["significant"]:
                    p_adj_s = f"\\textbf{{{p_adj_s}}}"
                frac_s = f"{int(frac * 100)}\\%"
                lines.append(
                    f"    {sl} & {cl} & {frac_s} & "
                    f"{r['n']} & {r['n01']} & {r['n10']} & {p_raw_s} & {p_adj_s} \\\\"
                )
                first_row = False
            lines.append(r"    \cmidrule(l){2-8}")
        lines.append(r"    \midrule")

    # Replace last \midrule with \bottomrule
    lines[-1] = r"    \bottomrule"
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device ────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── POI config ────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order  = [p["id"] for cat_data in poi_cfg["categories"].values() for p in cat_data["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    print("Loading graph …")
    G           = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    print(f"  {len(node_coords):,} nodes\n")

    # ── Load checkpoints ──────────────────────────────────────────────
    def _load_model(ckpt_path, model_cls, state_key="model_state", **kwargs):
        ckpt  = torch.load(ROOT / ckpt_path, map_location=device)
        state = ckpt[state_key]
        model = model_cls(**kwargs).to(device)
        model.load_state_dict(state)
        model.eval()
        return model

    gp_state  = torch.load(ROOT / GOAL_CKPT, map_location=device)["model_state"]
    num_pois  = gp_state["head.weight"].shape[0]
    d_gp      = gp_state["coord_proj.weight"].shape[0]
    gp_model  = GoalPredictor(num_pois=num_pois, d_model=d_gp).to(device)
    gp_model.load_state_dict(gp_state)
    gp_model.eval()
    print(f"GoalPredictor loaded  ({sum(p.numel() for p in gp_model.parameters()):,} params)")

    ctx_state = torch.load(ROOT / CTX_CKPT, map_location=device)["model_state"]
    d_ctx     = ctx_state["coord_proj.weight"].shape[0]
    ctx_model = ContextGoalPredictor(num_pois=num_pois, d_model=d_ctx).to(device)
    ctx_model.load_state_dict(ctx_state)
    ctx_model.eval()
    print(f"ContextGoalPredictor loaded ({sum(p.numel() for p in ctx_model.parameters()):,} params)")

    bnk_state = torch.load(ROOT / BNK_CKPT, map_location=device)["model_state"]
    d_bnk     = bnk_state["coord_proj.weight"].shape[0]
    bnk_model = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_bnk).to(device)
    bnk_model.load_state_dict(bnk_state)
    bnk_model.eval()
    print(f"BDIBottleneck loaded  ({sum(p.numel() for p in bnk_model.parameters()):,} params)\n")

    # ── Collect per-episode correctness and run tests ─────────────────
    # Structure: all_results[split][frac][comparison] = mcnemar dict
    all_results: dict = {}
    flat_tests: list[dict] = []  # for Holm–Bonferroni

    for split in SPLITS:
        path = ROOT / "data" / "splits" / f"{split}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found — skipping {split}.")
            continue
        records = _load_records(path)
        print(f"Split: {split}  ({len(records)} episodes)")
        all_results[split] = {}

        for frac in EVAL_FRACTIONS:
            print(f"  Fraction {frac:.2f} …", end=" ", flush=True)
            gp_corr, ctx_corr, bnk_corr = collect_correctness(
                gp_model, ctx_model, bnk_model,
                records, node_coords, poi_to_idx, device, frac,
            )
            acc_gp  = gp_corr.mean()
            acc_ctx = ctx_corr.mean()
            acc_bnk = bnk_corr.mean()
            print(f"GP={acc_gp:.3f}  CTX={acc_ctx:.3f}  BDI={acc_bnk:.3f}")

            bdi_vs_gp  = mcnemar_exact(bnk_corr, gp_corr)
            bdi_vs_ctx = mcnemar_exact(bnk_corr, ctx_corr)

            bdi_vs_gp.update({"split": split, "frac": frac, "comparison": "BDI_vs_GP",
                               "acc_bdi": float(acc_bnk), "acc_baseline": float(acc_gp)})
            bdi_vs_ctx.update({"split": split, "frac": frac, "comparison": "BDI_vs_CTX",
                                "acc_bdi": float(acc_bnk), "acc_baseline": float(acc_ctx)})

            all_results[split][frac] = {
                "BDI_vs_GP":  bdi_vs_gp,
                "BDI_vs_CTX": bdi_vs_ctx,
            }
            flat_tests.extend([bdi_vs_gp, bdi_vs_ctx])
        print()

    # ── Holm–Bonferroni correction ────────────────────────────────────
    holm_bonferroni(flat_tests)  # mutates in-place
    n_sig = sum(t["significant"] for t in flat_tests)
    print(f"Significant after Holm-Bonferroni correction: {n_sig}/{len(flat_tests)}\n")

    # Print summary
    print(f"{'Split':<20} {'Frac':>5}  {'Comparison':<20}  "
          f"{'BDI':>6}  {'Base':>6}  {'n01':>5}  {'n10':>5}  {'p_raw':>8}  {'p_adj':>8}  Sig")
    print("-" * 105)
    for t in flat_tests:
        print(
            f"{t['split']:<20} {t['frac']:>4.0%}  {t['comparison']:<20}  "
            f"{t['acc_bdi']:>5.1%}  {t['acc_baseline']:>5.1%}  "
            f"{t['n01']:>5}  {t['n10']:>5}  "
            f"{t['p_value']:>8.4f}  {t['p_adjusted']:>8.4f}  "
            f"{'*' if t['significant'] else ''}"
        )

    # ── Save results ──────────────────────────────────────────────────
    out_dir = ROOT / "results" / "goal_inference_significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON: convert float keys to strings
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    json_path = out_dir / "mcnemar_results.json"
    with open(json_path, "w") as f:
        json.dump(_json_safe(all_results), f, indent=2)
    print(f"\nResults  → {json_path}")

    # LaTeX tables
    tex_path = out_dir / "mcnemar_table.tex"
    with open(tex_path, "w") as f:
        f.write(build_latex_table(all_results))
    print(f"Table    → {tex_path}")

    tex_full = out_dir / "mcnemar_table_full.tex"
    with open(tex_full, "w") as f:
        f.write(build_latex_table_full(all_results))
    print(f"Full tbl → {tex_full}")


if __name__ == "__main__":
    main()
