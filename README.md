# Structured Machine Theory of Mind from Agent Trajectories

Official code, data, and trained models for

> **Structured Machine Theory of Mind from Agent Trajectories**
> Nigel F. Doering\*, Rahath Malladi\*, Arshia Sangwan, M Tanjid Hasan Tonmoy, Jacob Doan,
> David Danks, Tauhidur Rahman (\* equal contribution)
> *Transactions on Machine Learning Research* (TMLR), 2026 ·
> [OpenReview](https://openreview.net/forum?id=PfSNLg0zcK)

**SMToM** (Structured Machine Theory of Mind) attributes explicit mental states to an agent
from its observed trajectories: what it **desires**, what it **believes** is available, and
which **goal** it is heading to. The model, **BDIBottleneck**, predicts the goal *only through*
its inferred belief and desire channels, so counterfactual questions ("where would this agent
go if it believed this building were closed?") become a direct edit of the model's input.

Everything is evaluated in a controlled pedestrian-navigation simulator on the UCSD campus
walking graph, where the true beliefs and desires of every simulated agent are known.

---

## Citation

If you use this code, data, or the trained models, please cite:

```bibtex
@article{doering2026structured,
  title   = {Structured Machine Theory of Mind from Agent Trajectories},
  author  = {Doering, Nigel F. and Malladi, Rahath and Sangwan, Arshia and Tonmoy, M Tanjid Hasan
             and Doan, Jacob and Danks, David and Rahman, Tauhidur},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=PfSNLg0zcK}
}
```

GitHub's "Cite this repository" button (from [`CITATION.cff`](CITATION.cff)) gives the same reference.

---

## Contents

1. [Quick start](#1-quick-start)
2. [What can be reproduced, and how exactly](#2-what-can-be-reproduced-and-how-exactly)
3. [Repository layout](#3-repository-layout)
4. [The simulator and dataset](#4-the-simulator-and-dataset)
5. [Models](#5-models)
6. [Training](#6-training)
7. [Evaluation: paper figure and table map](#7-evaluation-paper-figure-and-table-map)
8. [Appendix diagnostics](#8-appendix-diagnostics)
9. [What to expect when you retrain](#9-what-to-expect-when-you-retrain)
10. [Known small discrepancies with the paper](#10-known-small-discrepancies-with-the-paper)

---

## 1. Quick start

Python **3.12**. All commands run from the repository root.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All downloads are attached to the
[v1.0 release](https://github.com/NigelDoering/structured-machine-tom/releases/tag/v1.0), with
SHA-256 checksums in `SHA256SUMS.txt`. From the repository root:

```bash
REL=https://github.com/NigelDoering/structured-machine-tom/releases/download/v1.0
curl -LO $REL/smtom-bdibottleneck.pt            # the trained model (3 MB)
curl -LO $REL/checkpoints-paper.tar.gz          # optional: exact reproduction of the main paper (43 MB)
curl -LO $REL/checkpoints-appendix.tar.gz       # optional: appendix models and dense dataset (191 MB)
curl -LO $REL/diagnostics-intermediate.tar.gz   # optional: per-episode appendix outputs (82 MB)
```

### Use the trained model

Most readers only need **`smtom-bdibottleneck.pt`**, the trained SMToM model (BDIBottleneck,
seed 0, the checkpoint behind all of the paper's counterfactual results). Put it in
`checkpoints/` and run the demo, which attributes desires, beliefs, and a goal for one test
episode and then edits a belief and a desire:

```bash
mkdir -p checkpoints && mv smtom-bdibottleneck.pt checkpoints/
.venv/bin/python scripts/demo_counterfactual.py --split test --episode 0 --fraction 0.5
```

```
Agent studious_stem_0 (studious_stem), true goal: pcynh, observed 20/40 steps
  inferred desires : biomedical_lib 0.27, erc_east 0.18, pcynh 0.12
  believed closed  : 64_degrees, pines, foodworx
  predicted goals  : biomedical_lib 0.32, center_hall 0.27, pcynh 0.26

If the agent believed biomedical_lib were closed:
  P(biomedical_lib): 0.32 -> 0.24
  predicted goals  : pcynh 0.30, center_hall 0.27, biomedical_lib 0.24

If the agent wanted food instead:
  P(any food POI)  : 0.01 -> 0.21
  predicted goals  : sun_god_lawn 0.28, the_pub 0.26, pines 0.13
```

The single checkpoint also works with the evaluation scripts, for example
`scripts/eval_counterfactual_belief.py --bnk-ckpt checkpoints/smtom-bdibottleneck.pt`.

### Reproduce the paper

Exact reproduction needs every model the paper reports (all baselines and ablations, 3 seeds
each). These ship as archives that unpack into `outputs/` at the repository root:

```bash
tar -xzf checkpoints-paper.tar.gz         # 18 main-paper models (6 models x 3 seeds) -> outputs/<model>/
tar -xzf checkpoints-appendix.tar.gz      # 69 appendix models + dense dataset       -> outputs/diagnostics/
tar -xzf diagnostics-intermediate.tar.gz  # large per-episode CSVs                   -> outputs/diagnostics/analyses/
```

Regenerate every main-paper figure and table, then check them against the committed copies:

```bash
reproduce/evaluate_main.sh
.venv/bin/python reproduce/check_outputs.py results/
```

The repository already contains the paper's outputs (`results/`, `outputs/**/accuracy_results.json`,
`outputs/diagnostics/**`), and every reproduction script writes to the same paths.
`reproduce/check_outputs.py` compares the regenerated files with the committed versions:
figures by their pixels and tables/JSON by their numbers. (Use it rather than `git status`:
PNG files can differ byte-for-byte across installs of the image libraries even when every
pixel is identical.)

Training uses CUDA if available, then Apple-silicon MPS, then CPU (`--no-cuda` forces CPU).
Set `PY=/path/to/python` to use an interpreter other than `.venv/bin/python` in the
`reproduce/` scripts.

---

## 2. What can be reproduced, and how exactly

| Level | What you run | What you get | Applies to |
|---|---|---|---|
| **A. Re-evaluate the released checkpoints** | `checkpoints-paper.tar.gz`, then `reproduce/evaluate_main.sh` (minutes) | **Identical** numbers and pixel-identical figures. Evaluation is fully deterministic. | All main-paper results: Figs. 2–4 and 6–19, Tables 1–3, 14–15, 20–22 |
| **B. Regenerate the appendix diagnostics from released checkpoints** | `STAGES="diagnostics aggregate trajectory tables" reproduce/appendix.sh` | Paper tables bit-identical; other diagnostic files match to floating-point precision (≈10⁻⁶) on different hardware | Tables 12, 13, 16–19 and the channel-use numbers in Appendix F |
| **C. Retrain from scratch** | `reproduce/train_main.sh`, then `reproduce/evaluate_main.sh` | **Statistically equivalent, not identical** results (Section 9) | Everything |
| **D. Regenerate the dataset** | `scripts/run_simulation.py` | A dataset drawn from the same generative process, **not** the paper's exact episodes (Section 4.3) | Data |

The committed `data/splits/` **is the paper's dataset**. Use it for any comparison with the paper.

Compute for level C: one 300-epoch run takes roughly 25–30 minutes on an Apple M4 Max (MPS);
the main paper needs 18 runs (6 models × 3 seeds). The appendix model families are 30-epoch
runs of a few minutes each, about 70 runs in total.

---

## 3. Repository layout

```
config/        POI list, agent archetypes, belief configurations (sparse and dense)
simulation/    probabilistic BDI data generator (no learned components)
model/         PyTorch models and dataset/dataloader code
scripts/       entry points: simulate, train_*, eval_*, diagnostics (diag_*, cf_*, robustness_*)
reproduce/     one-command pipelines: evaluate_main.sh, train_main.sh, appendix.sh
checkpoints/   (not committed) the downloaded single SMToM model for scripts/demo_counterfactual.py
data/          campus walking graph, BToM distance cache, dataset splits (the paper's dataset)
outputs/       training artifacts: per-model training histories/curves and, after download, checkpoints
results/       evaluation outputs: the paper's figures, tables (.tex), and JSON results
```

Conventions: training writes to `outputs/`, evaluation writes to `results/`, and checkpoints
(`*.pt`) are never committed; they ship as release assets.

---

## 4. The simulator and dataset

### 4.1 Environment and mental states

The environment is the UCSD campus pedestrian graph from OpenStreetMap
([`data/ucsd_walk.graphml`](data/ucsd_walk.graphml), ~4,200 nodes, cached so no download is
needed). There are **24 points of interest (POIs)** in **6 desire categories** of 4 POIs each:

```
attend_class : wlh, center_hall, pcynh, mandeville
get_food     : 64_degrees, pines, foodworx, oceanview
study        : geisel, biomedical_lib, price_center_study, erc_east
socialize    : price_center_west, sun_god_lawn, the_pub, sixth_village
exercise     : rimac, canyonview, rimac_pool, triton_track
rest_home    : erc_africa_hall, muir_residences, sixth_residences, warren_residences
```

[`config/poi_categories.json`](config/poi_categories.json) fixes this order; it defines the
0–23 index used by every belief, desire, and goal vector. The paper's appendix lists the same
POIs as `POI-01` … `POI-24` in this order.

| Component | Type | Meaning |
|---|---|---|
| Beliefs `b` | {0,1}^24 | which POIs the agent believes are open (0 = falsely believes closed) |
| Desires `π` | distribution over 24 POIs | the agent's stable preferences (category × within-category) |
| Goal `g` | POI index | the destination of the current trajectory |

The physical environment is always fully open; false beliefs exist only in `b`.

### 4.2 Generative process

Agents come from 5 archetypes ([`config/agent_characters.json`](config/agent_characters.json)),
each a set of Dirichlet parameters over categories and within-category POIs. Each episode:

1. sample a desire category from the agent's preferences;
2. mask POIs the agent believes are closed and pick a goal among the believed-open POIs of that category;
3. route a near-shortest path: sample among the *k* shortest simple paths with
   probability ∝ exp(−length / 100 m).

Belief configurations live in [`config/belief_configs.json`](config/belief_configs.json):
`B0`–`B7` for training (B0 = all correct, the others close 3–5 POIs) and `BH0`–`BH1`, held
out for evaluation only. The appendix's dense regime adds
[`config/belief_configs_dense.json`](config/belief_configs_dense.json).

### 4.3 Dataset splits

`data/splits/` holds one JSON object per line per episode (`agent_id`, `character`, `belief_config`,
`belief_vector`, `false_beliefs`, `desire_vector`, `desire`, `goal_poi`, `goal_node`,
`start_node`, `path`, …):

| Split | Agents | Belief configs | Episodes | Purpose |
|---|---|---|---|---|
| `train` | 15 training agents | B0–B7 | 2,040 | training |
| `val` | same | B0–B7 | 480 | model selection |
| `test` | same | B0–B7 | 480 | in-distribution test |
| `test_new_agent` | 15 unseen agents | B0–B7 | 480 | generalization to new agents |
| `test_bh` | training agents | BH0–BH1 | 480 | held-out belief configurations |
| `test_new_agent_bh` | unseen agents | BH0–BH1 | 480 | both shifts |
| `distractor_test*` | subsets | | | trajectories passing near a low-preference POI (`scripts/extract_distractor_episodes.py`) |

**Regenerating the data.** `scripts/run_simulation.py` reproduces the generative process, but
not the paper's exact episodes. When the dataset was generated, the per-agent order of belief
configurations was seeded with Python's built-in `hash()`, which is randomized per process, so
that draw cannot be recreated. This release seeds it with `zlib.crc32`, so regeneration is now
deterministic, but it yields a *different* draw from the same distribution. Write regenerated
data somewhere other than `data/splits` so that the committed dataset is not overwritten:

```bash
.venv/bin/python scripts/run_simulation.py --out data/splits_regenerated
```

---

## 5. Models

| Model | File | Description |
|---|---|---|
| `GoalPredictor` | `model/goal_predictor.py` | current trajectory only (no agent context) |
| `ContextGoalPredictor` | `model/context_goal_predictor.py` | adds same-agent context; goal head reads raw `[z_d‖z_r‖z_τ]`, no mental-state supervision |
| `AuxSupervisedContextGoalPredictor` ("AuxContext") | `model/context_goal_predictor.py` | same auxiliary desire/belief heads and losses as BDIBottleneck, but the goal head stays unbottlenecked |
| **`BDIBottleneckPredictor`** | `model/bdi_bottleneck.py` | **the paper's model**: goal head reads only `[π̂‖b̂‖z_τ]` |
| `BDIBottleneckNoDesire` / `NoBelief` | `model/bdi_bottleneck_ablations.py` | ablations that drop one channel |

All context models consume three trajectory streams: a random same-agent **desire context**
(K = 10 episodes), the agent's **recent context** (K_recent = 5), and the **current trajectory
prefix** (`model/meta_dataset.py`). The belief head reads `[z_d‖z_r]` and the desire head reads `z_d`.

**Counterfactual API.** `BDIBottleneckPredictor.encode_components()` returns
`(desire_probs, belief_probs, z_τ)`. Edit a channel, for example `belief_probs[:, g] = 0` or a
replacement desire prototype, and call `predict_goal(...)` to re-run only the goal head.
`scripts/demo_counterfactual.py` is a short, self-contained example; the full evaluations are
`scripts/eval_counterfactual_belief.py` and `scripts/eval_counterfactual_desire.py`.

`model/bdi_predictor.py` (`BDIPredictor`) and `scripts/train_bdi_predictor.py` are not models
evaluated in the paper. They hold the shared encoder builder and the training/evaluation
utilities that the other scripts import.

The two non-learned baselines, **VisitFrequency** and **BToM (Informed)** (Bayesian inverse
planning with a Boltzmann step likelihood), are implemented inside `scripts/eval_goal_inference.py`
and are deterministic.

---

## 6. Training

The main-paper models train for **300 epochs** with seeds **0, 5, 10**. Each script defaults to
`--epochs 30`, so pass `--epochs 300` explicitly. All other defaults match the paper's
hyperparameter tables (Appendix B). Retrain everything with:

```bash
reproduce/train_main.sh              # 18 runs, sequential
JOBS=3 reproduce/train_main.sh       # three at a time
SEEDS="0" EPOCHS=300 reproduce/train_main.sh   # one seed only
```

or train a single model:

```bash
.venv/bin/python scripts/train_bdi_bottleneck.py --epochs 300 --seed 0 --out outputs/bdi_bottleneck
```

With `--seed N`, a run writes to `<out>/seed_N/`: `best.pt` (selected by validation goal
top-1), `accuracy_results.json` (per-epoch validation history plus per-fraction accuracy), and
`training_curves.png`. Paper Figure 5 is `outputs/bdi_bottleneck/training_curves.png`.

**Checkpoint layout.** In `checkpoints-paper.tar.gz`, seed 0 sits directly in `outputs/<model>/`
and seeds 5 and 10 sit in `outputs/<model>/seed_5/` and `seed_10/`. The original seed-0 runs
were launched without `--seed`, which is why they have no subfolder. `reproduce/evaluate_main.sh`
accepts either layout and prefers `seed_0/` when it exists. AuxContext checkpoints are in
`outputs/context_goal_aux/seed_{0,5,10}/`.

---

## 7. Evaluation: paper figure and table map

`reproduce/evaluate_main.sh` runs everything in this table, in the right order.

| Paper item | Script | Output |
|---|---|---|
| Fig. 2, Table 1, Table 20 (goal inference, all models) | `eval_goal_inference.py`, then `eval_aux_context_bh.py` and `make_goal_inference_figures.py` (add AuxContext) | `results/goal_inference/` |
| Tables 21–22 (McNemar tests) | `eval_mcnemar_goal_inference.py` | `results/goal_inference_significance/` |
| Fig. 3, Table 2, Figs. 12–19 (desire counterfactuals) | `eval_counterfactual_desire.py --paper` | `results/counterfactual_desire/` |
| Fig. 4, Table 3, Tables 14–15, Fig. 9 (belief counterfactuals) | `eval_counterfactual_belief.py --paper` | `results/counterfactual_belief/results.json` (incl. agent-clustered tests) |
| Fig. 5 (training curves) | `train_bdi_bottleneck.py` | `outputs/bdi_bottleneck/training_curves.png` |
| Fig. 6 (ablations) | `eval_ablations.py` | `results/ablation/` |
| Fig. 7 (distractor robustness) | `eval_distractor.py --paper` | `results/distractor_eval/comparison.png` |
| Fig. 8 (belief-head sensitivity) | `eval_belief_evidence.py` | `results/belief_evidence/evidence_sensitivity_paper.png` |
| Tables 4, 8–11 (splits, configurations, archetypes) | — | `data/splits/`, `config/` |
| Tables 12–13, 16–19 | see [Section 8](#8-appendix-diagnostics) | `outputs/diagnostics/`, `results/trajectory_counterfactuals/` |

Figure 1 (architecture diagram) is not generated by code.

**How AuxContext enters Figure 2 and Tables 1 and 20.** Its test and new-agent curves come
from the per-fraction accuracy that `train_context_goal_predictor_aux.py` records at the end of
training (`outputs/context_goal_aux/seed_*/accuracy_results.json`). Its held-out-belief columns
come from `eval_aux_context_bh.py`, which evaluates the AuxContext checkpoints and writes
`results/goal_inference/aux_context_goal_quarters.csv`. `make_goal_inference_figures.py`
merges both into the figure and tables.

The counterfactual evaluations use the seed-0 BDIBottleneck by default; pass `--bnk-ckpt` to
evaluate another checkpoint.

---

## 8. Appendix diagnostics

The appendix's belief diagnostics, robustness sweeps, and trajectory-level counterfactuals use
**separate 30-epoch model families**, not the 300-epoch main-paper models:

| Family | Location | Used for |
|---|---|---|
| Sparse (default data), seeds 0/5/10 | `outputs/diagnostics/sparse/<model>/seed_*` | Table 12 (sparse rows), Appendix F channel-use numbers, sparse trajectory diagnostics |
| Dense false beliefs, seed 0 | `outputs/diagnostics/dense/{bdi_bottleneck,context_goal_aux_recovered}/seed_0` | Table 12 (dense rows), dense channel use, dense trajectory diagnostics |
| Robustness sweeps | `outputs/diagnostics/robustness/{sparse,dense}/<config>/<model>/seed_*` | Table 13 and the robustness rows of Tables 17–19 |

The dense regime adds medium (6–8 closed POIs) and dense (11–12 closed) belief configurations.
Its dataset (`outputs/diagnostics/dense/splits/`) ships in `checkpoints-appendix.tar.gz`. It was
generated with shortest-path routing (`--router-k 1`) before the seeding fix described in
Section 4.3, so like the main dataset it cannot be regenerated exactly; the archived copy is
the one the paper uses.

`reproduce/appendix.sh` runs the full pipeline in stages
(`data`, `train`, `diagnostics`, `robustness`, `aggregate`, `trajectory`, `tables`);
`robustness` retrains the sweeps, while `aggregate` only evaluates their checkpoints:

```bash
reproduce/appendix.sh                                            # everything (retrains)
STAGES="diagnostics aggregate trajectory tables" reproduce/appendix.sh  # from the released appendix checkpoints
```

| Paper item | Scripts | Reference output |
|---|---|---|
| Table 12 (sparse vs. dense closed-belief F1 / PR-AUC) | `diag_belief_attribution.py` | `outputs/diagnostics/analyses/{current,dense_current}_belief_attribution/` |
| Appendix F.3 channel use (KL under all-open/shuffled beliefs, goal-head norms, dose response) | `diag_belief_channel_usage.py`, `diag_belief_sensitivity.py`, `diag_belief_dose_response.py` | `outputs/diagnostics/analyses/{,dense_}belief_*` |
| Table 13 (same-budget robustness) | `robustness_sweep.py`, `robustness_aggregate.py` | `outputs/diagnostics/robustness/{sparse,dense}/aggregate/` |
| Table 16 (simulator-resampled rollouts) | `cf_simulator_rollouts.py`, `cf_bootstrap_ci.py`, `make_trajectory_cf_tables.py` | `results/trajectory_counterfactuals/simulator_rollout_table.tex` |
| Tables 17–19 (model-planned counterfactual paths) | `cf_model_planned_paths.py`, `robustness_model_planned.py`, `make_trajectory_cf_tables.py` | `results/trajectory_counterfactuals/model_planned_*_table.tex` |

`make_trajectory_cf_tables.py` reads the per-seed `goal_changes.csv` and
`model_vs_simulator.csv` files. Those per-episode files (~240 MB) are in
`diagnostics-intermediate.tar.gz` rather than in git, so download it to rebuild Tables 16–19
without rerunning the trajectory stage.

The appendix models were trained on a CUDA workstation, where some transformer runs needed two
stability guards (math SDPA attention and non-foreach AdamW) and a smaller batch size for a few
unstable seeds. The guards are off by default; enable them with `BDI_FORCE_MATH_SDP=1` and
`BDI_DISABLE_FOREACH_OPT=1` (see `scripts/diag_utils.py`). On other hardware the same commands
run unchanged.

Every evaluation command in `appendix.sh` (the `diagnostics`, `aggregate`, `trajectory`, and
`tables` stages) has been checked: run from a fresh clone on the released checkpoints, it
regenerates the committed outputs. The robustness-sweep training commands are the recorded
ones. The dense-dataset generation and the base-family training commands are reconstructed from
the scripts' defaults and run notes, are marked as such in the script, and have not been rerun
(Section 9 covers what to expect from retraining).

---

## 9. What to expect when you retrain

Retraining reproduces the paper's conclusions, not its exact digits.

**Why the digits differ.**

- *Hardware and kernels.* On a fixed machine, a run with a fixed `--seed` is deterministic
  (two runs produce identical weights). Across hardware (MPS vs. CUDA vs. CPU) and library
  versions, floating-point differences grow over 300 epochs into different checkpoints.
- *The original seed-0 runs were unseeded* (Section 6), so seed 0 cannot be recreated exactly.
- *Model selection.* `best.pt` is the epoch with the best validation top-1, so a small shift
  in the training trajectory can change which epoch is selected.

**How big the differences are.** These ranges come from the three paper seeds, all evaluated
with the same code:

- *Goal inference (Fig. 2).* The seed-to-seed standard deviation of top-1 accuracy ranges
  from **0.2 to 3.1 percentage points** across models, splits, and path fractions (the shaded
  bands in Fig. 2). Differences between learned models smaller than a few points, such as
  BDIBottleneck vs. AuxContext at some fractions, are within seed noise; the paper accordingly
  describes BDIBottleneck as competitive with AuxContext.
- *Desire counterfactuals (Fig. 3, Table 2).* These are very stable. Across seeds 0/5/10 the
  mean off-diagonal Δ is 0.144–0.151 (test) and 0.132–0.134 (new agent) with model-derived
  prototypes, 0.332–0.344 and 0.319–0.333 with handcrafted ones, and 26–30 of the 30 category
  pairs are significant for every checkpoint we evaluated (including a retrained one, below).
- *Belief counterfactuals (Fig. 4, Table 3).* These effects are small, so they vary more.
  Across seeds 0/5/10 the 50% goal drop is −0.009 to −0.015 (test) and −0.022 to −0.028
  (new agent). The new-agent effect is significant for every seed (episode-level
  p ≤ 5×10⁻⁵, agent-clustered p ≤ 0.0011). The held-in agent-clustered p-value ranges from
  0.03 to 0.20, which is why the paper rests the belief evidence on the new-agent split.
- *Appendix diagnostics.* The robustness sweep (Table 13) already reports mean ± SD over 5
  (sparse) or 3 (dense) seeds. Expect retrained values within about one SD.
- *Deterministic everywhere.* VisitFrequency, BToM (Informed), the dataset statistics, and
  every evaluation of a fixed checkpoint.

**Retraining check.** We retrained BDIBottleneck seed 5 from scratch with this repository
(`--epochs 300 --seed 5`, Apple M4 Max, MPS) and compared it with the released seed-5
checkpoint. Both selected epoch 216. Validation top-1 was 58.1% vs. 57.5% (original), and test
top-1 was 37.3% vs. 37.7% at 50% reveal and 58.8% vs. 58.3% at full reveal. The belief
intervention gave a goal drop of −0.0092 vs. −0.0093 (test) and −0.0287 vs. −0.0283 (new
agent); the new-agent effect stayed at episode-level p = 5×10⁻⁵ and agent-clustered p = 10⁻⁴.
Desire Δ moved by less than 0.002, and the significant category pairs were 29/30 vs. 29/30
(test) and 26/30 vs. 27/30 (new agent). The held-in agent-clustered belief p-value, the most
fragile statistic in the paper, moved from 0.027 to 0.063.

---

## 10. Known small discrepancies with the paper

These are rounding or reporting differences. None affects a conclusion.

- **Table 2.** `eval_counterfactual_desire.py` prints means and CIs to 4 decimals. The paper
  rounds them to 3 from the 4-decimal values, so the new-agent model-derived mean appears as
  0.135 in the paper and 0.1345 here; the unrounded value is 0.13447.
- **Table 14 and Section 6.3.** The agent-clustered goal-drop permutation p-value on the
  new-agent split is 0.00055. The paper reports it as 0.0006; it rounds to 0.0005.
- **Ablation evaluation (Fig. 6) vs. goal-inference evaluation (Fig. 2).** The two scripts
  sample evaluation contexts differently, so BDIBottleneck's top-1 accuracy differs slightly
  between the two figures (e.g. 57.0% vs. 55.6% at full reveal on `test`). Compare models
  within a figure.
- **Appendix diagnostics across hardware.** The appendix models were evaluated on CUDA for the
  paper; rerunning on other hardware reproduces every paper table exactly, while the
  underlying diagnostic files differ by floating-point noise (≈10⁻⁶). A few threshold-borderline
  values flip: one of 6,504 dense dose-response episodes changes its monotonicity flag, and in two
  robustness runs one belief entry out of ~9,000 in a held-out-belief split crosses the 0.5
  threshold. Neither enters a reported number.

---

## License

The code is released under the [MIT License](LICENSE).

Questions and issues: please open a GitHub issue, or contact Nigel Doering (nfdoerin@ucsd.edu)
or Rahath Malladi (ramalladi@ucsd.edu).
