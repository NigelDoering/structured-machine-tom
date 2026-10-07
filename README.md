# Structured Machine Theory of Mind from Agent Trajectories

Official code and data for **Structured Machine Theory of Mind from Agent Trajectories**,
published in *Transactions on Machine Learning Research* (TMLR), 2026.

Nigel F. Doering\*, Rahath Malladi\*, Arshia Sangwan, M Tanjid Hasan Tonmoy, Jacob Doan,
David Danks, Tauhidur Rahman
(\* equal contribution)

Reviews: [OpenReview](https://openreview.net/forum?id=PfSNLg0zcK)

> **Status:** the code, simulator, dataset splits, and trained checkpoints are being prepared for
> release and will be added to this repository shortly.

## Overview

SMToM (Structured Machine Theory of Mind) attributes explicit belief and desire representations
to an agent from its observed trajectories, within a Belief–Desire–Intention causal structure.
The model, **BDIBottleneck**, predicts the agent's goal only through its inferred belief and
desire channels, so counterfactual questions such as "what would this agent do if it believed
this location were closed?" become a direct intervention on the model.
