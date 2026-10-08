# Model-Planned Counterfactual Path Summary

- Model-planned path rows: 23040.
- Successful model-planned routes: 23040.
- Goal-change rate vs original goal: 0.8474.
- Goal-change rate vs normal model top-1: 0.5063.
- Mean node overlap with original future suffix: 0.3242.
- Mean edge overlap with original future suffix: 0.2928.
- Simulator comparison success rate: 1.0000.
- Simulator counterfactual goal top-1 agreement: 0.1399.
- Simulator counterfactual goal top-5 agreement: 0.4045.

This diagnostic freezes BDIBottleneck, edits belief/desire channels at an observed prefix, lets the model pick a counterfactual goal, and routes from the intervention node to that model-planned goal.
