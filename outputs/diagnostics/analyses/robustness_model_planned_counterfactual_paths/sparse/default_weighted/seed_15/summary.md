# Model-Planned Counterfactual Path Summary

- Model-planned path rows: 11520.
- Successful model-planned routes: 11520.
- Goal-change rate vs original goal: 0.8643.
- Goal-change rate vs normal model top-1: 0.4797.
- Mean node overlap with original future suffix: 0.2306.
- Mean edge overlap with original future suffix: 0.1881.
- Simulator comparison success rate: 1.0000.
- Simulator counterfactual goal top-1 agreement: 0.0968.
- Simulator counterfactual goal top-5 agreement: 0.3880.

This diagnostic freezes BDIBottleneck, edits belief/desire channels at an observed prefix, lets the model pick a counterfactual goal, and routes from the intervention node to that model-planned goal.
