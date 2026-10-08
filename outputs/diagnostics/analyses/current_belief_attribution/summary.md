# Current Belief Attribution Summary

Closed-class metrics treat false beliefs as the positive class.

- Overall belief accuracy: 0.6751.
- Closed-class precision/recall/F1: 0.2774 / 0.6072 / 0.3808.
- Macro-F1: 0.5803; balanced accuracy: 0.6478.
- Closed-entry proportion: 0.1646 (9480 closed of 57600 entries).

## By Split
- test: closed F1=0.3899, PR-AUC=0.3005, closed_prop=0.1354.
- test_bh: closed F1=0.3712, PR-AUC=0.2986, closed_prop=0.2083.
- test_new_agent: closed F1=0.3957, PR-AUC=0.3091, closed_prop=0.1354.
- test_new_agent_bh: closed F1=0.3717, PR-AUC=0.2926, closed_prop=0.2083.
- val: closed F1=0.3788, PR-AUC=0.2733, closed_prop=0.1354.

## Interpretation
- The label distribution is sparse and majority-open, so overall belief accuracy is expected to overstate closed-class performance.
- Low closed-class F1 should be interpreted together with PR-AUC, closed-entry prevalence, and evidence-conditioned breakdowns.
- This file is a diagnostic summary.
