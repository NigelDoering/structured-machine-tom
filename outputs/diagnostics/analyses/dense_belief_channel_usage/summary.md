# Belief Channel Usage Summary

Each variant holds predicted desire and `z_tau` fixed and changes only the belief channel.

- Mean KL from normal under all-open masking: 0.054228.
- Mean KL from normal under shuffled beliefs: 0.010575.
- Mean oracle true-goal probability: 0.124558.
- Belief-relevant subset rows written: 384. Compare KL and accuracy changes against the `all` subset.

Interpretation guide:
- Tiny all-open/shuffle KL values indicate weak aggregate belief usage.
- Larger effects on belief-relevant subsets indicate conditional belief usage.
- Oracle improvements over predicted belief indicate the belief head is the limiting factor.
- No oracle improvement suggests the goal head or environment does not strongly rely on belief.
