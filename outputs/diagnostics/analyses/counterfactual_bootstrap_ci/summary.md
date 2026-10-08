# Counterfactual Bootstrap CI Summary

- Stages summarized: 6.
- CI rows: 2256.
- Metrics: cf_prob_original_goal, cf_prob_simulator_goal, cf_prob_top1_goal, edge_overlap_original, edge_overlap_simulator, goal_changed_vs_normal, goal_changed_vs_original, node_overlap_original, node_overlap_simulator, planned_success, prob_counterfactual_goal, route_length_ratio_original, route_length_ratio_simulator, simulator_goal_top1_match, simulator_goal_top5_match, simulator_success, top1_correct, top5_correct.

Intervals are 95% percentile bootstrap intervals clustered by `agent_id` when available and `record_index` otherwise.

## Selected Overall Rows

- true_counterfactual_rollouts / simulator_resampled_rollout / top1_correct / belief / : mean=0.1750, 95% CI=[0.1333, 0.2208].
- true_counterfactual_rollouts / simulator_resampled_rollout / top1_correct / desire / : mean=0.1229, 95% CI=[0.0854, 0.1688].
- dense_true_counterfactual_rollouts / simulator_resampled_rollout / top1_correct / belief / : mean=0.1948, 95% CI=[0.1583, 0.2313].
- dense_true_counterfactual_rollouts / simulator_resampled_rollout / top1_correct / desire / : mean=0.1969, 95% CI=[0.1646, 0.2292].
- model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / belief / oracle_channel: mean=0.6729, 95% CI=[0.6188, 0.7271].
- model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / belief / predicted_channel: mean=0.7167, 95% CI=[0.6562, 0.7771].
- model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / desire / oracle_channel: mean=0.8854, 95% CI=[0.8479, 0.9229].
- model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / desire / predicted_channel: mean=0.8917, 95% CI=[0.8500, 0.9292].
- model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / belief / oracle_channel: mean=0.0563, 95% CI=[0.0333, 0.0792].
- model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / belief / predicted_channel: mean=0.0625, 95% CI=[0.0312, 0.0958].
- model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / desire / oracle_channel: mean=0.1271, 95% CI=[0.0917, 0.1604].
- model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / desire / predicted_channel: mean=0.1229, 95% CI=[0.0875, 0.1583].
- dense_model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / belief / oracle_channel: mean=0.6979, 95% CI=[0.6594, 0.7354].
- dense_model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / belief / predicted_channel: mean=0.6979, 95% CI=[0.6594, 0.7354].
- dense_model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / desire / oracle_channel: mean=0.8635, 95% CI=[0.8167, 0.9094].
- dense_model_planned_counterfactual_paths / model_planned_goal / goal_changed_vs_original / desire / predicted_channel: mean=0.8719, 95% CI=[0.8354, 0.9073].
- dense_model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / belief / oracle_channel: mean=0.0563, 95% CI=[0.0396, 0.0750].
- dense_model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / belief / predicted_channel: mean=0.0500, 95% CI=[0.0333, 0.0688].
- dense_model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / desire / oracle_channel: mean=0.1437, 95% CI=[0.1115, 0.1781].
- dense_model_planned_counterfactual_paths / model_vs_simulator / simulator_goal_top1_match / desire / predicted_channel: mean=0.1635, 95% CI=[0.1250, 0.2021].
- robustness_model_planned_sparse / model_planned_goal / goal_changed_vs_original / belief / oracle_channel: mean=0.7675, 95% CI=[0.7242, 0.8117].
- robustness_model_planned_sparse / model_planned_goal / goal_changed_vs_original / belief / predicted_channel: mean=0.7754, 95% CI=[0.7304, 0.8187].
- robustness_model_planned_sparse / model_planned_goal / goal_changed_vs_original / desire / oracle_channel: mean=0.8554, 95% CI=[0.8296, 0.8804].
- robustness_model_planned_sparse / model_planned_goal / goal_changed_vs_original / desire / predicted_channel: mean=0.8479, 95% CI=[0.8217, 0.8729].
