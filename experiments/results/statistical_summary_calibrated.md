## Results with 95% confidence intervals (n=177)

All intervals are Wilson score 95% CIs. p-values are two-sided two-proportion z-tests (Baseline vs PPO).

| Metric | Baseline (0.85/0.50) | PPO (A/B/C same totals) | Significance |
|---|:---:|:---:|:---:|
| Routing accuracy (over 177) | 42.4% [35.3, 49.7] | 63.3% [56.0, 70.0] | p<0.001 |
| Auto-approval precision | 100.0% [79.6, 100.0] | 77.8% [67.6, 85.5] | p=0.043 |
| Auto-approval error rate | 0.0% [0.0, 20.4] | 22.2% [14.5, 32.4] | p=0.043 |
| Auto-approval rate | 8.5% [5.2, 13.5] | 45.8% [38.6, 53.1] | p<0.001 |

### Per-tier auto-approval error rate (auto-approved subset only)

| Tier | Baseline | PPO |
|---|:---:|:---:|
| easy | 0.0% [0.0, 20.4] (n=15) | 22.2% [14.5, 32.4] (n=81) |
| medium | N/A | N/A (no auto-approvals) |
| hard | N/A | N/A (no auto-approvals) |

### Interpretation

- PPO A/B/C produce the same action totals on the eval set (total AUTO=81, SURFACE=96, REJECT=0), so one PPO column is reported in comparison tables (equal totals, not a check that every item gets the same action).
- The baseline emits 18 REJECT_FOR_MANUAL decisions on this eval set while PPO emits 0. The PPO policies do not choose REJECT: under the reward tables it beats sending for review only when the chance of a correct prediction is below about 0.22 under Variants A and C (at most 0.33 under B), and no tier in this data is that low. The baseline uses REJECT whenever the thresholded score falls below its lower threshold (0.50 on raw, not recalibrated for calibrated input).
- The Baseline-vs-PPO difference in auto-approval RATE has an item-level p below 0.05: the two policies sit at different points on the coverage/precision tradeoff. Item-level tests treat items as independent, but items cluster in seed scenarios; see the scenario-level comparison in the README before quoting this.
