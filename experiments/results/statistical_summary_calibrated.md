## Results with 95% confidence intervals (n=177)

All intervals are Wilson score 95% CIs. p-values are two-sided two-proportion z-tests (Baseline vs PPO).

| Metric | Baseline (0.85/0.50) | PPO (A/B/C identical) | Significance |
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

- PPO A/B/C produce identical action sequences on the eval set (total AUTO=81, SURFACE=96, REJECT=0), so only one PPO column needs reporting in comparison tables.
- The baseline emits 18 REJECT_FOR_MANUAL decisions on this eval set while PPO emits 0. The PPO policies assign zero probability to REJECT regardless of input; the baseline uses REJECT whenever the thresholded score falls below its lower threshold (0.50 on raw, not recalibrated for calibrated input).
- The Baseline-vs-PPO difference in auto-approval RATE is highly significant: the two policies sit at very different points on the coverage/precision tradeoff curve.
