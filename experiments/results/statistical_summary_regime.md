## Results with 95% confidence intervals (n=160)

All intervals are Wilson score 95% CIs. p-values are two-sided two-proportion z-tests (Baseline vs PPO).

| Metric | Baseline (0.85/0.50) | PPO (A representative; A/B/C diverge — see notes) | Significance |
|---|:---:|:---:|:---:|
| Routing accuracy (over 160) | 43.1% [35.7, 50.9] | 59.4% [51.6, 66.7] | p=0.00 |
| Auto-approval precision | 100.0% [70.1, 100.0] | 71.9% [59.9, 81.4] | p=0.07 |
| Auto-approval error rate | 0.0% [0.0, 29.9] | 28.1% [18.6, 40.1] | p=0.07 |
| Auto-approval rate | 5.6% [3.0, 10.3] | 40.0% [32.7, 47.7] | p=0.00 |

### Per-tier auto-approval error rate (auto-approved subset only)

| Tier | Baseline | PPO |
|---|:---:|:---:|
| easy | 0.0% [0.0, 29.9] (n=9) | 28.1% [18.6, 40.1] (n=64) |
| medium | N/A | N/A (no auto-approvals) |
| hard | N/A | N/A (no auto-approvals) |

### Interpretation

- PPO A/B/C DIVERGE on this eval set. Per-variant action totals (AUTO/SURFACE/REJECT): A=64/96/0, B=64/96/0, C=0/160/0. The main Baseline-vs-PPO table below uses Variant A as the PPO representative; the per-variant divergence is the headline finding.
- The baseline emits 18 REJECT_FOR_MANUAL decisions on this eval set while PPO emits 0. The PPO policies assign zero probability to REJECT regardless of input; the baseline uses REJECT whenever the thresholded score falls below its lower threshold (0.50 on raw, not recalibrated for calibrated input).
- The Baseline-vs-PPO difference in auto-approval error rate is NOT statistically significant at alpha=0.05.
- The Baseline-vs-PPO difference in auto-approval RATE is highly significant: the two policies sit at very different points on the coverage/precision tradeoff curve.
