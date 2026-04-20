## Results with 95% confidence intervals (n=177)

All intervals are Wilson score 95% CIs. p-values are two-sided two-proportion z-tests (Baseline vs PPO).

| Metric | Baseline (0.85/0.50) | PPO (A/B/C identical) | Significance |
|---|:---:|:---:|:---:|
| Routing accuracy (over 177) | 66.7% [59.4, 73.2] | 63.3% [56.0, 70.0] | p=0.504 |
| Auto-approval precision | 72.6% [63.7, 79.9] | 77.8% [67.6, 85.5] | p=0.410 |
| Auto-approval error rate | 27.4% [20.1, 36.3] | 22.2% [14.5, 32.4] | p=0.410 |
| Auto-approval rate | 63.8% [56.5, 70.6] | 45.8% [38.6, 53.1] | p<0.001 |

### Per-tier auto-approval error rate (auto-approved subset only)

| Tier | Baseline | PPO |
|---|:---:|:---:|
| easy | 19.4% [12.0, 30.0] (n=72) | 22.2% [14.5, 32.4] (n=81) |
| medium | 54.8% [37.8, 70.8] (n=31) | N/A (no auto-approvals) |
| hard | 0.0% [0.0, 27.8] (n=10) | N/A (no auto-approvals) |

### Interpretation

- PPO A/B/C produce identical action sequences on the eval set (total AUTO=81, SURFACE=96, REJECT=0), so only one PPO column needs reporting in comparison tables.
- Both policies emit 0 REJECT_FOR_MANUAL on this eval set. For the raw-confidence regime, this is because real Claude Haiku confidences never fall below 0.5 (the baseline's reject threshold); PPO's REJECT-elimination is a learned policy-level property invariant to input confidence, whereas the baseline's zero rejects reflect the data distribution.
- The Baseline-vs-PPO difference in auto-approval error rate is NOT statistically significant at alpha=0.05.
- The Baseline-vs-PPO difference in auto-approval RATE is highly significant: the two policies sit at very different points on the coverage/precision tradeoff curve.
