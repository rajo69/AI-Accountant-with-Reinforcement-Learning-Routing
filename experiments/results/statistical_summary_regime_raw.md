## Results with 95% confidence intervals (n=160)

All intervals are Wilson score 95% CIs. p-values are two-sided two-proportion z-tests (Baseline vs PPO).

| Metric | Baseline (0.85/0.50) | PPO (A representative; A/B/C diverge — see notes) | Significance |
|---|:---:|:---:|:---:|
| Routing accuracy (over 160) | 63.7% [56.1, 70.8] | 59.4% [51.6, 66.7] | p=0.42 |
| Auto-approval precision | 68.0% [58.2, 76.5] | 71.9% [59.9, 81.4] | p=0.60 |
| Auto-approval error rate | 32.0% [23.5, 41.8] | 28.1% [18.6, 40.1] | p=0.60 |
| Auto-approval rate | 60.6% [52.9, 67.9] | 40.0% [32.7, 47.7] | p=0.00 |

### Per-tier auto-approval error rate (auto-approved subset only)

| Tier | Baseline | PPO |
|---|:---:|:---:|
| easy | 25.0% [15.5, 37.7] (n=56) | 28.1% [18.6, 40.1] (n=64) |
| medium | 54.8% [37.8, 70.8] (n=31) | N/A (no auto-approvals) |
| hard | 0.0% [0.0, 27.8] (n=10) | N/A (no auto-approvals) |

### Interpretation

- PPO A/B/C DIVERGE on this eval set. Per-variant action totals (AUTO/SURFACE/REJECT): A=64/96/0, B=64/96/0, C=0/160/0. The main Baseline-vs-PPO table below uses Variant A as the PPO representative; the per-variant divergence is the headline finding.
- Both policies emit 0 REJECT_FOR_MANUAL on this eval set. For the raw-confidence regime, this is because real Claude Haiku confidences never fall below 0.5 (the baseline's reject threshold); PPO's REJECT-elimination is a learned policy-level property invariant to input confidence, whereas the baseline's zero rejects reflect the data distribution.
- The Baseline-vs-PPO difference in auto-approval error rate is NOT statistically significant at alpha=0.05.
- The Baseline-vs-PPO difference in auto-approval RATE is highly significant: the two policies sit at very different points on the coverage/precision tradeoff curve.
