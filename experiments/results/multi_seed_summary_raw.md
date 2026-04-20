# Multi-seed robustness — dataset `raw`

Each row reports mean across 5 seeds (0, 1, 2, 3, 4), with the population standard deviation in parentheses. Identical action totals across all seeds (where listed) indicate that the tier-level policy is seed-invariant for the given (variant, regime).

| Variant | Routing accuracy | Auto-precision | Auto-rate | Auto-error-rate | Distinct action totals |
|---|:---:|:---:|:---:|:---:|---|
| PPO-A | 63.3% (0.00pp) | 77.8% (0.00pp) | 45.8% (0.00pp) | 22.2% (0.00pp) | AUTO=81, SURFACE=96, REJECT=0 (all seeds) |
| PPO-B | 63.3% (0.00pp) | 77.8% (0.00pp) | 45.8% (0.00pp) | 22.2% (0.00pp) | AUTO=81, SURFACE=96, REJECT=0 (all seeds) |
| PPO-C | 63.3% (0.00pp) | 77.8% (0.00pp) | 45.8% (0.00pp) | 22.2% (0.00pp) | AUTO=81, SURFACE=96, REJECT=0 (all seeds) |

Per-seed raw numbers are in `multi_seed_summary_raw.json`.