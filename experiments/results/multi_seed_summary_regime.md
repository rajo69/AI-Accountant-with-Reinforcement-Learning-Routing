# Multi-seed robustness — dataset `regime`

Each row reports mean across 5 seeds (0, 1, 2, 3, 4), with the population standard deviation in parentheses. Identical action totals across all seeds (where listed) indicate that the tier-level policy is seed-invariant for the given (variant, regime).

| Variant | Routing accuracy | Auto-precision | Auto-rate | Auto-error-rate | Distinct action totals |
|---|:---:|:---:|:---:|:---:|---|
| PPO-A | 59.4% (0.00pp) | 71.9% (0.00pp) | 40.0% (0.00pp) | 28.1% (0.00pp) | AUTO=64, SURFACE=96, REJECT=0 (all seeds) |
| PPO-B | 59.4% (0.00pp) | 71.9% (0.00pp) | 40.0% (0.00pp) | 28.1% (0.00pp) | AUTO=64, SURFACE=96, REJECT=0 (all seeds) |
| PPO-C | 41.9% (0.00pp) | N/A | 0.0% (0.00pp) | N/A | AUTO=0, SURFACE=160, REJECT=0 (all seeds) |

Per-seed raw numbers are in `multi_seed_summary_regime.json`.