# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Phase 8 — Probe experiments (calibration and EV-divergence regime)

## Phase Status
In progress. Initial project tag v1.0.0 was cut after Phase 6; subsequent
phases strengthen the evaluation rather than add features. The repository is
usable as-is at HEAD and also at v1.0.0, with the HEAD framing being
substantially more defensible.

## Last Completed Step
Calibration probe shipped (commits 9baba1c, 04f3c2e). Regime probe on
calibrated confidence shipped (variant divergence confirmed). Regime probe
on raw confidence running as a robustness check.

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [x] Phase 1 — Environment and synthetic data
- [x] Phase 2 — RL agent training
- [x] Phase 3 — Evaluation and comparison
- [x] Phase 4 — Integration and API
- [x] Phase 5 — CI/CD and deployment
- [x] Phase 6 — README and documentation
- [x] Phase 7 — Statistical rigour (Wilson CIs, two-proportion z-tests,
      Key Finding #3 reframe)
- [x] Phase 7b — 1-page technical note companion (paper/)
- [x] Phase 8a — Calibration probe (Platt scaling; Key Finding #4 reframe)
- [ ] Phase 8b — Regime probe on calibrated and raw confidence (in progress)
- [ ] Phase 9 — Multi-seed training (planned; hygiene)
- [ ] Phase 10 — Repository cleanup (remove parent project from this repo)

## Open Issues
- Small evaluation set: 177 transactions (33 hard) from 50 labelled seeds
  × ~18 variants each. The effective unit of diversity is closer to 50
  scenarios than to 882 independent draws. Flagged in README Limitations.
- Variant A vs Variant B divergence at this sample size is not cleanly
  achievable: the EV break-even thresholds differ by only 0.026 (0.639 vs
  0.613), narrower than the sampling noise allows at n≈64 per tier.
  Flagged in README (Regime probe section) as future work.
- `experiments/results/comparison_report.md` was superseded; its placeholder
  now points to the authoritative current sources.

## Training Results Summary (raw confidence scores, Phase 2)
| Variant | Description          | Final Mean Reward | Training Time |
|---------|----------------------|-------------------|---------------|
| A       | Binary asymmetric    | 140.5 ± 0.0       | ~9 min        |
| B       | Workload-weighted    | 110.9 ± 18.3      | ~9 min        |
| C       | Conservative (-5.0)  | -27.5 ± 0.0       | ~13 min       |

All three PPO variants produced identical action distributions on the
177-transaction held-out set (AUTO 81 / SURFACE 96 / REJECT 0).

## Evaluation Results Summary (raw held-out set, Phase 3 + 7)
| Policy       | Routing Acc    | Auto-Prec      | Auto-Rate      | Error Rate      |
|--------------|---------------:|---------------:|---------------:|----------------:|
| Baseline     | 66.7% [59, 73] | 72.6% [64, 80] | 63.8% [57, 71] | 27.4% [20, 36]  |
| PPO-A / B / C| 63.3% [56, 70] | 77.8% [68, 86] | 45.8% [39, 53] | 22.2% [15, 32]  |

Two-proportion z-tests: routing accuracy p=0.50, precision p=0.41, error rate
p=0.41 (none significant at n=177). Auto-approval rate p=0.001 (highly
significant). All three PPO variants are identical on the eval set, reported
as one column. See `experiments/results/statistical_summary.md`.

## Calibration Probe Summary (Phase 8a)
Platt-scaled logistic regression on [raw_conf, amount_normalised,
tier_onehot] → is_correct, fit via 5-fold stratified CV. Improves proper
scoring rules on eval (Brier 0.295 → 0.215, log-loss 0.910 → 0.616).
Under calibrated confidence, all three PPO variants STILL produce identical
action sequences (AUTO 81 / SURFACE 96 / REJECT 0). Calibration alone does
not resolve the A/B/C convergence. See Results → Calibration probe in the
main README.

## Regime Probe Summary (Phase 8b, in progress)
Reshapes the easy tier to p=0.72 (inside the A-vs-C EV-divergence band of
(0.64, 0.80)). On the calibrated regime, Variant C diverges from A/B as the
EV math predicts: A and B auto-approve all 64 easy; C auto-approves zero and
surfaces all 160. Raw-confidence robustness check pending.

## Session Log
### Session 1 — 2026-03-25
- Completed: Phases 0 and 1 — env, reward functions, simulator, 17 tests

### Session 2 — 2026-03-25
- Completed: Phase 2 — policy_config.yaml, train.py, baseline.py, 3 PPO variants

### Session 3 — 2026-03-25
- Completed: Phase 3 — evaluate.py, analysis.ipynb, 4 figures

### Session 4 — 2026-03-25
- Completed: Phase 4 — integration/router.py, api/main.py+schemas.py, 4 tests, INTEGRATION_GUIDE.md

### Session 5 — 2026-03-25
- Completed: Phase 5 — ci.yml, train.yml, Dockerfile, railway.toml, DEPLOYMENT.md

### Session 6 — 2026-03-25
- Completed: Phase 6 — README.md, CONTRIBUTING.md, RESEARCH_NOTES.md, v1.0.0 tag

### Session 7 — 2026-03-26
- Completed: Re-ran simulator with real Claude Haiku API; committed real-confidence datasets and retrained all three variants

### Session 8 — 2026-04-20
- Completed: CI pipeline fix (scoped ruff/mypy/pytest to extension dirs), statistical rigour (Wilson CIs + two-proportion z-tests), Key Finding #3 reframe (commit 441d174); 1-page technical note companion (commit 4ea4bb2)

### Session 9 — 2026-04-20
- Completed: Calibration probe — Platt-scaled logistic regression on training set, full retrain of A/B/C, Key Finding #4 reframe from "calibration is the binding constraint" to "necessary but not sufficient" (commits 9baba1c, 04f3c2e)

### Session 10 — 2026-04-20
- In progress: Regime probe — reshape easy tier to 0.72 (EV-divergence band), retrain A/B/C on calibrated and raw regimes, confirm variant divergence exactly as EV math predicts
