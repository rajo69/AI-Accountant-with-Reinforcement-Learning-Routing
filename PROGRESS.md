# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Between phases. All critical research phases (0–8b) and defensibility work
are complete and merged to main. Phase 9 (multi-seed robustness) is the
remaining planned phase; next candidate is a Docker-reproducibility pass
to make the headline numbers regenerable with one command.

## Phase Status
All research phases complete. Initial project tag v1.0.0 was cut after
Phase 6; subsequent phases strengthen the evaluation and defensibility
rather than add features. The repository is usable as-is at HEAD and also
at v1.0.0, with the HEAD framing being substantially more defensible.

## Last Completed Step
Phase 10 repository cleanup shipped via PR #1 (merge commit 404f8fb):
vendored parent project removed (~1MB), CLAUDE.md and RL_ROUTING_PROJECT.md
trimmed to current-state scaffolding, seed fixtures relocated to data/seeds/.
The RL extension is now fully decoupled from the parent project's source.

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
- [x] Phase 8b — Regime probe on calibrated and raw confidence (EV-invariance
      mechanism confirmed on both signal regimes; commits 894cf82 + 2e80cb4)
- [x] Phase 10 — Repository cleanup: removed vendored parent project
      (backend/, frontend/, deployment configs); trimmed CLAUDE.md and
      RL_ROUTING_PROJECT.md to current-state scaffolding; seed fixtures
      relocated to data/seeds/
- [ ] Phase 9 — Multi-seed training (planned; hygiene)

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

Two-proportion z-tests: routing accuracy p=0.504, precision p=0.410, error
rate p=0.410 (none significant at n=177). Auto-approval rate p<0.001 (highly
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

## Regime Probe Summary (Phase 8b)
Reshapes the easy tier to p=0.72 (inside the A-vs-C EV-divergence band of
(0.64, 0.80)). On BOTH the calibrated and raw regimes, Variant C diverges
from A/B exactly as the EV math predicts: A and B auto-approve all 64 easy;
C auto-approves zero and surfaces all 160. Divergence replicating on both
signal regimes rules out a calibration-specific artefact and supports the
EV-invariance explanation for natural-regime convergence. See README
"Regime probe" subsection.

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
- Completed: Regime probe — reshape easy tier to 0.72 (EV-divergence band), retrain A/B/C on calibrated and raw regimes, confirm variant divergence exactly as EV math predicts on BOTH regimes (commits 894cf82 + 2e80cb4, including a defensibility sweep across README / RESEARCH_NOTES / data card / paper / superseded comparison report)

### Session 11 — 2026-04-20
- Completed: Phase 10 repository cleanup — removed vendored parent project (backend/, frontend/, Dockerfile, docker-compose.yml, railway.toml, DEPLOYMENT.md, docs/ARCHITECTURE.md); trimmed CLAUDE.md to a 31-line pointer and RL_ROUTING_PROJECT.md to 73 lines of research framing; relocated seed fixtures to data/seeds/ so the synthetic-data regeneration pipeline still works; added prerequisite pointer to the parent repository in integration/INTEGRATION_GUIDE.md
