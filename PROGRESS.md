# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Phase 6 — README and Documentation

## Phase Status
COMPLETE

## Last Completed Step
All phases complete. README.md, CONTRIBUTING.md, RESEARCH_NOTES.md written.
Tagged v1.0.0. All commits pushed.

## Next Step
Project complete. No further phases.

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [x] Phase 1 — Environment and synthetic data
- [x] Phase 2 — RL agent training
- [x] Phase 3 — Evaluation and comparison
- [x] Phase 4 — Integration and API
- [x] Phase 5 — CI/CD and deployment
- [x] Phase 6 — README and documentation

## Open Issues
- data/synthetic/transactions.jsonl uses mock confidence scores.
  Run `python -m environment.transaction_simulator` with ANTHROPIC_API_KEY
  to generate real scores for production-quality replication.

## Training Results Summary (Phase 2)
| Variant | Description          | Final Mean Reward | Training Time |
|---------|----------------------|-------------------|---------------|
| A       | Binary asymmetric    | 291.1 ± 0.0       | ~10 min       |
| B       | Workload-weighted    | 281.1 ± 6.1       | ~24 min       |
| C       | Conservative (-5.0)  | 67.3 ± 0.0        | ~24 min       |

## Evaluation Results Summary (Phase 3)
| Policy   | Routing Acc | Auto-Prec | Auto-Rate | Error Rate |
|----------|-------------|-----------|-----------|------------|
| Baseline | 44.6%       | 89.4%     | 37.3%     | 10.6%      |
| PPO-A    | 75.1%       | 81.2%     | 81.4%     | 18.8%      |
| PPO-B    | 75.1%       | 81.2%     | 81.4%     | 18.8%      |
| PPO-C    | 61.0%       | 90.1%     | 45.8%     |  9.9%      |

Best model: PPO-C. Deployed as default in LearnedRouter.

## Session Log
### Session 1 — 2026-03-25
- Completed: Phases 0 and 1 — env, reward functions, simulator, 17 tests

### Session 2 — 2026-03-25
- Completed: Phase 2 — policy_config.yaml, train.py, baseline.py, 3 PPO variants

### Session 3 — 2026-03-25
- Completed: Phase 3 — evaluate.py, analysis.ipynb, comparison_report.md, 4 figures

### Session 4 — 2026-03-25
- Completed: Phase 4 — integration/router.py, api/main.py+schemas.py, 4 tests, INTEGRATION_GUIDE.md

### Session 5 — 2026-03-25
- Completed: Phase 5 — ci.yml, train.yml, Dockerfile, railway.toml, DEPLOYMENT.md

### Session 6 — 2026-03-25
- Started: Phase 5 complete, Phase 6 not started
- Completed: Phase 6 — README.md, CONTRIBUTING.md, RESEARCH_NOTES.md, v1.0.0 tag
- Ended: ALL PHASES COMPLETE
