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

## Training Results Summary (Phase 2 — re-run with real confidence scores)
| Variant | Description          | Final Mean Reward | Training Time |
|---------|----------------------|-------------------|---------------|
| A       | Binary asymmetric    | 140.5 ± 0.0       | ~9 min        |
| B       | Workload-weighted    | 110.9 ± 18.3      | ~9 min        |
| C       | Conservative (-5.0)  | -27.5 ± 0.0       | ~13 min       |

*(Mock-score run for reference: A=291.1, B=281.1, C=67.3)*

## Evaluation Results Summary (Phase 3 — real confidence scores)
| Policy   | Routing Acc | Auto-Prec | Auto-Rate | Error Rate |
|----------|-------------|-----------|-----------|------------|
| Baseline | 66.7%       | 72.6%     | 63.8%     | 27.4%      |
| PPO-A    | 63.3%       | 77.8%     | 45.8%     | 22.2%      |
| PPO-B    | 63.3%       | 77.8%     | 45.8%     | 22.2%      |
| PPO-C    | 63.3%       | 77.8%     | 45.8%     | 22.2%      |

All three PPO variants converged to the same policy: auto-approve easy tier, surface
medium/hard. Best model: PPO-C (or any variant — results identical). Deployed as default.

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
