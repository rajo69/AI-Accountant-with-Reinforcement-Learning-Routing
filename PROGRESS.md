# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Phase 4 — Integration and API

## Phase Status
COMPLETE

## Last Completed Step
Phase 4 complete. integration/router.py, api/main.py, api/schemas.py,
integration/tests/test_router.py (4/4 passing), INTEGRATION_GUIDE.md.
All committed.

## Next Step
Phase 5, Step 1: Create .github/workflows/ci.yml (test + lint + type-check jobs).
Then .github/workflows/train.yml (manually triggered retraining workflow).
Then railway.toml and Dockerfile. Then DEPLOYMENT.md.

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [x] Phase 1 — Environment and synthetic data
- [x] Phase 2 — RL agent training
- [x] Phase 3 — Evaluation and comparison
- [x] Phase 4 — Integration and API
- [ ] Phase 5 — CI/CD and deployment
- [ ] Phase 6 — README and documentation

## Open Issues
- data/synthetic/transactions.jsonl contains dry-run mock data (no real API calls).
  Run `python -m environment.transaction_simulator` with ANTHROPIC_API_KEY set
  to generate real confidence scores for production-quality training.
- PPO-A and PPO-B produced identical evaluation results at neutral load.
  Variant B only differs at high vs low accountant_load values.

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

Best model: PPO-C (highest precision, lowest error rate, no REJECT actions).
Default in LearnedRouter.

## Session Log
### Session 1 — 2026-03-25
- Started: No PROGRESS.md existed; Phase 0 not yet started
- Completed: Phases 0 and 1 — env, reward functions, simulator, 17 tests
- Ended: Phase 1 COMPLETE

### Session 2 — 2026-03-25
- Started: Phase 1 complete, Phase 2 not started
- Completed: Phase 2 — policy_config.yaml, train.py, baseline.py,
  all three PPO variants trained (100k steps each)
- Ended: Phase 2 COMPLETE

### Session 3 — 2026-03-25
- Started: Phase 2 complete, Phase 3 not started
- Completed: Phase 3 — evaluate.py, analysis.ipynb, comparison_report.md,
  4 figures. Held-out evaluation on 177 transactions.
- Ended: Phase 3 COMPLETE

### Session 4 — 2026-03-25
- Started: Phase 3 complete, Phase 4 not started
- Completed: Phase 4 — integration/router.py (LearnedRouter with fallback),
  api/main.py + api/schemas.py (FastAPI, 4 endpoints),
  integration/tests/test_router.py (4/4 passing),
  integration/INTEGRATION_GUIDE.md
- Ended: Phase 4 COMPLETE
