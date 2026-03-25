# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Phase 5 — CI/CD and Deployment

## Phase Status
COMPLETE

## Last Completed Step
Phase 5 complete. CI/CD workflows, Dockerfile, railway.toml, DEPLOYMENT.md committed.

## Next Step
Phase 6, Step 1: Write README.md (research-grade, per the spec structure:
Research Question, Background, Methodology, Results, Key Findings, Limitations,
Implications, Repository Structure, Quickstart, Citing This Work, Licence).
Then CONTRIBUTING.md and RESEARCH_NOTES.md.
Final commit, tag v1.0.0, push tags.

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [x] Phase 1 — Environment and synthetic data
- [x] Phase 2 — RL agent training
- [x] Phase 3 — Evaluation and comparison
- [x] Phase 4 — Integration and API
- [x] Phase 5 — CI/CD and deployment
- [ ] Phase 6 — README and documentation

## Open Issues
- data/synthetic/transactions.jsonl contains dry-run mock data (no real API calls).
  Run `python -m environment.transaction_simulator` with ANTHROPIC_API_KEY set
  for production-quality training data.
- PPO-A and PPO-B produced identical evaluation results at neutral load.

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

## Session Log
### Session 1 — 2026-03-25
- Completed: Phases 0 and 1 — env, reward functions, simulator, 17 tests
- Ended: Phase 1 COMPLETE

### Session 2 — 2026-03-25
- Completed: Phase 2 — policy_config.yaml, train.py, baseline.py, 3 PPO variants trained
- Ended: Phase 2 COMPLETE

### Session 3 — 2026-03-25
- Completed: Phase 3 — evaluate.py, analysis.ipynb, comparison_report.md, 4 figures
- Ended: Phase 3 COMPLETE

### Session 4 — 2026-03-25
- Completed: Phase 4 — integration/router.py, api/main.py+schemas.py, 4 tests passing, INTEGRATION_GUIDE.md
- Ended: Phase 4 COMPLETE

### Session 5 — 2026-03-25
- Started: Phase 4 complete, Phase 5 not started
- Completed: Phase 5 — ci.yml (test+lint+type-check), train.yml (manual retrain),
  Dockerfile, railway.toml, DEPLOYMENT.md
- Ended: Phase 5 COMPLETE
