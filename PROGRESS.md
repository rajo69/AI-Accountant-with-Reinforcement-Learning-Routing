# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Phase 3 — Evaluation and Comparison

## Phase Status
COMPLETE

## Last Completed Step
Phase 3 complete. evaluate.py run on 177-transaction held-out set.
comparison_report.md auto-generated. 4 analysis figures saved.
All committed.

## Next Step
Phase 4, Step 1: Read parent project CLAUDE.md/ARCHITECTURE.md to understand
API patterns. Create integration/router.py — LearnedRouter class with
route(confidence_score, transaction_features) → RoutingDecision, fallback to
threshold policy if model file missing. Then create api/main.py (FastAPI app
with POST /route, GET /policy/info, GET /health, POST /policy/reload).

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [x] Phase 1 — Environment and synthetic data
- [x] Phase 2 — RL agent training
- [x] Phase 3 — Evaluation and comparison
- [ ] Phase 4 — Integration and API
- [ ] Phase 5 — CI/CD and deployment
- [ ] Phase 6 — README and documentation

## Open Issues
- data/synthetic/transactions.jsonl contains dry-run mock data (no real API calls).
  Run `python -m environment.transaction_simulator` with ANTHROPIC_API_KEY set
  to generate real confidence scores. This explains the tier-based routing behaviour
  in Phase 3 results (policy learned on coarse discrete features).
- PPO-A and PPO-B produced identical evaluation results. On higher-load scenarios
  their behaviour would diverge; evaluate separately if load-sensitive routing matters.

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

Key finding: All PPO variants eliminated REJECT_FOR_MANUAL entirely.
PPO-C is the best balance (highest precision, lowest error rate, no REJECT).

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
  4 figures (fig1–4). Held-out evaluation on 177 transactions.
- Ended: Phase 3 COMPLETE
