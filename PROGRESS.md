# PROGRESS.md — RL Routing Project Session Log

## Current Phase
Phase 2 — RL Agent Training

## Phase Status
COMPLETE

## Last Completed Step
Phase 2 complete. All three PPO variants trained (A, B, C) and committed.
Models saved to models/trained/. Training metadata saved to experiments/results/.

## Next Step
Phase 3, Step 1: Create agent/evaluate.py. Load held-out set from
data/evaluation/held_out_set.json. Evaluate baseline (ThresholdPolicy) and
all three PPO variants. Compute: routing accuracy, auto-approval precision,
auto-approval rate, unnecessary escalation rate, error rate by difficulty tier,
confusion matrix. Save to experiments/results/evaluation_results.json.

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [x] Phase 1 — Environment and synthetic data
- [x] Phase 2 — RL agent training
- [ ] Phase 3 — Evaluation and comparison
- [ ] Phase 4 — Integration and API
- [ ] Phase 5 — CI/CD and deployment
- [ ] Phase 6 — README and documentation

## Open Issues
- data/synthetic/transactions.jsonl contains dry-run mock data (no real API calls).
  For real confidence scores, run `python -m environment.transaction_simulator`
  with ANTHROPIC_API_KEY set. Training was done on mock data.
- Variant B training was slower (~24 min vs 10 min for A/C) because parallel
  training competed for CPU. Sequential training recommended in future.

## Training Results Summary (Phase 2)
| Variant | Description          | Final Mean Reward | Training Time |
|---------|----------------------|-------------------|---------------|
| A       | Binary asymmetric    | 291.1 ± 0.0       | ~10 min       |
| B       | Workload-weighted    | 281.1 ± 6.1       | ~24 min       |
| C       | Conservative (-5.0)  | 67.3 ± 0.0        | ~24 min       |

Notes:
- Variant A: converged early (~20k steps), stable mean reward of 291.1
- Variant B: slightly lower reward due to load-dependent escalation penalty;
  higher variance (±6.1) consistent with stochastic accountant_load state
- Variant C: significantly lower reward (67.3) as expected — heavy -5.0 penalty
  forces conservative routing with fewer auto-approvals, fewer +1.0 rewards
- All three model files are small (~140KB each) and committed to the repo

## Session Log
### Session 1 — 2026-03-25
- Started: No PROGRESS.md existed; Phase 0 not yet started
- Completed: Context files read, architecture confirmed, directory structure created, CLAUDE.md updated, .gitignore and .env.example created
- Ended: Phase 0 COMPLETE. Repo at https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing.git

### Session 1 cont. — 2026-03-25
- Completed: Phase 1 — RoutingEnv, reward_functions (A/B/C), transaction_simulator, 17 tests
- Synthetic data: 882 total records (705 train / 177 eval), dry-run verified
  - Easy: n=322 correct=89.1% mean_conf=0.900
  - Medium: n=252 correct=77.0% mean_conf=0.650
  - Hard: n=131 correct=55.0% mean_conf=0.376
- NOTE: data/synthetic/ contains dry-run mock data. Run `python -m environment.transaction_simulator`
  with ANTHROPIC_API_KEY set to generate real confidence scores before Phase 2 training.
- Ended: Phase 1 COMPLETE

### Session 2 — 2026-03-25
- Started: Phase 1 complete, Phase 2 not started. agent/ had only __init__.py.
- Completed: Phase 2 — agent/policy_config.yaml, agent/train.py, agent/baseline.py
  created. All three PPO variants (A, B, C) trained with 100k timesteps each.
  Models saved to models/trained/. Training metadata JSON written.
- Ended: Phase 2 COMPLETE
