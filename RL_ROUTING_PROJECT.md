# RL_ROUTING_PROJECT.md
# Master Prompt — AI Accountant: Learned Routing via Reinforcement Learning

---

## HOW TO USE THIS FILE

At the start of **every** Claude Code session, say exactly this:

> "Read RL_ROUTING_PROJECT.md, PROGRESS.md, CLAUDE.md, MEMORY.md, and any
> ARCHITECTURE.md present in this directory. Then resume the project from the
> current phase recorded in PROGRESS.md."

Claude Code will read all files, reconstruct full context, and continue from
exactly where the previous session ended. Never start a session without this
instruction — context from prior sessions is not automatically retained.

---

## CONTEXT AND MOTIVATION

This repository is a research extension of the AI Accountant project. Before
doing anything else in a new session, read CLAUDE.md, MEMORY.md, and
ARCHITECTURE.md thoroughly. Those files contain the complete architecture,
design decisions, technology stack, database schema, agent topology, and
deployment configuration of the parent project. Do not ask me to re-explain
anything that is documented there.

### What this project does and why

The AI Accountant's CategoriserAgent uses three hand-tuned confidence thresholds:
- Above 0.85 → auto-approve
- 0.50–0.85 → surface for accountant review
- Below 0.50 → flag for manual decision

These thresholds were chosen by the developer as reasonable defaults. They are
arbitrary. A natural research question is: can a lightweight reinforcement
learning agent learn better routing thresholds by observing accountant feedback
over time?

This project builds and evaluates exactly that. The RL agent observes the
CategoriserAgent's confidence scores and transaction features, chooses a routing
action, and receives reward based on whether the accountant accepts or corrects
the decision. We compare the learned policy against the hand-tuned baseline using
accuracy, accountant workload reduction, and error rate as metrics.

### Why this matters beyond this one application

Confidence-gated autonomy — deciding when an AI should act vs escalate to a
human — is an unsolved problem across agentic AI systems broadly. The findings
here are relevant to clinical decision support, autonomous compliance checking,
and any multi-agent pipeline where human oversight must be calibrated. This
framing must be reflected in the README and all writeups.

---

## PERSISTENT PROGRESS TRACKING

Claude Code must maintain PROGRESS.md throughout the project. This file is the
single source of truth for session continuity.

### PROGRESS.md format

Claude Code must create PROGRESS.md after Phase 0 completes and update it at
the end of every phase and at every natural stopping point within a phase.
The file must always contain:

```markdown
# PROGRESS.md — RL Routing Project Session Log

## Current Phase
[Phase number and name]

## Phase Status
[NOT STARTED / IN PROGRESS / COMPLETE]

## Last Completed Step
[Precise description of the last thing that was finished]

## Next Step
[Exact first action to take when resuming]

## Completed Phases
- [x] Phase 0 — Repository initialisation
- [ ] Phase 1 — Environment and synthetic data
- [ ] Phase 2 — RL agent training
- [ ] Phase 3 — Evaluation and comparison
- [ ] Phase 4 — Integration and API
- [ ] Phase 5 — CI/CD and deployment
- [ ] Phase 6 — README and documentation

## Open Issues
[Any blockers, decisions needed, or things to revisit]

## Session Log
### Session [N] — [date]
- Started: [what state the project was in]
- Completed: [what was done]
- Ended: [what state the project is in now]
```

Update PROGRESS.md before ending any session, even if mid-phase. A future
session must be able to resume from the Next Step entry without any additional
explanation from the user.

---

## PHASE DEFINITIONS

Work through phases sequentially. Do not start a phase until the previous one
is marked COMPLETE in PROGRESS.md. At the end of each phase, explicitly tell
the user what was completed, update PROGRESS.md, and wait for confirmation
before proceeding to the next phase unless the user says to continue.

---

### PHASE 0 — Repository Initialisation
**Goal:** Clean up the duplicated folder and establish the new project identity.

**Steps:**

1. Read CLAUDE.md, MEMORY.md, and ARCHITECTURE.md in full. Confirm you have
   understood the parent project architecture by summarising in 5 bullet points
   what the CategoriserAgent does, what its confidence routing logic is, what
   the database schema looks like, and how the human-in-the-loop correction
   mechanism works. Ask the user to confirm before proceeding.

2. Remove the .git directory if present (the user will have done this manually,
   but verify it is gone).

3. Create PROGRESS.md using the format above, marking Phase 0 as IN PROGRESS.

4. Update CLAUDE.md to add a section at the top:
   ```
   ## Project Identity
   This is the RL Routing Research Extension of the AI Accountant.
   Parent project context is documented below.
   New goal: Replace hand-tuned confidence thresholds in CategoriserAgent
   with a learned PPO routing policy. See RL_ROUTING_PROJECT.md.
   ```

5. Create the full directory structure exactly as specified in the REPOSITORY
   STRUCTURE section of this file. Create __init__.py files and .gitkeep files
   as needed. Do not create any implementation files yet — structure only.

6. Create a minimal .gitignore appropriate for Python/FastAPI/Next.js plus:
   ```
   data/raw/
   experiments/results/*.json
   models/trained/
   .env
   .env.local
   ```

7. Create a minimal .env.example documenting all environment variables that
   will be needed (derive these from the parent project's existing env vars
   plus new ones for the RL components: MODEL_PATH, TRAINING_EPISODES,
   EVAL_EPISODES, REWARD_VARIANT).

8. **At end of Phase 0:** Ask the user for the new GitHub repository URL.
   Then run:
   ```bash
   git init
   git add .
   git commit -m "chore: initialise rl-routing research extension"
   git branch -M main
   git remote add origin [URL provided by user]
   git push -u origin main
   ```
   Confirm push succeeded. Update PROGRESS.md to mark Phase 0 COMPLETE.

---

### PHASE 1 — Environment and Synthetic Data
**Goal:** Build the Gym-compatible RL environment and synthetic transaction
dataset. This is the most intellectually important phase — the reward function
design must be carefully reasoned and documented.

**Steps:**

1. Read the CategoriserAgent implementation from the parent codebase carefully.
   Identify exactly: what confidence score is produced, what features accompany
   each transaction at routing time, and how accountant corrections are
   currently stored. These become the state space and reward signal.

2. Create `environment/routing_env.py` — a Gymnasium-compatible environment:

   **State space:** A numpy array containing:
   - confidence_score (float, 0–1) — from CategoriserAgent output
   - amount_normalised (float, 0–1) — transaction amount normalised by log scale
   - difficulty_tier (int, 0/1/2) — easy/medium/hard from existing eval framework
   - category_entropy (float, 0–1) — entropy across top-3 category probabilities
     if available from the model output, else 0

   **Action space:** Discrete(3)
   - 0 = AUTO_APPROVE
   - 1 = SURFACE_FOR_REVIEW
   - 2 = REJECT_FOR_MANUAL

   **Reward function:** Implement THREE variants in `environment/reward_functions.py`
   and document the reasoning for each:

   Variant A — Binary:
   - AUTO_APPROVE correct: +1.0
   - AUTO_APPROVE incorrect: -2.0 (asymmetric: false confidence is worse)
   - SURFACE_FOR_REVIEW correct: +0.3 (correct escalation but costs accountant time)
   - SURFACE_FOR_REVIEW unnecessary: -0.3 (workload cost with no benefit)
   - REJECT correct: +0.5
   - REJECT incorrect: -1.0

   Variant B — Workload-weighted:
   Same as A but SURFACE_FOR_REVIEW penalties scale with a simulated
   accountant_load parameter (0–1) injected into the state — higher load means
   unnecessary escalation is penalised more heavily.

   Variant C — Conservative:
   AUTO_APPROVE incorrect: -5.0 (models a high-stakes compliance environment
   where false auto-approvals are catastrophic)

   Document in reward_functions.py why these asymmetries exist. This
   documentation IS the research contribution — the reward design choices
   reflect real-world priorities and must be explainable.

3. Create `environment/transaction_simulator.py`:

   - Load the existing 50 labelled transactions from the parent project's
     evaluation framework. These are the seed dataset.
   - Use the Claude API (already configured in the parent project) to generate
     synthetic variations: given a real transaction, generate 10 variants that
     change amount, date, merchant name phrasing, and description while
     preserving the correct category label.
   - Target: 800–1000 synthetic transactions across all categories, balanced
     by difficulty tier.
   - Run each synthetic transaction through the existing CategoriserAgent to
     get real confidence scores — do not fabricate confidence scores. The
     simulator calls the live agent.
   - Store results in data/synthetic/ as JSONL with schema:
     {transaction_id, features, confidence_score, true_category,
      difficulty_tier, is_synthetic: true}
   - Reserve 20% as held-out evaluation set in data/evaluation/held_out_set.json
     (stratified by difficulty tier). Never train on this set.
   - Write a data card in data/synthetic/README.md documenting: generation
     methodology, synthetic vs real split, category distribution, difficulty
     distribution, and known limitations.

4. Write unit tests in environment/tests/:
   - test_env_reset_returns_valid_state
   - test_env_step_returns_correct_shape
   - test_reward_variant_a_correct_approval
   - test_reward_variant_a_incorrect_approval_penalised_more_than_unnecessary_escalation
   - test_simulator_produces_balanced_categories
   - test_held_out_set_not_in_training_set

5. Commit: `feat: add routing environment and synthetic data pipeline`

6. Update PROGRESS.md. Tell user Phase 1 is complete and summarise: number of
   synthetic transactions generated, category distribution, reward variants
   implemented. Wait for confirmation before Phase 2.

---

### PHASE 2 — RL Agent Training
**Goal:** Train a PPO policy for each reward variant, log results, and save
trained models.

**Steps:**

1. Install Stable Baselines3 and add to requirements:
   ```
   stable-baselines3>=2.3.0
   gymnasium>=0.29.0
   tensorboard>=2.16.0
   ```

2. Create `agent/policy_config.yaml`:
   ```yaml
   ppo:
     learning_rate: 3.0e-4
     n_steps: 2048
     batch_size: 64
     n_epochs: 10
     gamma: 0.99
     gae_lambda: 0.95
     clip_range: 0.2
     total_timesteps: 100000
     policy: MlpPolicy
     net_arch: [64, 64]

   training:
     reward_variants: [A, B, C]
     n_eval_episodes: 100
     eval_freq: 5000
     save_freq: 10000
     seed: 42
   ```

3. Create `agent/train.py`:
   - Accepts reward_variant as CLI argument: `python train.py --reward A`
   - Wraps the routing environment in a SB3 Monitor wrapper for logging
   - Trains PPO with config from policy_config.yaml
   - Logs to TensorBoard in experiments/tensorboard/
   - Saves checkpoints to models/checkpoints/ every save_freq steps
   - Saves final model to models/trained/ppo_variant_{A|B|C}.zip
   - Saves training metadata to experiments/results/training_meta_{A|B|C}.json:
     {reward_variant, total_timesteps, final_mean_reward, training_time_seconds,
      n_transactions_seen, config_snapshot}
   - Prints a clear summary on completion

4. Create `agent/baseline.py`:
   - Implements the original hand-tuned threshold policy as a class with the
     same interface as the SB3 policy (predict method taking observation,
     returning action)
   - Thresholds loaded from environment variable or config so they match
     exactly what the parent project uses
   - This is the comparison baseline — it must faithfully reproduce the
     existing behaviour

5. Train all three variants. If training fails or produces degenerate results
   (mean reward not improving after 50k steps), document this honestly in
   PROGRESS.md and proceed — a null or negative result is a valid finding.

6. Commit trained models and results:
   `feat: train ppo routing policies for all three reward variants`

   Note: Add models/trained/*.zip to .gitignore if files are large (>50MB).
   Instead commit training metadata JSON only and document how to retrain.

7. Update PROGRESS.md. Tell user Phase 2 is complete with training summary
   for all three variants. Wait for confirmation.

---

### PHASE 3 — Evaluation and Comparison
**Goal:** Rigorously compare learned policies against the hand-tuned baseline
on the held-out set. Produce honest, citable results.

**Steps:**

1. Create `agent/evaluate.py`:
   - Loads held-out set from data/evaluation/held_out_set.json
   - Evaluates baseline policy and all three PPO variants on identical data
   - For each policy, computes:
     - Overall routing accuracy (was the routing decision appropriate)
     - Auto-approval precision (of auto-approved, % that were correct)
     - Auto-approval rate (workload reduction proxy)
     - Unnecessary escalation rate (surfaced for review but would have been
       correct to auto-approve)
     - Error rate on auto-approvals by difficulty tier
     - Confusion matrix over routing actions
   - Saves to experiments/results/evaluation_results.json
   - Prints a formatted comparison table to stdout

2. Create `experiments/notebooks/analysis.ipynb`:
   - Loads evaluation_results.json
   - Produces four plots saved as PNG to experiments/results/figures/:
     a. Accuracy vs workload tradeoff curve for all policies
     b. Per-difficulty-tier error rates (grouped bar chart)
     c. Routing action distribution for each policy (stacked bar)
     d. Learning curve from TensorBoard logs (reward over timesteps)
   - Use matplotlib with a clean, publication-appropriate style

3. Create `experiments/results/comparison_report.md` — auto-generated from
   evaluation results. Structure:
   ```
   ## Routing Policy Comparison Report
   ### Summary
   [2–3 sentence plain-English summary of main finding]
   ### Methodology
   [Environment, state space, action space, reward variants, training setup]
   ### Results Table
   [Markdown table: policy × metric]
   ### Key Findings
   [Bullet points — honest, including if baseline wins]
   ### Limitations
   [Synthetic data, small evaluation set, single environment, no real
    accountant feedback loop — all must be documented]
   ### Implications
   [What this means for confidence-gated autonomy in production systems]
   ```

4. Run evaluation and generate report. If the PPO policy does NOT outperform
   the baseline, this is fine and must be reported honestly. Document in
   the limitations section why this might be (insufficient training data,
   reward misspecification, etc.). This honest negative result is more
   valuable for PhD applications than a fabricated positive one.

5. Commit: `feat: evaluation framework and comparison report`

6. Update PROGRESS.md. Give user a plain-English summary of results.
   Wait for confirmation before Phase 4.

---

### PHASE 4 — Integration and API
**Goal:** Make the trained policy a drop-in replacement for the threshold
logic, and expose it via a FastAPI endpoint consistent with the parent
project's API architecture.

**Steps:**

1. Read the parent project's API architecture from ARCHITECTURE.md and
   CLAUDE.md carefully. The new API must be consistent with existing
   patterns — same auth approach, same response structure conventions,
   same error handling style, same logging approach.

2. Create `integration/router.py`:
   - A class `LearnedRouter` with identical interface to the existing
     threshold router
   - Loads the best-performing trained model (determined by Phase 3 results)
   - `route(confidence_score, transaction_features) -> RoutingDecision`
   - Falls back to hand-tuned thresholds if model file not found (graceful
     degradation — never break production)
   - Logs routing decision, confidence score, and which policy was used
     to the existing audit log table (reuse the parent project's audit
     infrastructure exactly)

3. Create `integration/tests/test_router.py`:
   - test_learned_router_returns_valid_action
   - test_fallback_to_baseline_when_model_missing
   - test_router_logs_to_audit_table
   - test_router_interface_matches_baseline_interface

4. Create `api/main.py` — a FastAPI app exposing:
   - POST /route — accepts transaction features, returns routing decision
     with explanation (which policy, confidence, action, reasoning)
   - GET /policy/info — returns current policy metadata (variant, training
     date, evaluation metrics)
   - GET /health — standard health check
   - POST /policy/reload — hot-reload the policy from disk without restart

5. Create `api/schemas.py` with Pydantic models for all request/response
   types. Use the same Pydantic version and validation style as the parent
   project.

6. Write a brief integration guide in integration/INTEGRATION_GUIDE.md
   explaining exactly which file and which class in the parent project
   needs to be updated to use LearnedRouter instead of threshold logic,
   with a code diff example.

7. Commit: `feat: integration layer and routing API`

8. Update PROGRESS.md. Wait for confirmation before Phase 5.

---

### PHASE 5 — CI/CD and Deployment
**Goal:** Production-grade GitHub Actions pipeline and deployment
instructions for Railway (API) and Vercel (if applicable).

**Steps:**

1. Create `.github/workflows/ci.yml`:
   ```yaml
   name: CI
   on:
     push:
       branches: [main, develop]
     pull_request:
       branches: [main]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.12'
             cache: 'pip'
         - run: pip install -r requirements.txt
         - run: pip install pytest pytest-asyncio pytest-cov
         - run: pytest --cov=. --cov-report=xml -v
         - uses: codecov/codecov-action@v4

     lint:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.12'
         - run: pip install ruff black
         - run: ruff check .
         - run: black --check .

     type-check:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.12'
         - run: pip install mypy
         - run: mypy api/ integration/ agent/ environment/
   ```

2. Create `.github/workflows/train.yml` — manually triggered workflow
   to retrain the policy on GitHub Actions:
   ```yaml
   name: Retrain Policy
   on:
     workflow_dispatch:
       inputs:
         reward_variant:
           description: 'Reward variant (A, B, or C)'
           required: true
           default: 'A'
   jobs:
     train:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.12'
         - run: pip install -r requirements.txt
         - run: python agent/train.py --reward ${{ github.event.inputs.reward_variant }}
         - uses: actions/upload-artifact@v4
           with:
             name: trained-model-${{ github.event.inputs.reward_variant }}
             path: models/trained/
   ```

3. Create `railway.toml` in root:
   ```toml
   [build]
   builder = "nixpacks"

   [deploy]
   startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
   healthcheckPath = "/health"
   healthcheckTimeout = 30
   restartPolicyType = "on_failure"
   restartPolicyMaxRetries = 3
   ```

4. Create `Dockerfile` for the API:
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

5. Create `DEPLOYMENT.md` with step-by-step instructions:

   ```markdown
   # Deployment Guide

   ## Railway (API Backend)

   ### Prerequisites
   - Railway account at railway.app
   - Railway CLI: npm install -g @railway/cli
   - Trained model file in models/trained/

   ### Steps
   1. railway login
   2. railway init (select "Empty Project")
   3. railway link [project-id]
   4. Set environment variables in Railway dashboard:
      - Copy all variables from .env.example
      - Set MODEL_PATH=models/trained/ppo_variant_A.zip
      - Set any API keys from parent project
   5. railway up
   6. railway domain (to get your public URL)
   7. Test: curl https://[your-url]/health

   ### Environment Variables Required
   [List every variable from .env.example with description]

   ### Redeployment
   Push to main branch triggers automatic redeployment via Railway's
   GitHub integration. Connect repo in Railway dashboard under
   Settings > Source.

   ### Persistent Model Storage
   Railway's filesystem is ephemeral. For persistent model storage:
   Option A: Commit model file to repo (fine if <50MB)
   Option B: Store model on Railway Volume (Settings > Volumes)
   Option C: Store on S3/R2 and download on startup (production approach)
   The api/main.py startup event handles Option C if MODEL_S3_URL is set.

   ## Vercel (if frontend is added)

   ### Note
   This project is currently API-only. If a frontend analysis dashboard
   is added in future, deploy via:
   1. vercel login
   2. vercel --prod from the frontend/ directory
   3. Set NEXT_PUBLIC_API_URL to your Railway URL

   ## Local Development
   1. cp .env.example .env
   2. Fill in .env values
   3. pip install -r requirements.txt
   4. uvicorn api.main:app --reload
   5. Visit http://localhost:8000/docs for Swagger UI
   ```

6. Commit: `feat: cicd workflows and deployment configuration`

7. Update PROGRESS.md. Wait for confirmation before Phase 6.

---

### PHASE 6 — README and Final Documentation
**Goal:** A thorough, research-grade README that matches the quality of the
parent project and is appropriate for PhD application portfolios.

**Steps:**

1. Create README.md with the following structure. Write it as a genuine
   research document — not a list of features. Use clear prose. Every
   section should add information not present in another section.

   ```markdown
   # AI Accountant: Learned Routing via Reinforcement Learning

   > A research extension investigating whether a learned routing policy
   > can outperform hand-tuned confidence thresholds in a production
   > multi-agent accounting pipeline.

   ## Research Question
   [One paragraph stating the precise question, why it matters, and
   what approach is taken]

   ## Background
   [2–3 paragraphs on the parent project — what the AI Accountant does,
   what the CategoriserAgent's role is, what the existing threshold logic
   looks like and why it is arbitrary. Link to parent repo.]

   ## Methodology

   ### Environment Design
   [State space, action space, reward function variants — with reasoning
   for every design decision. This is the most important section.]

   ### Synthetic Data Generation
   [How the 50-transaction seed dataset was expanded, what the Claude API
   generated, data card summary, known limitations]

   ### Training
   [PPO configuration, training setup, computational requirements]

   ### Evaluation
   [Held-out set methodology, metrics chosen and why]

   ## Results
   [Honest results table. If baseline wins, say so clearly. If PPO wins
   on some metrics but not others, explain the tradeoff.]

   ## Key Findings
   [3–5 bullet points — the actual takeaways a reader should remember]

   ## Limitations and Future Work
   [Synthetic data limitations, small evaluation set, single environment,
   no real accountant feedback loop, reward misspecification risk.
   Future work: real deployment feedback, multi-environment generalisation,
   online learning]

   ## Implications for Agentic AI Systems
   [1–2 paragraphs connecting findings to the broader problem of
   confidence-gated autonomy in production multi-agent systems]

   ## Repository Structure
   [Annotated tree of all directories and key files]

   ## Quickstart
   [Clone, install, generate data, train, evaluate — 5 commands]

   ## Running the API
   [Local and deployed instructions]

   ## Deployment
   [Link to DEPLOYMENT.md]

   ## Citing This Work
   [BibTeX entry — author Rajarshi Nandi, year 2026, GitHub URL]

   ## Licence
   MIT
   ```

2. Create CONTRIBUTING.md — brief guide for anyone who wants to add a new
   reward variant or swap the RL algorithm.

3. Create RESEARCH_NOTES.md — a running log of design decisions, things
   tried that did not work, and observations during training. This is
   valuable for PhD interviews where you may be asked "walk me through
   your design choices." Populate it with the actual decisions made
   during this project.

4. Final commit: `docs: research readme and documentation`

5. Tag the release: `git tag v1.0.0 -m "Complete RL routing research project"`
   Push tags: `git push origin --tags`

6. Update PROGRESS.md marking ALL phases COMPLETE.

7. Tell the user the project is complete. Provide:
   - Summary of what was built
   - Key result in one sentence
   - The two sentences they should use to describe this project in a
     PhD application supporting statement
   - Link to the GitHub repo

---

## REPOSITORY STRUCTURE (target state)

```
ai-accountant-rl-routing/
├── RL_ROUTING_PROJECT.md          ← This file
├── PROGRESS.md                    ← Session continuity (auto-maintained)
├── CLAUDE.md                      ← Updated from parent project
├── MEMORY.md                      ← Updated from parent project
├── README.md                      ← Research README
├── DEPLOYMENT.md                  ← Railway and Vercel instructions
├── CONTRIBUTING.md
├── RESEARCH_NOTES.md
├── railway.toml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
│
├── environment/
│   ├── __init__.py
│   ├── routing_env.py
│   ├── transaction_simulator.py
│   └── reward_functions.py
│
├── agent/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baseline.py
│   └── policy_config.yaml
│
├── data/
│   ├── synthetic/
│   │   ├── README.md              ← Data card
│   │   └── transactions.jsonl
│   ├── evaluation/
│   │   └── held_out_set.json
│   └── raw/                       ← Gitignored
│
├── experiments/
│   ├── results/
│   │   ├── baseline_results.json
│   │   ├── evaluation_results.json
│   │   ├── comparison_report.md
│   │   └── figures/
│   ├── tensorboard/               ← Gitignored
│   └── notebooks/
│       └── analysis.ipynb
│
├── models/
│   ├── trained/                   ← Gitignored if large
│   └── checkpoints/               ← Gitignored
│
├── integration/
│   ├── __init__.py
│   ├── router.py
│   ├── INTEGRATION_GUIDE.md
│   └── tests/
│       └── test_router.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── health.py
│
└── .github/
    └── workflows/
        ├── ci.yml
        └── train.yml
```

---

## GENERAL RULES FOR CLAUDE CODE

These rules apply throughout all phases and all sessions.

**Architecture consistency:** Before writing any new code, check CLAUDE.md
and ARCHITECTURE.md to understand existing patterns. All new code must be
consistent with the parent project's conventions — same async patterns,
same error handling style, same logging approach, same Pydantic version.

**No fabrication:** Never invent training results, evaluation metrics, or
data statistics. If something does not work, document it honestly.

**Reward function documentation is non-negotiable:** The reward_functions.py
file must contain extensive inline comments explaining the reasoning behind
every numerical value. This is the intellectual core of the project.

**Test before commit:** Run tests before every commit. Never commit broken
code. If tests fail and you cannot fix them, document the failure in
PROGRESS.md and flag it to the user.

**PROGRESS.md is sacred:** Update it at the end of every phase and every
natural stopping point. A future session must be able to resume with zero
additional context from the user beyond "resume from PROGRESS.md."

**Commit messages follow conventional commits:**
- feat: new functionality
- fix: bug fixes
- docs: documentation only
- chore: setup, config, tooling
- test: test additions
- refactor: code restructuring without behaviour change

**Never skip phases:** Do not start Phase N+1 without confirming with the
user that Phase N is complete and they are ready to continue.

**Ask before major decisions:** If you encounter an architectural decision
not covered by this file or the existing MD files, stop and ask the user
rather than assuming. Document the decision in RESEARCH_NOTES.md.

---

## STARTING INSTRUCTION

When you read this file for the first time in a new session:

1. Read PROGRESS.md to find the current phase and next step.
2. If PROGRESS.md does not exist, start Phase 0.
3. If PROGRESS.md exists, resume from the "Next Step" entry.
4. Confirm with the user what you are about to do before doing it.
5. Never summarise this file back to the user — just act on it.
