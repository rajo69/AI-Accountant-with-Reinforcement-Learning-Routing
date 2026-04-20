# CLAUDE.md

Guidance for Claude Code sessions working on this repository.

## What this repo is

An RL routing research extension of the [AI Accountant](https://github.com/rajo69/agentic-ai-accounting)
parent project. Replaces hand-tuned confidence thresholds in the parent's
`CategoriserAgent` with a learned PPO policy. See `README.md` for the research
story, headline findings, and reproduction instructions.

## Where things live

- Code: `agent/`, `api/`, `environment/`, `experiments/`, `integration/`
- Data: `data/seeds/` (parent-project seeds), `data/synthetic/` (generated training set)
- Trained models: `models/trained/`
- Research artefacts: `paper/`, `experiments/results/`
- Status: `PROGRESS.md` (phase log), `RESEARCH_NOTES.md` (design decisions)
- Integration with parent: `integration/INTEGRATION_GUIDE.md`

## Standing rules

- Every numerical claim in README, paper/, and the data card must trace to a
  reproducible script in the repo. When numbers change, update both.
- Framing must distinguish "supported by a run on committed artefacts" from
  "hypothesised / candidate explanation". See `feedback_defensibility` in the
  user's memory for the full list of checks.
- CI is scoped to `agent api environment integration` (see `pyproject.toml`).
  Do not widen scope without a reason.
- Never push, commit, or open a PR without explicit per-session approval from
  the user.
