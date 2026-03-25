# Deployment Guide

## Railway (API Backend)

### Prerequisites
- Railway account at [railway.app](https://railway.app)
- Railway CLI: `npm install -g @railway/cli`
- Trained model file committed at `models/trained/ppo_variant_C.zip` (already committed)

### Steps

```bash
# 1. Login
railway login

# 2. Initialise a new Railway project
railway init    # select "Empty Project"

# 3. Link to local repo
railway link [project-id]

# 4. Set environment variables in Railway dashboard
#    (Settings > Variables — copy from .env.example)
#    Required:
#      ROUTING_MODEL_VARIANT=C
#      ANTHROPIC_API_KEY=sk-...   (only needed if running the simulator)
#
#    Optional overrides:
#      ROUTING_AUTO_THRESHOLD=0.85
#      ROUTING_REVIEW_THRESHOLD=0.50

# 5. Deploy
railway up

# 6. Get your public URL
railway domain

# 7. Verify
curl https://[your-url]/health
```

Expected health response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "policy": "ppo_variant_C",
  "using_fallback": false
}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ROUTING_MODEL_VARIANT` | No | `C` | Which PPO variant to load (A, B, or C) |
| `ROUTING_AUTO_THRESHOLD` | No | `0.85` | Fallback auto-approval threshold |
| `ROUTING_REVIEW_THRESHOLD` | No | `0.50` | Fallback review threshold |
| `ANTHROPIC_API_KEY` | No | — | Only needed to run the transaction simulator |
| `PORT` | Auto | `8000` | Set automatically by Railway |

### Redeployment

Connect the GitHub repo in Railway under **Settings > Source**. Every push to
`main` triggers an automatic redeploy. The trained model files are committed
to the repo so they are included in every deployment.

### Persistent Model Storage

The committed model files (~140KB each) are included in the Docker image.
For larger models (>50MB) or frequent retraining, use one of:

- **Option A (current):** Commit model files to repo — simple, fine for small models.
- **Option B:** Railway Volume — mount at `/app/models/trained/` via Settings > Volumes.
- **Option C:** S3/R2 — store model on object storage, download on startup. Set
  `MODEL_S3_URL` and add download logic to `api/main.py`'s lifespan handler.

After replacing a model file, call `POST /policy/reload` to hot-swap the policy
without restarting the server.

---

## Docker (Local)

```bash
# Build image
docker build -t rl-routing-api .

# Run
docker run -p 8000:8000 \
  -e ROUTING_MODEL_VARIANT=C \
  rl-routing-api

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"confidence_score": 0.88, "transaction_features": {"amount": 450.0}}'
```

---

## Local Development (no Docker)

```bash
# 1. Clone and install
git clone https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing.git
cd AI-Accountant-with-Reinforcement-Learning-Routing
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# 3. Run the API
uvicorn api.main:app --reload --port 8000

# 4. Open interactive docs
# Visit http://localhost:8000/docs
```

### Quickstart (generate data, train, evaluate, serve)

```bash
# Generate synthetic training data (requires ANTHROPIC_API_KEY in .env)
python -m environment.transaction_simulator

# Train all three PPO variants
python -m agent.train --reward A
python -m agent.train --reward B
python -m agent.train --reward C

# Evaluate on held-out set
python -m agent.evaluate

# Serve the routing API
uvicorn api.main:app --reload
```

---

## Manually Retrain via GitHub Actions

1. Go to **Actions** tab in the GitHub repo
2. Select **Retrain Policy** workflow
3. Click **Run workflow**
4. Choose reward variant (A, B, or C)
5. Download the trained model artifact when the run completes
6. Commit the new `.zip` to `models/trained/` and push

---

## API Reference

### `POST /route`

Route a transaction using the loaded PPO policy.

```bash
curl -X POST https://[your-url]/route \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_score": 0.72,
    "transaction_features": {
      "amount": 1250.00,
      "difficulty_tier": 1,
      "category_entropy": 0.0
    }
  }'
```

Response:
```json
{
  "status": "auto_categorised",
  "action_index": 0,
  "confidence_score": 0.72,
  "policy_used": "ppo_variant_C",
  "model_variant": "C",
  "latency_ms": 1.2,
  "audit_data": { ... }
}
```

### `GET /policy/info`

Returns the current policy variant, training config, and Phase 3 evaluation metrics.

### `POST /policy/reload`

Hot-reloads the model from disk. Use after deploying a retrained model without restart.

### `GET /health`

Standard health check. Returns 200 if the service is ready.
