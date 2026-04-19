"""
agent/train.py — PPO training script for the RL routing agent.

Usage:
    python agent/train.py --reward A
    python agent/train.py --reward B
    python agent/train.py --reward C

Each invocation trains a PPO policy on the routing environment with the
specified reward variant, logs to TensorBoard, saves checkpoints, and
writes training metadata to experiments/results/.

The trained model is saved to models/trained/ppo_variant_{A|B|C}.zip.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment.routing_env import RoutingEnv


# ---------------------------------------------------------------------------
# Paths (relative to repo root — run this script from repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "agent" / "policy_config.yaml"
TENSORBOARD_DIR = REPO_ROOT / "experiments" / "tensorboard"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
TRAINED_DIR = REPO_ROOT / "models" / "trained"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def make_env(reward_variant: str, seed: int) -> Monitor:
    """Create a monitored RoutingEnv for training."""
    log_dir = TENSORBOARD_DIR / f"variant_{reward_variant}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = RoutingEnv(reward_variant=reward_variant, seed=seed)
    return Monitor(env, filename=str(log_dir / "monitor"))


def make_eval_env(reward_variant: str, seed: int) -> Monitor:
    """Create a separate monitored RoutingEnv for evaluation callbacks."""
    eval_log_dir = TENSORBOARD_DIR / f"variant_{reward_variant}_eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    env = RoutingEnv(reward_variant=reward_variant, seed=seed + 1000)
    return Monitor(env, filename=str(eval_log_dir / "eval_monitor"))


def train(reward_variant: str) -> None:
    """Train a PPO policy for the given reward variant."""
    cfg = load_config()
    ppo_cfg = cfg["ppo"]
    train_cfg = cfg["training"]

    seed: int = train_cfg["seed"]
    total_timesteps: int = ppo_cfg["total_timesteps"]
    save_freq: int = train_cfg["save_freq"]
    eval_freq: int = train_cfg["eval_freq"]
    n_eval_episodes: int = train_cfg["n_eval_episodes"]

    # Ensure output directories exist
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training PPO — Reward Variant {reward_variant}")
    print(f"{'='*60}")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Seed            : {seed}")
    print(f"  Net arch        : {ppo_cfg['net_arch']}")
    print(f"  Learning rate   : {ppo_cfg['learning_rate']}")
    print(f"  Checkpoint dir  : {CHECKPOINT_DIR}")
    print(f"  TensorBoard dir : {TENSORBOARD_DIR}")
    print()

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = make_env(reward_variant, seed)
    eval_env = make_eval_env(reward_variant, seed)

    n_transactions = train_env.unwrapped.n_transactions  # type: ignore[attr-defined]
    print(f"  Training dataset: {n_transactions} transactions per episode")
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix=f"ppo_variant_{reward_variant}",
        verbose=1,
    )

    best_model_path = str(TRAINED_DIR / f"best_ppo_variant_{reward_variant}")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=str(RESULTS_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    policy_kwargs = {"net_arch": list(ppo_cfg["net_arch"])}

    model = PPO(
        policy=ppo_cfg["policy"],
        env=train_env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        tensorboard_log=str(TENSORBOARD_DIR),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=f"ppo_variant_{reward_variant}",
        progress_bar=True,
    )

    training_time = time.time() - start_time

    # ── Save final model ──────────────────────────────────────────────────────
    final_model_path = TRAINED_DIR / f"ppo_variant_{reward_variant}.zip"
    model.save(str(final_model_path))
    print(f"\nFinal model saved: {final_model_path}")

    # ── Evaluate final policy to get summary metrics ───────────────────────────
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    # evaluate_policy returns float when return_episode_rewards is False (default).
    mean_reward_f: float = float(mean_reward)  # type: ignore[arg-type]
    std_reward_f: float = float(std_reward)  # type: ignore[arg-type]

    # ── Training metadata ─────────────────────────────────────────────────────
    meta = {
        "reward_variant": reward_variant,
        "total_timesteps": total_timesteps,
        "final_mean_reward": round(mean_reward_f, 4),
        "final_std_reward": round(std_reward_f, 4),
        "training_time_seconds": round(training_time, 1),
        "n_transactions_in_dataset": n_transactions,
        "config_snapshot": {
            "ppo": ppo_cfg,
            "training": train_cfg,
        },
    }

    meta_path = RESULTS_DIR / f"training_meta_{reward_variant}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved: {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete — Variant {reward_variant}")
    print(f"{'='*60}")
    print(f"  Final mean reward : {mean_reward_f:.4f} ± {std_reward_f:.4f}")
    print(f"  Training time     : {training_time:.1f}s")
    print(f"  Model path        : {final_model_path}")
    print(f"  Metadata path     : {meta_path}")
    print(f"{'='*60}\n")

    train_env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO routing policy for the AI Accountant RL extension."
    )
    parser.add_argument(
        "--reward",
        choices=["A", "B", "C"],
        required=True,
        help="Reward variant to train with (A=binary, B=workload-weighted, C=conservative)",
    )
    args = parser.parse_args()
    train(args.reward)


if __name__ == "__main__":
    main()
