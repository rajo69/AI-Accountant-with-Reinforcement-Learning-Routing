# Contributing

This project is primarily a research artefact. The most valuable contributions
are new reward variants, alternative RL algorithms, and improved data generation.

---

## Adding a New Reward Variant

1. Open `environment/reward_functions.py`.

2. Add a new function following the existing pattern:

   ```python
   def reward_d(action: int, is_correct: bool, **kwargs: float) -> float:
       """
       Variant D: [Your description].

       Design rationale:
       ─────────────────
       [Document EVERY numerical value. This is the intellectual contribution.]
       """
       ...
   ```

3. Register it in the `REWARD_FUNCTIONS` dict at the bottom of the file:

   ```python
   REWARD_FUNCTIONS: dict[str, callable] = {
       "A": reward_a,
       "B": reward_b,
       "C": reward_c,
       "D": reward_d,   # add here
   }
   ```

4. Add the variant to `agent/policy_config.yaml`:

   ```yaml
   training:
     reward_variants: [A, B, C, D]
   ```

5. If the new variant requires extra state (like Variant B's `accountant_load`),
   update `RoutingEnv.__init__` to add the dimension to the observation space
   when `reward_variant == "D"`, following the Variant B pattern.

6. Write at least one test in `environment/tests/test_routing_env.py`.

7. Train: `python -m agent.train --reward D`

8. Evaluate: `python -m agent.evaluate` (update the script to include Variant D
   in the policy list).

---

## Swapping the RL Algorithm

The environment is a standard Gymnasium `Env`, so any Stable Baselines3
algorithm can be used as a drop-in.

1. Open `agent/train.py`.

2. Replace the `PPO` import and instantiation:

   ```python
   # Before
   from stable_baselines3 import PPO
   model = PPO(policy=..., env=train_env, ...)

   # After (e.g. DQN)
   from stable_baselines3 import DQN
   model = DQN(policy=..., env=train_env, ...)
   ```

3. DQN and A2C use different hyperparameters. Add a new section to
   `agent/policy_config.yaml`:

   ```yaml
   dqn:
     learning_rate: 1.0e-3
     buffer_size: 50000
     learning_starts: 1000
     ...
   ```

4. Update `agent/evaluate.py` to load the new model type:

   ```python
   from stable_baselines3 import DQN
   model = DQN.load(str(model_path))
   ```

5. Update the trained model filename convention in `train.py` and `router.py`
   (e.g. `dqn_variant_A.zip`).

---

## Running Tests

```bash
pytest -v                              # all tests
pytest environment/tests/ -v          # environment tests only
pytest integration/tests/ -v          # integration tests only
```

---

## Code Style

```bash
ruff check .        # linting
black .             # formatting
mypy api/ integration/ agent/ environment/ --ignore-missing-imports
```

All three must pass before committing. The CI pipeline enforces this.
