# Taller 1 — Dynamic Programming on MilanTaxi

**Course:** Reinforcement Learning  
**Date:** September 2026  
**Project Root:** `D:\ReforcingLearning\Taller1-Dynamic-Programming`

## Objective

Implement and analyze **Policy Iteration (PI)** and **Value Iteration (VI)** on the **MilanTaxi** environment. The environment is a 5×5 grid with internal walls (4 landmarks), 500 states, and 6 actions. Two variants are studied:

- **Original:** Deterministic transitions.
- **Stochastic:** Movement actions have slip probability (0.1 each for left/right slip).

## MDP Formulation

| Component | Description |
|---|---|
| **State space S** | (row, col, passenger_idx, destination_idx) → 500 states |
| **Action space A** | SOUTH(0), NORTH(1), EAST(2), WEST(3), PICKUP(4), DROPOFF(5) |
| **Transition P(s'\|s,a)** | Deterministic (original) or stochastic slip (stochastic variant) |
| **Reward R(s,a)** | −1 per step, −10 failed PICKUP/DROPOFF, +20 successful DROPOFF |
| **Discount γ** | 0.99 (unless specified) |

## Installation

```bash
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
.venv\Scripts\python.exe -m pip install -e .
```

## Running Experiments

```bash
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
$env:PYTHONPATH="src"
.venv\Scripts\python.exe src\rl_project\experiment.py
```

## Running Tests

```bash
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
$env:PYTHONPATH="src"
.venv\Scripts\pytest tests\test_environment.py -q
.venv\Scripts\pytest tests\test_mdp.py -q
.venv\Scripts\pytest tests\test_policy_iteration.py -q
.venv\Scripts\pytest tests\test_value_iteration.py -q
```

Expected: **42/42 PASS** (VI tests may take ~60s combined with γ=0.99, tol=1e-10).

## Results Summary

| Environment | Algorithm | Iter/Sweeps | V(start) | Time (s) |
|---|---|---|---|---|
| Original | Policy Iteration | 11 | 1818.39 | ~5.0 |
| Original | Value Iteration | 2591 | 1818.39 | ~6.1 |
| Stochastic | Policy Iteration | 7 | 1771.30 | ~2.9 |
| Stochastic | Value Iteration | 2592 | 1771.30 | ~5.3 |

PI and VI produce the same optimal value function (max diff < 1e-8). The stochastic variant reduces V* by ~2.6%.

## Empirical Evaluation (1000 episodes)

| Policy | Mean Return | 95% CI | Success Rate |
|---|---|---|---|
| Original + PI | ~28.1 | ±0.6 | 100% |
| Original + VI | ~28.1 | ±0.6 | 100% |
| Stochastic + PI | ~5.5 | ±0.7 | 100% |
| Stochastic + VI | ~5.5 | ±0.7 | 100% |

## Key Findings

1. **PI is iteration-efficient** (7-11 iterations) while **VI is per-sweep efficient** (~2600 cheap sweeps).
2. **Stochasticity reduces value** — V(start) drops 2.6%; empirical return drops ~80% (amplified by 100-step horizon).
3. **Both algorithms converge to identical V*** (max diff < 1e-8).
4. **Lower γ** leads to faster convergence but more myopic policies (V(start) negative at γ=0.5).

## Project Structure

```
Taller1-Dynamic-Programming/
├── .venv/                    # Virtual environment (Python 3.12)
├── configs/                  # YAML experiment configs
├── notebooks/
│   └── milan_taxi.ipynb      # Main notebook (all experiments + analysis)
├── runs/                     # Experiment results (JSON, CSV)
├── src/
│   └── rl_project/
│       ├── agents/
│       │   └── dynamic_programming.py  # PI and VI implementations
│       ├── envs/
│       │   └── milan_taxi.py           # MilanTaxi environment
│       ├── models/
│       │   └── mdp.py                  # MDP builder (P, R tensors)
│       ├── evaluation.py               # Policy evaluation via rollouts
│       ├── experiment.py               # Experiment orchestration
│       └── policies.py                 # Policy utilities
├── tests/
│   ├── test_environment.py     # 14 tests
│   ├── test_mdp.py             # 10 tests
│   ├── test_policy_iteration.py # 10 tests
│   └── test_value_iteration.py # 8 tests
├── README.md
├── TALLER1_FINAL_REPORT.md
├── requirements.txt
├── pyproject.toml
└── runs/README.md
```

## Limitations

- Horizon truncation (100 steps) → gap between theoretical and empirical values.
- Only movement actions stochastic; PICKUP/DROPOFF remain deterministic.
- Tabular DP does not scale beyond this problem size.
- Policy ties (multiple actions with same Q-value) cause ~6% disagreement.

## Reproducibility

- All random seeds fixed (seed=42 for evaluation, seed=0 for configs).
- Results saved in `runs/` as JSON + CSV with timestamps.
- Independent `.venv` used (Python 3.12.13).
- Source code in `src/rl_project/`, tests in `tests/`.
- Visualizations generated in the notebook.

## Note

The project `D:\ReforcingLearning\rl-basics` is a reference repository and was **not modified** by this Taller.