# TALLER 1 — FINAL REPORT
## Dynamic Programming on MilanTaxi

**Course:** Reinforcement Learning
**Date:** September 2026
**Project:** `D:\ReforcingLearning\Taller1-Dynamic-Programming`

---

## 1. Project Status

**COMPLETE** — All phases of the Taller are implemented and validated.

## 2. Final Structure

```
Taller1-Dynamic-Programming/
├── .venv/                          # Python 3.12.13 virtual env
├── configs/
│   ├── policy_iteration_milan.yaml
│   └── value_iteration_milan.yaml
├── notebooks/
│   └── milan_taxi.ipynb            # Main deliverable
├── runs/
│   ├── README.md
│   ├── result_*.json               # Per-experiment results
│   ├── gamma_experiment.json
│   ├── experiment_results_*.json   # Full results
│   ├── experiment_summary_*.csv    # Summary table
│   └── vi_convergence.png          # Convergence plot
├── src/rl_project/
│   ├── __init__.py
│   ├── evaluation.py               # Monte Carlo evaluation
│   ├── experiment.py               # Experiment orchestration
│   ├── policies.py                 # Policy helpers
│   ├── agents/dynamic_programming.py  # PI & VI
│   ├── envs/milan_taxi.py           # Milantaxi env
│   └── models/mdp.py               # P,R builder
├── tests/
│   ├── test_environment.py         # 14 tests
│   ├── test_mdp.py                 # 10 tests
│   ├── test_policy_iteration.py    # 10 tests
│   └── test_value_iteration.py     # 8 tests
├── README.md
├── TALLER1_FINAL_REPORT.md
├── requirements.txt
├── pyproject.toml
└── runs/README.md
```

## 3. Environment — Original

- Deterministic transitions.
- 5×5 grid with internal walls, 4 landmarks.
- 100-step horizon (always truncates).
- Rewards: −1 per step, −10 failed service, +20 successful DROPOFF.

## 4. Environment — Stochastic Variant

- Movement slip: intended=0.8, left=0.1, right=0.1.
- PICKUP/DROPOFF remain deterministic.
- Walls block movement (taxi stays in place).

## 5. MDP Formulation

| Symbol | Value |
|---|---|
| S | (row, col, pass_idx, dest_idx), \|S\|=500 |
| A | 6 actions: S,N,E,W,PICKUP,DROPOFF |
| P(s'\|s,a) | (500,6,500) tensor |
| R(s,a) | (500,6) matrix |
| γ | 0.99 |

Encoding: idx = row·100 + col·20 + pass_idx·4 + dest_idx

## 6. P and R Validation

- P.shape=(500,6,500), R.shape=(500,6)
- P≥0, sum(P[s,a,:])=1 for all (s,a)
- Original: one successor per (s,a).
- Stochastic: up to 3 successors [0.8, 0.1, 0.1].
- Rewards: −1, −10, +20.

## 7. Policy Iteration

Algorithm: evaluate (solve (I−γP^π)V = R^π) → improve (greedy) → repeat.
## 10. Experiments

Four main combinations with seed 42. Additional gamma experiment with VI and γ∈{0.5, 0.9, 0.99}.

## 11. Results

### PI vs VI Comparison
- V* agreement: max|V_PI − V_VI| < 1e-8 (both variants).
- Policy argmax agreement: ~94% (original), 100% (stochastic). Differences due to Q-value ties.

### Gamma effect

| Variant | γ | Sweeps | V(start) | Time (s) |
|---|---|---|---|---|
| Original | 0.50 | 39 | −1.92 | 0.09 |
| Original | 0.90 | 248 | +71.36 | 0.51 |
| Original | 0.99 | 2591 | +1818.39 | 5.44 |
| Stochastic | 0.50 | 39 | −1.97 | 0.09 |
| Stochastic | 0.90 | 248 | +54.51 | 0.57 |
| Stochastic | 0.99 | 2592 | +1771.30 | 5.41 |

## 12. Comparison PI vs VI

| Criterion | PI | VI |
|---|---|---|
| Outer iterations | 7−11 | N/A |
| Sweeps | N/A | ~2590 |
| Per-iteration cost | High (500×500 linear solve) | Low (O(n²·m)) |
| Convergence | Policy stable | Delta < tol |
| Time (original) | ~5s | ~6s |
| Time (stochastic) | ~3s | ~5s |

## 13. Five Guiding Questions

Full answers in notebook Section 12. Summary:

**Q1:** MDP defined by S (500), A (6), P, R (−1,−10,+20), γ=0.99.

**Q2:** P(s'|s,a) changes from degenerate to distribution; V* decreases ~2.6%.

**Q3:** PI = two-loop (Newton-like). VI = single-loop (gradient-like).

**Q4:** Delta threshold, policy stability, cross-validation, empirical rollouts.

**Q5:** V* drops 2.6%; empirical return drops ~80% (horizon amplified); convergence similar.

## 14. Limitations

1. Horizon truncation (100 steps) → theoretical/empirical gap.
2. Non-terminal environment.
3. Only movements stochastic; PICKUP/DROPOFF deterministic.
4. VI slow with γ=0.99, tol=1e-10 (~2600 sweeps).
5. Policy ties (Q-value ties → arbitrary argmax).
6. Tabular DP doesn't scale.

## 15. Reproducibility

Fixed seeds (42 eval, 0 experiment), .venv (Python 3.12.13), results saved as JSON+CSV, YAML configs.

## 16. Tests

| Suite | Tests | Status | Time |
|---|---|---|---|
| test_environment.py | 14 | PASS | 0.4s |
| test_mdp.py | 10 | PASS | 1.1s |
| test_policy_iteration.py | 10 | PASS | 26.6s |
| test_value_iteration.py | 8 | PASS | ~60s* |
| **Total** | **42** | **PASS** | **~88s** |

*VI suite times out at 30s tool limit but passes individually.

## 17. Notebook

- **File:** `notebooks/milan_taxi.ipynb`
- **14 sections** covering MDP, environment, P/R, PI, VI, convergence, comparison, evaluation, gamma, guiding questions, limitations, conclusions.
- Requires `.venv` kernel. Execution time ~45s total.

## 18. Deliverable Files

1. `notebooks/milan_taxi.ipynb`
2. `src/rl_project/` (source code)
3. `tests/` (42 tests)
4. `README.md`
5. `TALLER1_FINAL_REPORT.md`
6. `runs/` (JSON, CSV, PNG results)
7. `configs/` (YAML)
8. `requirements.txt`
9. `pyproject.toml`

## 19. rl-basics Protection

**CONFIRMED:** `D:\ReforcingLearning\rl-basics` was NOT modified by this Taller. All work is within `D:\ReforcingLearning\Taller1-Dynamic-Programming`.

| Variant | Iterations | V(start) | Time (s) |
|---|---|---|---|
| Original | 11 | 1818.39 | 4.99 |
| Stochastic | 7 | 1771.30 | 2.88 |

## 8. Value Iteration

Algorithm: V_{k+1}(s) = max_a[R(s,a) + γ Σ P(s'|s,a)V_k(s')], then extract greedy policy.

| Variant | Sweeps | V(start) | Time (s) | Final Delta |
|---|---|---|---|---|
| Original | 2591 | 1818.39 | 6.06 | 9.98e-11 |
| Stochastic | 2592 | 1771.30 | 5.32 | 9.91e-11 |

## 9. Evaluation (1000 episodes)

| Policy | Mean Return | 95% CI | Success Rate | Deliveries |
|---|---|---|---|---|
| Original + PI | 28.13 | ±0.62 | 1.000 | 7.09 |
| Original + VI | 28.13 | ±0.62 | 1.000 | 7.09 |
| Stochastic + PI | 5.45 | ±0.65 | 1.000 | 5.39 |
| Stochastic + VI | 5.45 | ±0.65 | 1.000 | 5.39 |