"""
Experiment orchestration for Taller 1 — Dynamic Programming.

Runs the 4 main experiment configurations:
    A. Original + Policy Iteration
    B. Original + Value Iteration
    C. Stochastic + Policy Iteration
    D. Stochastic + Value Iteration
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

from rl_project.envs.milan_taxi import MilanTaxiEnv, encode_state
from rl_project.models.mdp import build_model, check_probability_distribution
from rl_project.agents.dynamic_programming import (
    policy_iteration, value_iteration,
)
from rl_project.evaluation import evaluate_policy
@dataclass
class ExperimentResult:
    environment: str = ""
    variant: str = ""
    algorithm: str = ""
    gamma: float = 0.99
    slip_probability: float = 0.0
    n_iterations: int = 0
    v_start: float = 0.0
    runtime_s: float = 0.0
    return_mean: float = 0.0
    return_std: float = 0.0
    return_ci: float = 0.0
    success_rate: float = 0.0
    mean_ep_length: float = 0.0
    n_deliveries: float = 0.0
    notes: str = ""


def run_single_experiment(variant, algorithm, gamma=0.99, slip_probability=0.1,
                          tol=1e-10, max_iter=500, eval_runs=1000, eval_seed=42):
    """Run a single (variant, algorithm) experiment."""
    result = ExperimentResult(
        environment="MilanTaxiEnv", variant=variant, algorithm=algorithm,
        gamma=gamma,
        slip_probability=slip_probability if variant == "stochastic" else 0.0,
    )

    # 1. Create environment
    env = MilanTaxiEnv(variant=variant, slip_probability=slip_probability)

    # 2. Build P and R
    P, R = build_model(env)
    assert check_probability_distribution(P), "P rows must sum to 1"

    # 3. Run DP algorithm
    t0 = time.perf_counter()
    if algorithm == "policy_iteration":
        V, pi, n_iters = policy_iteration(P, R, gamma, max_iter=max_iter)
        result.n_iterations = n_iters
    elif algorithm == "value_iteration":
        V, pi, n_sweeps, _ = value_iteration(P, R, gamma, tol=tol, max_iter=max_iter)
        result.n_iterations = n_sweeps
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    result.runtime_s = time.perf_counter() - t0

    # 4. V(start) at a canonical start state
    s_start = encode_state(0, 0, 0, 1)
    result.v_start = float(V[s_start])

    # 5. Evaluate via rollouts
    eval_result = evaluate_policy(env, pi, n_runs=eval_runs, gamma=gamma,
                                  seed=eval_seed, verbose=False)
    result.return_mean = eval_result["mean_return"]
    result.return_std = eval_result["std_return"]
    result.return_ci = eval_result["ci_return"]
    result.success_rate = eval_result["success_rate"]
    result.mean_ep_length = eval_result["mean_ep_length"]
    result.n_deliveries = eval_result["n_deliveries"]
    return result


def run_all_experiments(gamma=0.99, tol=1e-10, max_iter=500, eval_runs=1000,
                        eval_seed=42, runs_dir="runs"):
    """Run all 4 experiment configurations and save results."""
    from pathlib import Path
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("original", "policy_iteration"),
        ("original", "value_iteration"),
        ("stochastic", "policy_iteration"),
        ("stochastic", "value_iteration"),
    ]

    results = []
    for variant, algorithm in configs:
        print(f"\n{'='*60}")
        print(f"Running: {variant} + {algorithm}")
        print(f"{'='*60}")

        res = run_single_experiment(variant=variant, algorithm=algorithm,
                                    gamma=gamma, tol=tol, max_iter=max_iter,
                                    eval_runs=eval_runs, eval_seed=eval_seed)
        results.append(res)

        print(f"  iterations/sweeps: {res.n_iterations}")
        print(f"  V(start):          {res.v_start:.4f}")
        print(f"  runtime:           {res.runtime_s:.3f}s")
        print(f"  eval return:       {res.return_mean:.4f} +/- {res.return_ci:.4f}")
        print(f"  success rate:      {res.success_rate:.3f}")
        print(f"  avg ep length:     {res.mean_ep_length:.1f}")

    # Save results
    results_dicts = [asdict(r) for r in results]
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = runs_dir / f"experiment_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_dicts, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    summary_path = runs_dir / f"experiment_summary_{timestamp}.csv"
    with open(summary_path, "w") as f:
        f.write("variant,algorithm,iterations,V_start,runtime_s,return_mean,return_ci,success_rate,mean_ep_len\n")
        for r in results:
            f.write(f"{r.variant},{r.algorithm},{r.n_iterations},{r.v_start:.6f},"
                    f"{r.runtime_s:.4f},{r.return_mean:.4f},{r.return_ci:.4f},"
                    f"{r.success_rate:.4f},{r.mean_ep_length:.2f}\n")
    print(f"Summary saved to:  {summary_path}")
    return results


if __name__ == "__main__":
    run_all_experiments()