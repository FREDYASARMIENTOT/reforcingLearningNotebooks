"""
Policy evaluation via Monte Carlo rollouts.

Given a policy (represented as an (nS, nA) matrix) and an environment,
estimate the empirical return by running episodes.

Key distinction:
    - V^pi(s): theoretical expected return under the MDP dynamics.
    - Empirical return: sample-average of actual episode returns in the
      (possibly stochastic) environment.

The two should *agree in expectation* but differ due to:
    - Finite-sample noise.
    - Stochastic transitions (the slip_probability variant).
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from rl_project.envs.milan_taxi import (
    MilanTaxiEnv, N_S, N_A, HORIZON,
    encode_state,
)


def evaluate_policy(
    env: MilanTaxiEnv,
    policy_matrix: np.ndarray,
    n_runs: int = 1000,
    gamma: float = 0.99,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate a policy matrix by running episodes in the environment.

    The policy matrix is used to select actions greedily (argmax).

    Parameters
    ----------
    env : MilanTaxiEnv
        Environment instance (original or stochastic).
    policy_matrix : ndarray, shape (N_S, N_A)
        One-hot policy matrix.
    n_runs : int, default=1000
        Number of episodes to run.
    gamma : float, default=0.99
        Discount factor for computing discounted return.
    seed : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=True
        Show progress bar.

    Returns
    -------
    results : dict with keys:
        - 'mean_return': float, average discounted return
        - 'std_return': float, standard deviation of returns
        - 'ci_return': float, 95% confidence interval half-width
        - 'mean_ep_length': float, average episode length
        - 'success_rate': float, fraction of episodes with at least one delivery
        - 'n_deliveries': float, average number of deliveries per episode
        - 'returns': list of all returns (for histogram)
        - 'ep_lengths': list of all episode lengths
    """
    rng = np.random.default_rng(seed)

    returns = []
    ep_lengths = []
    deliveries = []

    iterator = range(n_runs)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating policy")

    for _ in iterator:
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_return = 0.0
        step = 0

        while True:
            # Encode state to get index, then look up action from policy
            s_idx = encode_state(*state)
            action = int(policy_matrix[s_idx].argmax())

            next_state, reward, terminated, truncated, info = env.step(action)
            ep_return += (gamma ** step) * reward
            step += 1
            state = next_state

            if terminated or truncated:
                break

        returns.append(ep_return)
        ep_lengths.append(step)
        deliveries.append(info.get("delivered_passengers", 0))

    returns = np.array(returns)
    ep_lengths = np.array(ep_lengths)
    deliveries = np.array(deliveries)

    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=1))
    ci_return = 1.96 * std_return / np.sqrt(n_runs) if n_runs > 1 else 0.0

    results = {
        "mean_return": mean_return,
        "std_return": std_return,
        "ci_return": ci_return,
        "mean_ep_length": float(ep_lengths.mean()),
        "success_rate": float((deliveries > 0).mean()),
        "n_deliveries": float(deliveries.mean()),
        "returns": returns.tolist(),
        "ep_lengths": ep_lengths.tolist(),
    }
    return results