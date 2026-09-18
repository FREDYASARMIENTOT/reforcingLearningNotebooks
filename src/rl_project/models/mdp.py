"""
MDP model construction for MilanTaxiEnv.

Builds the explicit transition tensor P(s, a, s') and reward matrix R(s, a)
from a MilanTaxiEnv instance.

Reference: This is the Taller 1 equivalent of RL01's ``build_model()``.
"""
import numpy as np
from rl_project.envs.milan_taxi import (
    MilanTaxiEnv, N_S, N_A, encode_state, decode_state,
)


def build_model(env: MilanTaxiEnv) -> tuple[np.ndarray, np.ndarray]:
    """Construct the MDP transition tensor P and reward matrix R.

    Parameters
    ----------
    env : MilanTaxiEnv
        An instance of the environment (original or stochastic variant).

    Returns
    -------
    P : np.ndarray, shape (N_S, N_A, N_S)
        P[s, a, s'] = probability of transitioning to s' from (s, a).
        Satisfies: P >= 0  and  sum(P[s, a, :]) = 1.0 for all (s, a).
    R : np.ndarray, shape (N_S, N_A)
        R[s, a] = expected immediate reward from taking action a in state s.
    """
    nS = N_S
    nA = N_A
    P = np.zeros((nS, nA, nS), dtype=np.float64)
    R = np.zeros((nS, nA), dtype=np.float64)

    for s_idx in range(nS):
        row, col, pass_idx, dest_idx = decode_state(s_idx)

        for a_idx in range(nA):
            if env.variant == "original":
                # Deterministic: exactly one next state
                nr, nc, np_, nd, r, _ = env.get_deterministic_outcome(
                    row, col, pass_idx, dest_idx, a_idx
                )
                ns_idx = encode_state(nr, nc, np_, nd)
                P[s_idx, a_idx, ns_idx] = 1.0
                R[s_idx, a_idx] = r
            else:
                # Stochastic: multiple outcomes with probabilities
                outcomes = env.get_stochastic_outcomes(
                    row, col, pass_idx, dest_idx, a_idx
                )
                expected_reward = 0.0
                for prob, nr, nc, np_, nd, r, _ in outcomes:
                    ns_idx = encode_state(nr, nc, np_, nd)
                    P[s_idx, a_idx, ns_idx] += prob
                    expected_reward += prob * r
                R[s_idx, a_idx] = expected_reward

    return P, R


def check_probability_distribution(P: np.ndarray) -> bool:
    """Check that each row P[s, a, :] sums to 1.0 (within floating tolerance).

    Returns True if all rows are valid probability distributions.
    """
    row_sums = P.sum(axis=2)
    return bool(np.allclose(row_sums, 1.0))