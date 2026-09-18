"""
Tests for Policy Iteration.

Verifies:
- V finito (no NaN, no Inf)
- pi válida (suma 1 por fila)
- converge en número razonable de iteraciones
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from rl_project.envs.milan_taxi import MilanTaxiEnv, N_S, N_A
from rl_project.models.mdp import build_model
from rl_project.agents.dynamic_programming import (
    policy_iteration, policy_matrices, solve_policy_direct,
    iterative_policy_evaluation, q_from_v, greedy_policy,
)

GAMMA = 0.99


def _get_P_R(variant="original"):
    env = MilanTaxiEnv(variant=variant)
    return build_model(env)


def test_policy_matrices_output_shape():
    P, R = _get_P_R()
    pi = np.ones((N_S, N_A)) / N_A
    P_pi, R_pi = policy_matrices(P, R, pi)
    assert P_pi.shape == (N_S, N_S), f"P_pi.shape={P_pi.shape}"
    assert R_pi.shape == (N_S,), f"R_pi.shape={R_pi.shape}"


def test_policy_matrices_P_rows_sum_to_one():
    P, R = _get_P_R()
    pi = np.ones((N_S, N_A)) / N_A
    P_pi, R_pi = policy_matrices(P, R, pi)
    row_sums = P_pi.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "P_pi rows must sum to 1"


def test_solve_policy_direct_finite():
    P, R = _get_P_R()
    pi = np.ones((N_S, N_A)) / N_A
    V = solve_policy_direct(P, R, pi, GAMMA)
    assert np.all(np.isfinite(V)), "V has non-finite values"


def test_iterative_policy_evaluation_converges():
    P, R = _get_P_R()
    pi = np.ones((N_S, N_A)) / N_A
    V, n_sweeps, deltas = iterative_policy_evaluation(P, R, pi, GAMMA, tol=1e-10)
    assert np.all(np.isfinite(V)), "Iterative V has non-finite values"
    assert n_sweeps > 0, "n_sweeps should be positive"
    assert n_sweeps < 100_000, "Too many sweeps"
    assert deltas[-1] < 1e-10, f"Final delta={deltas[-1]} not below tol"


def test_iterative_matches_direct():
    """Iterative policy evaluation should match direct solve within tolerance"""
    P, R = _get_P_R()
    pi = np.ones((N_S, N_A)) / N_A
    V_direct = solve_policy_direct(P, R, pi, GAMMA)
    V_iter, _, _ = iterative_policy_evaluation(P, R, pi, GAMMA, tol=1e-10)
    err = np.max(np.abs(V_iter - V_direct))
    assert err < 1e-8, f"max |V_iter - V_direct| = {err} (> 1e-8)"


def test_q_from_v_shape():
    P, R = _get_P_R()
    pi = np.ones((N_S, N_A)) / N_A
    V = solve_policy_direct(P, R, pi, GAMMA)
    Q = q_from_v(P, R, V, GAMMA)
    assert Q.shape == (N_S, N_A), f"Q.shape={Q.shape}"


def test_greedy_policy_is_one_hot():
    Q = np.random.randn(N_S, N_A)
    pi = greedy_policy(Q)
    assert pi.shape == (N_S, N_A)
    assert np.allclose(pi.sum(axis=1), 1.0), "Greedy policy rows must sum to 1"
    assert ((pi == 0) | (pi == 1)).all(), "Greedy policy must be one-hot"


def test_policy_iteration_converges_original():
    P, R = _get_P_R("original")
    V, pi, n_iters = policy_iteration(P, R, GAMMA, max_iter=1000)
    assert np.all(np.isfinite(V)), "PI V has non-finite values"
    assert np.allclose(pi.sum(axis=1), 1.0), "PI policy must be valid"
    assert n_iters > 0, "n_iters should be positive"
    assert n_iters < 100, f"PI took {n_iters} iterations (expected < 100)"


def test_policy_iteration_converges_stochastic():
    P, R = _get_P_R("stochastic")
    V, pi, n_iters = policy_iteration(P, R, GAMMA, max_iter=1000)
    assert np.all(np.isfinite(V)), "PI (stochastic) V has non-finite values"
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert n_iters > 0
    assert n_iters < 100, f"PI stochastic took {n_iters} iterations (expected < 100)"


def test_policy_iteration_improves_over_random():
    """Optimal V should be better than random V at start state"""
    P, R = _get_P_R("original")
    pi_random = np.ones((N_S, N_A)) / N_A
    V_random = solve_policy_direct(P, R, pi_random, GAMMA)
    V_opt, _, _ = policy_iteration(P, R, GAMMA)
    from rl_project.envs.milan_taxi import encode_state
    s_start = encode_state(0, 0, 0, 1)
    assert V_opt[s_start] >= V_random[s_start], \
        f"Optimal V={V_opt[s_start]:.2f} < Random V={V_random[s_start]:.2f}"