"""
Dynamic Programming algorithms for the MilanTaxi MDP.

Implements the core DP algorithms from Sutton & Barto Ch. 4:
- policy_matrices, solve_policy_direct, iterative_policy_evaluation
- q_from_v, greedy_policy, policy_iteration, value_iteration

Reference: These are the Taller 1 equivalents of the RL01 algorithms.
The mathematical structure is identical; only the MDP (MilanTaxi vs CliffWalking)
differs.
"""
import numpy as np


def policy_matrices(P, R, pi):
    """Average the dynamics over a policy.

    Returns (P_pi, R_pi) where:
        P_pi[s,t] = sum_a pi[s,a] * P[s,a,t]
        R_pi[s]   = sum_a pi[s,a] * R[s,a]

    Parameters
    ----------
    P : ndarray, shape (nS, nA, nS)
    R : ndarray, shape (nS, nA)
    pi : ndarray, shape (nS, nA)

    Returns
    -------
    P_pi : ndarray, shape (nS, nS)
    R_pi : ndarray, shape (nS,)
    """
    P_pi = np.einsum("sa,sat->st", pi, P)
    R_pi = np.einsum("sa,sa->s", pi, R)
    return P_pi, R_pi
def solve_policy_direct(P, R, pi, gamma):
    """Exact V^pi by solving (I - gamma * P^pi) V = R^pi.
    Uses np.linalg.solve (not inv) for numerical stability.
    """
    P_pi, R_pi = policy_matrices(P, R, pi)
    nS = P.shape[0]
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    return V


def iterative_policy_evaluation(P, R, pi, gamma, tol=1e-10, max_iter=100_000):
    """Iterative policy evaluation: V_{k+1} = R^pi + gamma * P^pi @ V_k"""
    P_pi, R_pi = policy_matrices(P, R, pi)
    V = np.zeros(P.shape[0])
    deltas = []

    for k in range(max_iter):
        V_new = R_pi + gamma * P_pi @ V
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break

    return V, k + 1, deltas


def q_from_v(P, R, V, gamma):
    """Q[s,a] = R[s,a] + gamma * sum_{s'} P[s,a,s'] * V[s']"""
    return R + gamma * np.einsum("sat,t->sa", P, V)


def greedy_policy(Q):
    """Deterministic greedy policy from Q, as a one-hot matrix."""
    pi = np.zeros_like(Q)
    pi[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return pi


def policy_iteration(P, R, gamma, max_iter=1_000):
    """Policy Iteration: evaluate -> improve -> repeat until stable."""
    nS, nA = R.shape
    pi = np.ones((nS, nA)) / nA

    for k in range(max_iter):
        V = solve_policy_direct(P, R, pi, gamma)
        Q = q_from_v(P, R, V, gamma)
        pi_new = greedy_policy(Q)

        if np.array_equal(pi_new.argmax(axis=1), pi.argmax(axis=1)):
            return V, pi_new, k + 1

        pi = pi_new

    return V, pi, k + 1


def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    """Value Iteration: V_{k+1}(s) = max_a [R[s,a] + gamma * sum P * V_k]"""
    V = np.zeros(R.shape[0])
    deltas = []

    for k in range(max_iter):
        Q = q_from_v(P, R, V, gamma)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break

    Q = q_from_v(P, R, V, gamma)
    pi = greedy_policy(Q)

    return V, pi, k + 1, deltas