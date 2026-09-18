"""
Policy representations for the Taller 1 Dynamic Programming project.

A policy is represented as a matrix ``pi`` of shape (nS, nA) where:
    pi[s, a] = P(a | s)   (probability of taking action a in state s)
    sum_a pi[s, :] = 1.0  for all s

A *deterministic* policy has pi[s, :] as a one-hot vector.
"""
import numpy as np


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy from action-value function Q.

    pi[s, a] = 1 if a == argmax_a' Q[s, a'] else 0

    Parameters
    ----------
    Q : ndarray, shape (nS, nA)

    Returns
    -------
    pi : ndarray, shape (nS, nA)
        One-hot deterministic greedy policy.
    """
    pi = np.zeros_like(Q)
    pi[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return pi


def random_policy(nS: int, nA: int) -> np.ndarray:
    """Uniform random policy: all actions equally likely in all states.

    Parameters
    ----------
    nS : int
        Number of states.
    nA : int
        Number of actions.

    Returns
    -------
    pi : ndarray, shape (nS, nA)
    """
    return np.ones((nS, nA)) / nA


def policy_from_argmax(argmax: np.ndarray, nA: int) -> np.ndarray:
    """Build a one-hot policy matrix from argmax action indices.

    Parameters
    ----------
    argmax : ndarray, shape (nS,)
        argmax[s] = index of the chosen action in state s.
    nA : int
        Number of actions.

    Returns
    -------
    pi : ndarray, shape (nS, nA)
    """
    nS = len(argmax)
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), argmax] = 1.0
    return pi


def policy_argmax(pi: np.ndarray) -> np.ndarray:
    """Extract argmax actions from a one-hot policy matrix.

    Parameters
    ----------
    pi : ndarray, shape (nS, nA)

    Returns
    -------
    argmax : ndarray, shape (nS,)
    """
    return pi.argmax(axis=1)


def policy_agreement(pi1: np.ndarray, pi2: np.ndarray) -> float:
    """Proportion of states where two deterministic policies agree.

    Parameters
    ----------
    pi1, pi2 : ndarray, shape (nS, nA)
        One-hot deterministic policy matrices.

    Returns
    -------
    agreement : float
        Fraction of states with the same argmax action.
    """
    return (policy_argmax(pi1) == policy_argmax(pi2)).mean()