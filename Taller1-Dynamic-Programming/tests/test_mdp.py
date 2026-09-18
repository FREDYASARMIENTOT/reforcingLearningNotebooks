"""
Tests for the MDP model (build_model).

Verifies:
- P.shape, R.shape
- P >= 0
- sum(P[s,a,:]) == 1 for all (s,a)
- Original: deterministic (each row has exactly one 1.0)
- Stochastic: some rows have multiple positive entries
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from rl_project.envs.milan_taxi import MilanTaxiEnv, N_S, N_A
from rl_project.models.mdp import build_model, check_probability_distribution


def test_P_shape():
    env = MilanTaxiEnv(variant="original")
    P, R = build_model(env)
    assert P.shape == (N_S, N_A, N_S), f"P.shape={P.shape} != ({N_S},{N_A},{N_S})"


def test_R_shape():
    env = MilanTaxiEnv(variant="original")
    P, R = build_model(env)
    assert R.shape == (N_S, N_A), f"R.shape={R.shape} != ({N_S},{N_A})"


def test_P_nonnegative():
    env = MilanTaxiEnv(variant="original")
    P, R = build_model(env)
    assert (P >= 0).all(), "P has negative values"


def test_P_rows_sum_to_one():
    env = MilanTaxiEnv(variant="original")
    P, R = build_model(env)
    assert check_probability_distribution(P), "P rows do not sum to 1"


def test_original_is_deterministic():
    """Original: each (s,a) has exactly one s' with P[s,a,s'] = 1.0"""
    env = MilanTaxiEnv(variant="original")
    P, R = build_model(env)
    for s in range(N_S):
        for a in range(N_A):
            ones = (P[s, a] == 1.0).sum()
            assert ones == 1, f"P[s={s},a={a}] has {ones} ones (expected 1)"


def test_original_rewards_appropriate():
    """Verify some known reward values"""
    env = MilanTaxiEnv(variant="original")
    P, R = build_model(env)
    # Default per-step reward should be -1 or -10 (failed pickup/dropoff) or 20 (success)
    unique_rewards = set(np.unique(R))
    expected = {-1, -10, 20}
    for e in expected:
        assert e in unique_rewards, f"Expected reward {e} not found in R"


def test_stochastic_has_multiple_successors():
    """Stochastic variant: movement actions should have >1 successor states"""
    env = MilanTaxiEnv(variant="stochastic", slip_probability=0.1)
    P, R = build_model(env)
    multi = 0
    total = 0
    for s in range(50):  # Check first 50 states (sampling)
        for a in range(4):  # movement actions only
            total += 1
            if (P[s, a] > 0).sum() > 1:
                multi += 1
    assert multi > 0, f"No stochastic transitions found among {total} (s,a) pairs"


def test_stochastic_rows_still_sum_to_one():
    env = MilanTaxiEnv(variant="stochastic", slip_probability=0.1)
    P, R = build_model(env)
    assert check_probability_distribution(P), "Stochastic P rows do not sum to 1"


def test_stochastic_PICKUP_DROPOFF_deterministic():
    """PICKUP and DROPOFF should remain deterministic even in stochastic variant"""
    env = MilanTaxiEnv(variant="stochastic", slip_probability=0.1)
    P, R = build_model(env)
    for s in range(N_S):
        for a in [4, 5]:  # PICKUP, DROPOFF
            assert (P[s, a] == 1.0).sum() == 1, \
                f"PICKUP/DROPOFF must be deterministic at s={s}, a={a}"


def test_stochastic_slip_probabilities():
    """Stochastic movement actions: check probabilities are correct"""
    env = MilanTaxiEnv(variant="stochastic", slip_probability=0.2)
    P, R = build_model(env)
    slip = 0.2
    intended = 1.0 - 2 * slip
    # Check a specific state where movement isn't blocked by walls
    for s in range(N_S):
        for a in range(4):
            probs = P[s, a, P[s, a] > 0]
            if len(probs) == 1:
                # Wall-blocked: stays in place (prob=1.0)
                assert np.isclose(probs[0], 1.0)
            elif len(probs) == 2:
                # Some outcomes may merge if slip goes into same cell as intended
                assert np.isclose(probs.sum(), 1.0)
            elif len(probs) == 3:
                # Three distinct outcomes
                assert any(np.isclose(p, intended) for p in probs), \
                    f"No probability matches intended={intended}: {probs}"
                # The two slip probabilities
                slip_count = sum(np.isclose(p, slip) for p in probs)
                assert slip_count == 2, \
                    f"Expected 2 slip probabilities, found {slip_count}: {probs}"