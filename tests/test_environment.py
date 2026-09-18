"""Tests for the MilanTaxiEnv implementation."""
import sys
sys.path.insert(0, "src")

import numpy as np
from rl_project.envs.milan_taxi import (
    MilanTaxiEnv, N_S, N_A, NUM_ROWS, NUM_COLS, NUM_PASS, NUM_DEST,
    HORIZON, encode_state, decode_state, check_wall,
    SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, LOCS, PASS_IN_TAXI,
)


def test_n_states():
    assert N_S == NUM_ROWS * NUM_COLS * NUM_PASS * NUM_DEST
    assert N_S == 500


def test_n_actions():
    assert N_A == 6


def test_encode_decode_roundtrip():
    for s_idx in range(N_S):
        row, col, pass_idx, dest_idx = decode_state(s_idx)
        assert 0 <= row < NUM_ROWS
        assert 0 <= col < NUM_COLS
        assert 0 <= pass_idx < NUM_PASS
        assert 0 <= dest_idx < NUM_DEST
        s2 = encode_state(row, col, pass_idx, dest_idx)
        assert s_idx == s2


def test_all_states_covered():
    seen = set()
    for s_idx in range(N_S):
        decoded = decode_state(s_idx)
        assert decoded not in seen
        seen.add(decoded)
    assert len(seen) == N_S


def test_reset_returns_tuple():
    env = MilanTaxiEnv()
    state, info = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 4


def test_step_returns_correct_types():
    env = MilanTaxiEnv()
    env.reset()
    for a in range(N_A):
        ns, r, term, trunc, info = env.step(a)
        assert isinstance(ns, tuple) and len(ns) == 4
        assert term is False
        assert "delivered_passengers" in info
        env.reset()


def test_movement_reward_is_negative_one():
    env = MilanTaxiEnv()
    env.reset(seed=0)
    _, r, _, _, _ = env.step(SOUTH)
    assert r == -1


def test_failed_pickup_reward_is_negative_ten():
    env = MilanTaxiEnv()
    for _ in range(20):
        s, _ = env.reset(seed=42)
        row, col, pass_idx, _ = s
        if (row, col) != LOCS[pass_idx]:
            _, r, _, _, _ = env.step(PICKUP)
            assert r == -10
            break


def test_never_terminates():
    env = MilanTaxiEnv()
    env.reset(seed=0)
    for _ in range(10):
        a = np.random.randint(0, N_A)
        _, _, term, _, _ = env.step(a)
        assert term is False


def test_truncated_at_horizon():
    env = MilanTaxiEnv()
    env.reset(seed=0)
    for t in range(HORIZON):
        _, _, _, trunc, _ = env.step(SOUTH)
        if t < HORIZON - 1:
            assert not trunc
        else:
            assert trunc


def test_original_is_deterministic():
    env = MilanTaxiEnv(variant="original")
    env.reset(seed=42)
    outcomes = [env.step(a)[0] for a in range(N_A)]
    for _ in range(3):
        env.reset(seed=42)
        for a in range(N_A):
            s2 = env.step(a)[0]
            assert outcomes[a] == s2


def test_stochastic_differs_from_original():
    env_o = MilanTaxiEnv(variant="original")
    env_s = MilanTaxiEnv(variant="stochastic", slip_probability=0.5)
    differing = False
    for a in [SOUTH, NORTH, EAST, WEST]:
        env_o.reset(seed=0)
        ns_o, _, _, _, _ = env_o.step(a)
        for _ in range(10):
            env_s.reset(seed=0)
            ns_s, _, _, _, _ = env_s.step(a)
            if ns_s != ns_o:
                differing = True
                break
        if differing:
            break
    assert differing


def test_wall_blocking():
    assert check_wall(0, 1, 0, 2)
    assert check_wall(0, 2, 0, 1)
    assert not check_wall(0, 0, 0, 1)
    assert not check_wall(0, 1, 0, 0)


def test_wall_prevents_movement():
    env = MilanTaxiEnv()
    env.state = (0, 1, 0, 0)
    ns, _, _, _, _ = env.step(EAST)
    assert ns[0] == 0 and ns[1] == 1