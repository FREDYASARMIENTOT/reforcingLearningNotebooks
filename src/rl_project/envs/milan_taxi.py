"""
MilanTaxiEnv --- The Milan Taxi environment for Taller 1 — Dynamic Programming.

Based on the same problem formulation as the original MilanTaxiEnv in
``rl-basics``, but implemented as an independent, self-contained environment.

State space:   (taxi_row, taxi_col, passenger_idx, destination_idx)
   taxi_row ∈ [0,4]; taxi_col ∈ [0,4]; passenger_idx ∈ [0,4]; dest_idx ∈ [0,3]
   Total: 5 × 5 × 5 × 4 = 500 states

Action space:  6 actions: SOUTH(0), NORTH(1), EAST(2), WEST(3), PICKUP(4), DROPOFF(5)

Reward structure:
   -1   per step
   -10  on failed PICKUP / DROPOFF
   +20  on successful DROPOFF

Variants:
   "original":     deterministic transitions
   "stochastic":   movement actions have slip probability
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ROWS = 5
NUM_COLS = 5
NUM_PASS = 5       # 4 locations + 1 (PASS_IN_TAXI = 4)
NUM_DEST = 4       # 4 landmarks
HORIZON = 100

STATE_SHAPE = (NUM_ROWS, NUM_COLS, NUM_PASS, NUM_DEST)
N_S = NUM_ROWS * NUM_COLS * NUM_PASS * NUM_DEST  # = 500
N_A = 6

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5
PASS_IN_TAXI = 4

LOCS: list[tuple[int, int]] = [(0, 0), (0, 4), (4, 0), (4, 3)]

INTERNAL_WALLS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((0, 1), (0, 2)), ((1, 1), (1, 2)),
    ((3, 0), (3, 1)), ((4, 0), (4, 1)),
    ((3, 2), (3, 3)), ((4, 2), (4, 3)),
]

MOVE_DELTAS = {SOUTH: (1, 0), NORTH: (-1, 0), EAST: (0, 1), WEST: (0, -1)}

_SLIP_MAP = {
    SOUTH: (EAST, WEST),
    NORTH: (WEST, EAST),
    EAST: (NORTH, SOUTH),
    WEST: (SOUTH, NORTH),
}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_wall(row: int, col: int, new_row: int, new_col: int) -> bool:
    """Return True if moving from (row, col) to (new_row, new_col) crosses a wall."""
    return ((row, col), (new_row, new_col)) in INTERNAL_WALLS or \
           ((new_row, new_col), (row, col)) in INTERNAL_WALLS


def encode_state(row: int, col: int, pass_idx: int, dest_idx: int) -> int:
    """Encode a 4D state tuple into a scalar state index in [0, N_S).

    idx = row * 100 + col * 20 + pass_idx * 4 + dest_idx
    """
    return (row * NUM_COLS * NUM_PASS * NUM_DEST
            + col * NUM_PASS * NUM_DEST
            + pass_idx * NUM_DEST
            + dest_idx)


def decode_state(idx: int) -> tuple[int, int, int, int]:
    """Decode a scalar state index back into the 4D state tuple."""
    dest_idx = idx % NUM_DEST
    pass_idx = (idx // NUM_DEST) % NUM_PASS
    col = (idx // (NUM_PASS * NUM_DEST)) % NUM_COLS
    row = idx // (NUM_COLS * NUM_PASS * NUM_DEST)
    return (row, col, pass_idx, dest_idx)


def _compute_new_position(row: int, col: int, action: int) -> tuple[int, int]:
    """Compute the new (row, col) after a movement action (boundaries only)."""
    dr, dc = MOVE_DELTAS[action]
    new_row = max(0, min(NUM_ROWS - 1, row + dr))
    new_col = max(0, min(NUM_COLS - 1, col + dc))
    return new_row, new_col
# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MilanTaxiEnv(gym.Env):
    """Milan Taxi environment for Dynamic Programming exercises.

    Parameters
    ----------
    variant : str, default="original"
        - "original":      deterministic transitions
        - "stochastic":    movement actions have a slip probability
    slip_probability : float, default=0.1
        Probability of slipping left/right when taking a movement action.
        Must satisfy: 0 <= slip_probability <= 0.5.
    """

    def __init__(self, variant: str = "original", slip_probability: float = 0.1):
        super().__init__()
        assert variant in ("original", "stochastic"), \
            f"variant must be 'original' or 'stochastic', got '{variant}'"
        assert 0.0 <= slip_probability <= 0.5, \
            f"slip_probability must be in [0, 0.5], got {slip_probability}"

        self.variant = variant
        self.slip_probability = slip_probability if variant == "stochastic" else 0.0
        self.state = None
        self.time_step = 0
        self._delivered_passengers = 0
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset the environment and return (state, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._delivered_passengers = 0
        self.time_step = 0
        taxi_row = int(self._rng.integers(0, NUM_ROWS))
        taxi_col = int(self._rng.integers(0, NUM_COLS))
        pass_idx, dest_idx = self._spawn_new_passenger()
        self.state = (taxi_row, taxi_col, int(pass_idx), int(dest_idx))
        return self.state, {}

    def step(self, action: int) -> tuple:
        """Take an action and return (next_state, reward, terminated, truncated, info)."""
        row, col, pass_idx, dest_idx = self.state

        if self.variant == "original":
            new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = \
                self._transition_deterministic(row, col, pass_idx, dest_idx, action)
        else:
            new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = \
                self._transition_stochastic(row, col, pass_idx, dest_idx, action)

        self.time_step += 1
        truncated = self.time_step >= HORIZON

        if dest_reached:
            self._delivered_passengers += 1
            if not truncated:
                new_pass_idx, new_dest_idx = self._spawn_new_passenger()

        self.state = (new_row, new_col, new_pass_idx, new_dest_idx)
        return self.state, reward, False, truncated, {
            "delivered_passengers": self._delivered_passengers
        }

    # ------------------------------------------------------------------
    # Internal transition logic (deterministic)
    # ------------------------------------------------------------------
    def _transition_deterministic(
        self, row: int, col: int, pass_idx: int, dest_idx: int, action: int
    ) -> tuple:
        """Deterministic transition: compute outcome for given (s, a).

        Returns (new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached).
        """
        new_row, new_col = row, col
        new_pass_idx = pass_idx
        dest_reached = False
        reward = -1

        if action in (SOUTH, NORTH, EAST, WEST):
            new_row, new_col = _compute_new_position(row, col, action)
            if check_wall(row, col, new_row, new_col):
                new_row, new_col = row, col
        elif action == PICKUP:
            if pass_idx != PASS_IN_TAXI and (row, col) == LOCS[pass_idx]:
                new_pass_idx = PASS_IN_TAXI
            else:
                reward = -10
        elif action == DROPOFF:
            if pass_idx == PASS_IN_TAXI and (row, col) == LOCS[dest_idx]:
                reward = 20
                dest_reached = True
            else:
                reward = -10

        return new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached
# ------------------------------------------------------------------
    # Internal transition logic (stochastic)
    # ------------------------------------------------------------------
    def _transition_stochastic(
        self, row: int, col: int, pass_idx: int, dest_idx: int, action: int
    ) -> tuple:
        """Stochastic transition: sample from the slip distribution."""
        outcomes = self.get_stochastic_outcomes(row, col, pass_idx, dest_idx, action)
        probs = [o[0] for o in outcomes]
        idx = self._rng.choice(len(outcomes), p=probs)
        _, nr, nc, np_, nd, r, dr = outcomes[idx]
        return nr, nc, np_, nd, r, dr

    # ------------------------------------------------------------------
    # Public helpers used by build_model (mdp.py)
    # ------------------------------------------------------------------
    def get_deterministic_outcome(
        self, row: int, col: int, pass_idx: int, dest_idx: int, action: int
    ) -> tuple:
        """Deterministic outcome — used by build_model for original variant."""
        return self._transition_deterministic(row, col, pass_idx, dest_idx, action)

    def get_stochastic_outcomes(
        self, row: int, col: int, pass_idx: int, dest_idx: int, action: int
    ) -> list:
        """Full stochastic outcome list — used by build_model for stochastic variant.

        Each element: (prob, new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached)

        For movement actions: intended direction + two slip directions.
        For PICKUP/DROPOFF: deterministic (prob=1.0).
        """
        p_intended = 1.0 - 2.0 * self.slip_probability
        outcomes = []

        if action in (SOUTH, NORTH, EAST, WEST):
            # Intended outcome
            nr, nc = _compute_new_position(row, col, action)
            if check_wall(row, col, nr, nc):
                nr, nc = row, col
            outcomes.append((p_intended, nr, nc, pass_idx, dest_idx, -1, False))

            # Left slip
            slip_l = _SLIP_MAP[action][0]
            nr_l, nc_l = _compute_new_position(row, col, slip_l)
            if check_wall(row, col, nr_l, nc_l):
                nr_l, nc_l = row, col
            outcomes.append((self.slip_probability, nr_l, nc_l, pass_idx, dest_idx, -1, False))

            # Right slip
            slip_r = _SLIP_MAP[action][1]
            nr_r, nc_r = _compute_new_position(row, col, slip_r)
            if check_wall(row, col, nr_r, nc_r):
                nr_r, nc_r = row, col
            outcomes.append((self.slip_probability, nr_r, nc_r, pass_idx, dest_idx, -1, False))
        else:
            # PICKUP / DROPOFF — deterministic even in stochastic variant
            nr, nc, np_, nd, r, dr = self._transition_deterministic(
                row, col, pass_idx, dest_idx, action
            )
            outcomes.append((1.0, nr, nc, np_, nd, r, dr))

        return outcomes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _spawn_new_passenger(self) -> tuple[int, int]:
        """Sample a new (pickup_location, dropoff_location) without replacement."""
        pickup, dropoff = self._rng.choice(4, size=2, replace=False)
        return int(pickup), int(dropoff)

    def render(self):
        """String visualization of the environment."""
        desc = [[" " for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
        row, col, pass_idx, dest_idx = self.state
        for i, (lr, lc) in enumerate(LOCS):
            desc[lr][lc] = str(i)
        if pass_idx != PASS_IN_TAXI:
            pr, pc = LOCS[pass_idx]
            desc[pr][pc] = f"P{desc[pr][pc]}" if desc[pr][pc] != " " else "P"
        desc[row][col] = f"T{desc[row][col]}" if desc[row][col] != " " else "T"
        lines = ["+" + "---+" * NUM_COLS]
        for r in range(NUM_ROWS):
            cells = []
            for c in range(NUM_COLS):
                cells.append(f" {desc[r][c]:1} ")
            lines.append("|" + "|".join(cells) + "|")
            lines.append("+" + "---+" * NUM_COLS)
        return "\n".join(lines)

    def close(self):
        pass