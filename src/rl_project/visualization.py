"""
Interactive visualization for PI vs VI comparison on MilanTaxi.

Provides:
- render_taxi_grid: Draw a static 5x5 taxi grid with taxi, passenger, destination.
- animate_pi_vs_vi: Creates an ipywidgets interactive side-by-side comparison.

All text is in Spanish.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import asyncio
from IPython.display import display

from rl_project.envs.milan_taxi import (
    MilanTaxiEnv, N_S, N_A, HORIZON,
    encode_state, decode_state,
    SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, LOCS, PASS_IN_TAXI,
    NUM_ROWS, NUM_COLS, INTERNAL_WALLS,
)

_ACTION_NAMES = {
    SOUTH: "SUR",
    NORTH: "NORTE",
    EAST: "ESTE",
    WEST: "OESTE",
    PICKUP: "RECOGER",
    DROPOFF: "DEJAR",
}

# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------
def render_taxi_grid(
    state: tuple[int, int, int, int] | int,
    action: int | None = None,
    reward: float | None = None,
    ax: plt.Axes | None = None,
    title: str = "",
    show_info: bool = True,
) -> plt.Axes:
    """Render the MilanTaxi 5x5 grid on *ax* with annotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if isinstance(state, (int, np.integer)):
        state = decode_state(int(state))
    row, col, pass_idx, dest_idx = state

    ax.clear()
    ax.set_xlim(0, NUM_COLS)
    ax.set_ylim(0, NUM_ROWS)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(NUM_COLS + 1))
    ax.set_yticks(range(NUM_ROWS + 1))
    ax.tick_params(which="both", labelleft=False, labelbottom=False, length=0)
    ax.grid(True, color="black", linewidth=1.5)

    # Walls
    for (r1, c1), (r2, c2) in INTERNAL_WALLS:
        if r1 == r2:
            ax.plot([c1 + 1, c2 + 1], [r1, r1], color="black", linewidth=4, zorder=5)
        elif c1 == c2:
            ax.plot([c1, c1], [r1 + 1, r2 + 1], color="black", linewidth=4, zorder=5)

    # Landmarks
    for i, (lr, lc) in enumerate(LOCS):
        ax.add_patch(
            FancyBboxPatch(
                (lc + 0.05, lr + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.1",
                facecolor="lightblue", edgecolor="blue", linewidth=1.5, zorder=2,
            )
        )
        ax.text(lc + 0.5, lr + 0.5, str(i), ha="center", va="center",
                fontsize=12, fontweight="bold", color="blue", zorder=3)

    # Passenger
    if pass_idx != PASS_IN_TAXI:
        pr, pc = LOCS[pass_idx]
        ax.text(pc + 0.5, pr + 0.3, "\U0001f464", ha="center", va="center",
                fontsize=18, zorder=5)

    # Destination
    dr, dc = LOCS[dest_idx]
    ax.text(dc + 0.5, dr + 0.7, "\U0001f3af", ha="center", va="center",
            fontsize=14, zorder=5)

    # Taxi
    ax.text(col + 0.5, row + 0.5, "\U0001f695", ha="center", va="center",
            fontsize=24, zorder=6)

    # Title
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    # Info box
    if show_info:
        action_str = _ACTION_NAMES.get(action, "\u2014") if action is not None else "\u2014"
        reward_str = f"{reward:+.0f}" if reward is not None else "\u2014"
        state_str = f"fila={row} | col={col} | pasajero={pass_idx} | destino={dest_idx}"
        info_text = (
            f"Estado: {state_str}\n"
            f"Acci\u00f3n: {action_str}  |  Recompensa: {reward_str}"
        )
        ax.text(
            0.5, -0.35, info_text, ha="center", va="top",
            fontsize=9, family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
        )

    return ax


# ---------------------------------------------------------------------------
# Interactive widget — PI vs VI comparison
# ---------------------------------------------------------------------------
def animate_pi_vs_vi(
    pi_snapshots: list[dict],
    vi_snapshots: list[dict],
    env: MilanTaxiEnv,
    seed: int = 42,
    scenario_name: str = "Determinista",
    figsize: tuple = (14, 6),
):
    """Create an interactive ipywidgets panel comparing PI and VI."""
    import ipywidgets as widgets
    from ipywidgets import HBox, VBox, Layout

    n_pi = len(pi_snapshots)
    n_vi = len(vi_snapshots)

    # Precompute rollouts for each snapshot
    print("Precomputando rollouts...", end=" ")
    pi_rollouts = [compute_rollout(s["pi"], env, seed=seed) for s in pi_snapshots]
    vi_rollouts = [compute_rollout(s["pi"], env, seed=seed) for s in vi_snapshots]
    print("OK")

    pi_total = pi_snapshots[-1]["iteration"]
    vi_total = vi_snapshots[-1]["sweep"]

    # Widgets
    progress_slider = widgets.FloatSlider(
        value=0, min=0, max=100, step=1,
        description="Progreso:",
        style={"description_width": "initial"},
        layout=Layout(width="80%"),
        readout=True, readout_format=".0f",
    )
    step_slider = widgets.IntSlider(
        value=0, min=0, max=99, step=1,
        description="Paso taxi:",
        style={"description_width": "initial"},
        layout=Layout(width="60%"),
    )
    play_button = widgets.Play(
        value=0, min=0, max=100, step=1,
        interval=800, repeat=True, show_repeat=False,
    )
    widgets.jsdlink((play_button, "value"), (progress_slider, "value"))

    btn_prev = widgets.Button(description="\u25c0 Anterior")
    btn_next = widgets.Button(description="Siguiente \u25b6")
    btn_reset = widgets.Button(description="Reiniciar")

    speed_dropdown = widgets.Dropdown(
        options=[("Muy lenta (2s)", 2000), ("Lenta (1s)", 1000),
                 ("Normal (800ms)", 800), ("R\u00e1pida (400ms)", 400),
                 ("Muy r\u00e1pida (100ms)", 100)],
        value=800, description="Velocidad:",
        style={"description_width": "initial"},
    )
    speed_dropdown.observe(
        lambda c: setattr(play_button, "interval", c["new"]), names="value")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    fig.tight_layout(pad=3.0)
    plt.close(fig)

    scenario_label = widgets.HTML(
        value=(f"<b>Escenario:</b> {scenario_name}  |  "
               f"<b>\u03b3</b> = 0.99  |  <b>Horizonte</b> = {HORIZON}  |  "
               f"<b>Semilla</b> = {seed}")
    )
    if env.variant == "stochastic":
        scenario_label.value += (
            "  |  <b>P(intencionado)</b> = 0.8  "
            "<b>P(izquierda)</b> = 0.1  <b>P(derecha)</b> = 0.1")

    # Progress mapping
    def _progress_to_indices(pct):
        frac = pct / 100.0
        pi_i = min(int(round(frac * (n_pi - 1))), n_pi - 1)
        vi_i = min(int(round(frac * (n_vi - 1))), n_vi - 1)
        return pi_i, vi_i

    def _update(pi_idx, vi_idx, taxi_step):
        snap_pi = pi_snapshots[pi_idx]
        snap_vi = vi_snapshots[vi_idx]
        rl_pi = pi_rollouts[pi_idx]
        rl_vi = vi_rollouts[vi_idx]
        sp = min(taxi_step, len(rl_pi) - 1)
        sv = min(taxi_step, len(rl_vi) - 1)
        f1, f2 = rl_pi[sp], rl_vi[sv]

        s0 = encode_state(0, 0, 0, 1)
        pi_v0 = snap_pi["V"][s0]
        pi_t = (f"Iteraci\u00f3n PI: {snap_pi['iteration']} / {pi_total}\n"
                f"Paso taxi: {f1['step'] + 1} / {len(rl_pi)}\n"
                f"V(start) \u2248 {pi_v0:.1f}")
        ax_l.clear()
        render_taxi_grid(f1["state"], action=f1["action"],
                         reward=f1["reward"], ax=ax_l,
                         title=f"Iteraci\u00f3n de Pol\u00edtica (PI)\n{pi_t}")

        vi_v0 = snap_vi["V"][s0]
        vi_t = (f"Barrido VI: {snap_vi['sweep']} / {vi_total}\n"
                f"Paso taxi: {f2['step'] + 1} / {len(rl_vi)}\n"
                f"V(start) \u2248 {vi_v0:.1f}\n"
                f"\u0394 m\u00e1x = {snap_vi.get('delta', 0):.2e}")
        ax_r.clear()
        render_taxi_grid(f2["state"], action=f2["action"],
                         reward=f2["reward"], ax=ax_r,
                         title=f"Iteraci\u00f3n de Valores (VI)\n{vi_t}")
        fig.canvas.draw_idle()

    progress_slider.observe(
        lambda c: _update(*_progress_to_indices(c["new"]),
                          min(step_slider.value,
                              max(len(pi_rollouts[_progress_to_indices(c["new"])[0]]),
                                  len(vi_rollouts[_progress_to_indices(c["new"])[1]])) - 1)),
        names="value")
    step_slider.observe(lambda c: _update(*_progress_to_indices(progress_slider.value), c["new"]),
                        names="value")
    btn_prev.on_click(lambda _: setattr(progress_slider, "value",
                                        max(0, progress_slider.value - 5)))
    btn_next.on_click(lambda _: setattr(progress_slider, "value",
                                        min(100, progress_slider.value + 5)))
    btn_reset.on_click(lambda _: (setattr(progress_slider, "value", 0),
                                  setattr(step_slider, "value", 0)))

    # Initial render
    pi_i, vi_i = _progress_to_indices(0)
    _update(pi_i, vi_i, 0)

    controls = HBox([btn_prev, play_button, btn_next, btn_reset])
    side = HBox([step_slider, speed_dropdown])
    ui = VBox([scenario_label, progress_slider, controls, side])

    display(fig)
    display(ui)
    # Auto-play with loop — se reproduce automáticamente hasta que se pausa
    try:
        loop = asyncio.get_running_loop()
        loop.call_later(0.3, lambda: setattr(play_button, "_playing", True))
    except RuntimeError:
        play_button._playing = True
    return ui


def compute_rollout(
    policy: np.ndarray,
    env: MilanTaxiEnv,
    seed: int = 42,
    max_steps: int = HORIZON,
) -> list[dict]:
    """Run a single rollout using *policy* and return frame data.

    Each frame dict::
        {'state', 'action', 'reward', 'step', 'terminated'}
    """
    state, _ = env.reset(seed=seed)
    frames = []
    for step in range(max_steps):
        s_idx = encode_state(*state)
        action = int(policy[s_idx].argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        frames.append({
            "state": state,
            "action": action,
            "reward": reward,
            "step": step,
            "terminated": terminated or truncated,
        })
        state = next_state
        if terminated or truncated:
            break
    return frames