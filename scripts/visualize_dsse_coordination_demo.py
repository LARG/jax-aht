"""3-panel DSSE coordination demo: shows what n_drones_to_rescue does.

Renders three side-by-side scenarios on the same seed/layout:
  Panel 1: ndr=1, one drone moves to the target alone -> rescue.
  Panel 2: ndr=2, one drone moves to the target alone -> no rescue
           (the second drone idles at its spawn).
  Panel 3: ndr=2, both drones move to the target together -> rescue.

Each panel shows the final probability heatmap, the drone trajectories
with fading alpha, and a green/red star at the target position depending
on whether the rescue actually happened. The point is to make it visually
obvious that ndr>=2 turns DSSE into a coordination task: a perfectly
competent solo agent that solves ndr=1 fails on ndr=2.

Output: writeup/imgs/dsse_coordination_demo.png

Usage:
    PYTHONPATH=. MPLBACKEND=Agg python scripts/visualize_dsse_coordination_demo.py
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

from envs.dsse.dsse_jax import DSSEJax


def _greedy_action(env, drone_xy):
    """Greedy single-drone policy: walk toward argmax of prob_matrix.

    Returns an int action 0-8. drone_xy is a length-2 jnp array (x, y).
    Uses Chebyshev distance (8-connectivity) since DSSE supports diagonals.
    """
    # Built outside JIT — use plain Python ints to keep this readable.
    return 8  # placeholder, see _select_actions which closes over state


def _select_actions(state, env, ego_only):
    """Pick actions for the two drones in this scenario.

    If ego_only is True, drone 1 just SEARCHes in place every step (acts
    as a passive teammate that does nothing useful). If ego_only is False,
    both drones walk toward the (oracle) target position.

    For this demo we cheat and use the true target position as the goal,
    so the only variable across panels is whether the teammate helps.
    The actual training agents do not get oracle target access — they see
    the prob_matrix and have to infer it.
    """
    target_x = int(state.target_positions[0, 0])
    target_y = int(state.target_positions[0, 1])

    def greedy(i):
        x = int(state.drone_positions[i, 0])
        y = int(state.drone_positions[i, 1])
        dx = np.sign(target_x - x)
        dy = np.sign(target_y - y)
        if dx == 0 and dy == 0:
            return 8  # SEARCH
        if dx == -1 and dy == 0:
            return 0  # LEFT
        if dx == 1 and dy == 0:
            return 1  # RIGHT
        if dx == 0 and dy == -1:
            return 2  # UP
        if dx == 0 and dy == 1:
            return 3  # DOWN
        if dx == -1 and dy == -1:
            return 4  # UP_LEFT
        if dx == 1 and dy == -1:
            return 5  # UP_RIGHT
        if dx == -1 and dy == 1:
            return 6  # DOWN_LEFT
        if dx == 1 and dy == 1:
            return 7  # DOWN_RIGHT
        return 8

    a0 = greedy(0)
    a1 = 8 if ego_only else greedy(1)  # passive teammate just SEARCHes in place
    return jnp.array([a0, a1], dtype=jnp.int32)


def run_scenario(ndr, ego_only, seed=42, max_steps=40):
    """Roll out one episode and return (env, traj, final_state, rescued).

    traj is a list of (drone_positions_np, prob_matrix_np) tuples, one per
    step including the initial state.
    """
    env = DSSEJax(
        grid_size=7,
        n_drones=2,
        n_targets=1,
        timestep_limit=max_steps,
        n_drones_to_rescue=ndr,
        target_cluster_radius=1,
    )
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    traj = [(np.array(state.drone_positions), np.array(state.prob_matrix))]
    rescued = False
    last_t = 0
    for t in range(max_steps):
        actions = _select_actions(state, env, ego_only)
        rng, rng_step = jax.random.split(rng)
        state, _, done, _ = env.step(rng_step, state, actions)
        traj.append((np.array(state.drone_positions), np.array(state.prob_matrix)))
        last_t = t + 1
        if bool(state.targets_found.all()):
            rescued = True
            break
        if bool(done):
            break

    return env, traj, state, rescued, last_t


def render(out_path, seed=42, max_steps=40):
    scenarios = [
        ("ndr=1, solo greedy",   1, True,  "rescued"),   # rescue should succeed
        ("ndr=2, solo greedy",   2, True,  "no rescue"), # rescue should fail
        ("ndr=2, both greedy",   2, False, "rescued"),   # rescue should succeed
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    drone_colors = ["#1976D2", "#43A047"]  # blue (ego), green (teammate)
    target_ok_color = "#2E7D32"
    target_fail_color = "#C62828"

    for ax, (label, ndr, ego_only, _expected) in zip(axes, scenarios):
        env, traj, final_state, rescued, last_t = run_scenario(
            ndr=ndr, ego_only=ego_only, seed=seed, max_steps=max_steps,
        )
        gs = env.grid_size

        # Background: final probability heatmap (where the search field
        # has drifted to by the end of the episode)
        prob = traj[-1][1]
        ax.imshow(
            prob, cmap="YlOrRd", interpolation="nearest", origin="upper",
            extent=[-0.5, gs - 0.5, gs - 0.5, -0.5],
            norm=Normalize(vmin=0, vmax=prob.max() + 1e-8),
        )

        # Grid lines
        for i in range(gs + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
            ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)

        # Trajectory overlay: one polyline per drone, with fading alpha so
        # earlier steps are faint and later steps are saturated. Draw the
        # teammate first with a thicker line so the ego trajectory remains
        # visible even when both drones overlap (panel 3).
        n_steps = len(traj)
        line_widths = {0: 2.5, 1: 5.0}  # ego thinner, teammate thicker
        for d in [1, 0]:
            xs = [step[0][d, 0] for step in traj]
            ys = [step[0][d, 1] for step in traj]
            for k in range(n_steps - 1):
                alpha = 0.25 + 0.75 * (k / max(n_steps - 1, 1))
                ax.plot(
                    [xs[k], xs[k + 1]], [ys[k], ys[k + 1]],
                    color=drone_colors[d], linewidth=line_widths[d], alpha=alpha,
                    solid_capstyle="round",
                )
            # Mark start and end positions
            ax.plot(xs[0], ys[0], "o", color=drone_colors[d],
                    markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                    alpha=0.5)
            ax.plot(xs[-1], ys[-1], "o", color=drone_colors[d],
                    markersize=14 if d == 0 else 16,
                    markeredgecolor="white", markeredgewidth=2.0)

        # Target: green star if rescued, red if not
        tx = int(final_state.target_positions[0, 0])
        ty = int(final_state.target_positions[0, 1])
        target_color = target_ok_color if rescued else target_fail_color
        ax.plot(tx, ty, "*", color=target_color, markersize=24,
                markeredgecolor="white", markeredgewidth=1.8)

        # Title and outcome
        outcome = "rescued" if rescued else "no rescue"
        ax.set_title(f"{label}\n{outcome} (t={last_t})", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(gs - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # One shared legend below the panels with explicit padding so it
    # never overlaps the bottom row of cells. tight_layout() shrinks
    # the panel area to make room for the legend rect.
    handles = [
        mpatches.Patch(color=drone_colors[0], label="Drone 0 (ego)"),
        mpatches.Patch(color=drone_colors[1], label="Drone 1 (teammate)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=target_ok_color, markersize=14,
                   label="Target (rescued)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=target_fail_color, markersize=14,
                   label="Target (not rescued)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.02))

    # Reserve a strip at the bottom of the figure for the legend so
    # it does not collide with the panels. The 0.10 leaves about a
    # legend-row of vertical space.
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--output", default="writeup/imgs/dsse_coordination_demo.png")
    args = parser.parse_args()

    render(out_path=args.output, seed=args.seed, max_steps=args.max_steps)
