"""Render a DSSE episode as an animated GIF: probability heatmap,
drones, and targets, one frame per step. Produces
writeup/gifs/dsse_episode.gif.

Usage:
    PYTHONPATH=. python scripts/visualize_dsse_episode.py [--steps 100] [--output dsse_episode.gif]
"""

import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

from envs.dsse.dsse_jax import DSSEJax


def visualize_episode(
    grid_size=7, n_drones=4, n_targets=2, timestep_limit=100,
    n_drones_to_rescue=2,
    policy="greedy", output="dsse_episode.gif", seed=42, fps=4,
):
    env = DSSEJax(
        grid_size=grid_size, n_drones=n_drones, n_targets=n_targets,
        timestep_limit=timestep_limit, n_drones_to_rescue=n_drones_to_rescue,
    )

    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    frames = []
    drone_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63'][:n_drones]
    target_color = '#F44336'

    for t in range(timestep_limit):
        # Render frame
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Probability heatmap
        prob = np.array(state.prob_matrix)
        ax.imshow(prob, cmap='YlOrRd', interpolation='nearest',
                  origin='upper', extent=[-0.5, grid_size-0.5, grid_size-0.5, -0.5],
                  norm=Normalize(vmin=0, vmax=prob.max() + 1e-8))

        # Grid lines
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)

        # Targets (stars)
        for j in range(n_targets):
            if not state.targets_found[j]:
                tx, ty = int(state.target_positions[j, 0]), int(state.target_positions[j, 1])
                ax.plot(tx, ty, '*', color=target_color, markersize=18,
                        markeredgecolor='white', markeredgewidth=1.5)

        # Drones (circles)
        for i in range(n_drones):
            dx, dy = int(state.drone_positions[i, 0]), int(state.drone_positions[i, 1])
            ax.plot(dx, dy, 'o', color=drone_colors[i], markersize=14,
                    markeredgecolor='white', markeredgewidth=2)

        # Info text
        n_found = int(state.targets_found.sum())
        ax.set_title(f'DSSE  t={t}  found={n_found}/{n_targets}', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_aspect('equal')

        # Legend
        legend_elements = [
            mpatches.Patch(color=drone_colors[i], label=f'Drone {i}') for i in range(n_drones)
        ]
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                           markerfacecolor=target_color, markersize=12, label='Target'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # RGBA -> RGB
        frames.append(frame)
        plt.close(fig)

        # Select actions
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if policy == "greedy":
            actions = _greedy_actions(state, env)
        else:
            actions = jax.random.randint(rng_act, (n_drones,), 0, 9)

        state, rewards, done, info = env.step(rng_step, state, actions)
        if done:
            break

    # Save as GIF
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(output, save_all=True, append_images=imgs[1:],
                     duration=1000//fps, loop=0)
        print(f"Saved {len(frames)} frames to {output}")
    except ImportError:
        # Fallback: save last frame as PNG
        png_path = output.replace('.gif', '.png')
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.array(state.prob_matrix), cmap='YlOrRd', interpolation='nearest',
                  origin='upper', extent=[-0.5, grid_size-0.5, grid_size-0.5, -0.5])
        ax.set_title(f'DSSE Final State (t={t})')
        fig.savefig(png_path, dpi=150)
        print(f"Pillow not installed. Saved final frame to {png_path}")


def _greedy_actions(state, env):
    """Simple greedy: each drone moves toward highest-probability cell."""
    gs = env.grid_size
    prob = state.prob_matrix
    best_idx = jnp.argmax(prob.flatten())
    best_x = best_idx % gs
    best_y = best_idx // gs

    actions = []
    for i in range(env.n_drones):
        dx = jnp.sign(best_x - state.drone_positions[i, 0])
        dy = jnp.sign(best_y - state.drone_positions[i, 1])
        at_target = (dx == 0) & (dy == 0)

        action = jnp.where(
            at_target, 8,  # SEARCH
            jnp.where(dx == -1,
                jnp.where(dy == -1, 4, jnp.where(dy == 1, 6, 0)),
            jnp.where(dx == 1,
                jnp.where(dy == -1, 5, jnp.where(dy == 1, 7, 1)),
            jnp.where(dy == -1, 2, 3)))
        )
        actions.append(int(action))
    return jnp.array(actions, dtype=jnp.int32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults match the 2-drone ego training variant of the DSSE benchmark
    # so the rendered gif matches the headline configuration in the writeup.
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--n_drones", type=int, default=2)
    parser.add_argument("--n_targets", type=int, default=1)
    parser.add_argument("--n_drones_to_rescue", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--policy", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--output", default="writeup/gifs/dsse_episode.gif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    visualize_episode(
        grid_size=args.grid_size, n_drones=args.n_drones,
        n_targets=args.n_targets, timestep_limit=args.steps,
        n_drones_to_rescue=args.n_drones_to_rescue,
        policy=args.policy, output=args.output, seed=args.seed, fps=args.fps,
    )
