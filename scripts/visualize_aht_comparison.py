"""Three-scenario rescue-rate plot for the 7x7 / 2-drone / ndr=2 task.

Runs the headline benchmark variant under three scripted teammates and
counts rescue rates: (1) Coordinated, both greedy to the target; (2)
Antagonist, teammate greedy to the opposite corner; (3) Random,
teammate samples uniform actions. Output: writeup/imgs/aht_comparison.png.

Usage:
    PYTHONPATH=envs/dsse MPLBACKEND=Agg python scripts/visualize_aht_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'envs', 'dsse'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from PIL import Image
from dsse_jax import DSSEJax


def greedy_toward(state, agent_idx, target_pos, gs):
    x = int(state.drone_positions[agent_idx, 0])
    y = int(state.drone_positions[agent_idx, 1])
    tx, ty = int(target_pos[0]), int(target_pos[1])
    dx, dy = np.sign(tx - x), np.sign(ty - y)
    if dx == 0 and dy == 0: return 8  # SEARCH
    if dx == -1: return 4 if dy == -1 else (6 if dy == 1 else 0)
    if dx == 1: return 5 if dy == -1 else (7 if dy == 1 else 1)
    return 2 if dy < 0 else 3


def eval_rescue_rate(env, get_actions_fn, n_trials=200, base_seed=1000):
    """Run n_trials rollouts and count how many end with all targets rescued.

    A rescue is when state.targets_found.sum() reaches env.n_targets before
    the episode terminates. With n_targets=1 this is just "the single target
    was found at least once during the episode", which is the same metric
    used by the headline experiments in writeup/dsse.tex.
    """
    rescues = 0
    for trial in range(n_trials):
        rng = jax.random.PRNGKey(base_seed + trial)
        state = env.reset(rng)
        rescued = False
        for t in range(env.timestep_limit):
            rng, rng_s = jax.random.split(rng)
            actions = get_actions_fn(state, rng, t)
            state, _rewards, done, _info = env.step(rng_s, state, jnp.array(actions))
            if int(state.targets_found.sum()) == env.n_targets:
                rescued = True
                break
            if bool(done):
                break
        if rescued:
            rescues += 1
    return rescues, n_trials


def run_scenario(env, rng, get_actions_fn, label):
    state = env.reset(rng)
    gs = env.grid_size
    frames = []

    for t in range(40):
        rng, rng_s = jax.random.split(rng)
        actions = get_actions_fn(state, rng, t)

        # Render frame
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        prob = np.array(state.prob_matrix)
        ax.imshow(prob, cmap='YlOrRd', interpolation='nearest', origin='upper',
                  extent=[-0.5, gs-0.5, gs-0.5, -0.5],
                  norm=Normalize(vmin=0, vmax=max(prob.max(), 0.01)))

        for i in range(gs + 1):
            ax.axhline(i - 0.5, color='gray', lw=0.2, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', lw=0.2, alpha=0.3)

        # Targets
        for j in range(env.n_targets):
            if not state.targets_found[j]:
                tx, ty = int(state.target_positions[j, 0]), int(state.target_positions[j, 1])
                ax.plot(tx, ty, '*', color='#D32F2F', markersize=18,
                        markeredgecolor='white', markeredgewidth=1, zorder=10)

        # Drones: 0=ego (blue), 1=teammate (orange)
        colors = ['#1565C0', '#FF8F00', '#9E9E9E', '#9E9E9E']
        labels_d = ['EGO', 'TEAM', 'D2', 'D3']
        for i in range(env.n_drones):
            dx, dy = int(state.drone_positions[i, 0]), int(state.drone_positions[i, 1])
            ax.plot(dx, dy, 'o', color=colors[i], markersize=22,
                    markeredgecolor='white', markeredgewidth=2.5, zorder=10)
            ax.text(dx, dy, labels_d[i], ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', zorder=11)

        found = int(state.targets_found.sum())
        ax.set_title(f'{label}\nt={t} found={found}/{env.n_targets}', fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(gs - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame.copy())
        plt.close(fig)

        state, rewards, done, info = env.step(rng_s, state, jnp.array(actions))
        if done:
            # Repeat last frame
            for _ in range(5): frames.append(frame.copy())
            break

    return frames


def _opposite_corner(target_pos, gs):
    """Return the cell diagonally opposite the target on a gs x gs grid."""
    tx, ty = int(target_pos[0]), int(target_pos[1])
    cx = 0 if tx >= gs // 2 else gs - 1
    cy = 0 if ty >= gs // 2 else gs - 1
    return np.array([cx, cy])


def main():
    # Match the headline experimental task in writeup/dsse.tex:
    # 7x7 grid, 2 drones, 1 target, ndr=2.
    env = DSSEJax(grid_size=7, n_drones=2, n_targets=1,
                  timestep_limit=50, n_drones_to_rescue=2)

    rng = jax.random.PRNGKey(7)

    # Scenario 1: Coordinated (both drones greedy toward the target)
    def coordinated(state, rng, t):
        target = np.array(state.target_positions[0])
        return [
            greedy_toward(state, 0, target, env.grid_size),
            greedy_toward(state, 1, target, env.grid_size),
        ]

    # Scenario 2: Antagonist (ego greedy, teammate runs to opposite corner)
    def antagonist(state, rng, t):
        target = np.array(state.target_positions[0])
        far = _opposite_corner(target, env.grid_size)
        return [
            greedy_toward(state, 0, target, env.grid_size),
            greedy_toward(state, 1, far, env.grid_size),
        ]

    # Scenario 3: Random teammate (ego greedy, teammate uniform random)
    def random_team(state, rng, t):
        target = np.array(state.target_positions[0])
        return [
            greedy_toward(state, 0, target, env.grid_size),
            int(jax.random.randint(rng, (), 0, 9)),
        ]

    print("Measuring rescue rates over 200 trials per scenario...")
    coord_r, n_trials = eval_rescue_rate(env, coordinated)
    antag_r, _ = eval_rescue_rate(env, antagonist)
    rand_r, _ = eval_rescue_rate(env, random_team)
    print(f"  Coordinated:       {coord_r}/{n_trials}")
    print(f"  Antagonist team:   {antag_r}/{n_trials}")
    print(f"  Random teammate:   {rand_r}/{n_trials}")

    print("Generating scenarios...")
    frames_coord = run_scenario(env, rng, coordinated,
                                "Coordinated\n(both drones greedy to target)")
    frames_antag = run_scenario(env, rng, antagonist,
                                "Antagonist team\n(teammate runs to far corner)")
    frames_rand = run_scenario(env, rng, random_team,
                               "Random teammate\n(teammate samples uniform actions)")

    # Combine side by side
    max_len = max(len(frames_coord), len(frames_antag), len(frames_rand))
    for lst in [frames_coord, frames_antag, frames_rand]:
        while len(lst) < max_len:
            lst.append(lst[-1])

    combined = []
    for i in range(max_len):
        row = np.concatenate([frames_coord[i], frames_antag[i], frames_rand[i]], axis=1)
        combined.append(row)

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'aht_comparison.gif')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imgs = [Image.fromarray(f) for f in combined]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=500, loop=0)
    print(f"Saved: {out}")

    # Save summary panel: last frame of each scenario with rescue rate.
    summary = os.path.join(os.path.dirname(__file__), '..', 'results', 'aht_comparison.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, frames, title in zip(axes,
        [frames_coord, frames_antag, frames_rand],
        [f"Coordinated: {coord_r}/{n_trials} rescues\n(both drones greedy to target)",
         f"Antagonist team: {antag_r}/{n_trials} rescues\n(teammate runs to far corner)",
         f"Random teammate: {rand_r}/{n_trials} rescues\n(teammate samples uniform actions)"]):
        ax.imshow(frames[-1])  # Last frame = final outcome
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    fig.suptitle(
        "DSSE 7x7, n_drones=2, n_targets=1, n_drones_to_rescue=2: "
        "the gate fires only under joint action",
        fontsize=14, fontweight='bold', y=1.02)
    # Reserve a strip at the top for the suptitle so it does not collide
    # with the per-panel titles below it.
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(summary, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
