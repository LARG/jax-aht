#!/usr/bin/env python
"""Coverage, overlap, coordination score, and per-agent visit heatmaps
for DSSE scripted policies. Produces policy_comparison.png and the
per-drone heatmaps used in the qualitative validity section of
writeup/dsse.tex.

Usage:
    PYTHONPATH=. python scripts/dsse_metrics.py [--episodes 100] [--output metrics_output/]
"""

import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from envs.dsse.dsse_jax import DSSEJax


def run_episode_with_metrics(env, rng, policy_fn):
    """Run one episode, collecting per-step metrics."""
    state = env.reset(rng)

    grid_visited = np.zeros((env.n_drones, env.grid_size, env.grid_size), dtype=np.int32)
    all_positions = []
    first_find_step = None
    total_reward = 0.0

    for t in range(env.timestep_limit):
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        actions = policy_fn(rng_act, state, env)

        # Track positions before step
        positions = np.array(state.drone_positions)
        all_positions.append(positions.copy())

        # Mark visited cells
        for i in range(env.n_drones):
            x, y = int(positions[i, 0]), int(positions[i, 1])
            grid_visited[i, y, x] += 1

        # Step
        state, rewards, done, info = env.step(rng_step, state, actions)
        total_reward += float(rewards.sum())

        if first_find_step is None and float(rewards.sum()) > 0:
            first_find_step = t

        if done:
            break

    # Compute metrics
    total_visited = (grid_visited.sum(axis=0) > 0).sum()
    total_cells = env.grid_size * env.grid_size
    coverage = total_visited / total_cells

    overlap_cells = (grid_visited.sum(axis=0) > 1).sum()
    overlap_ratio = overlap_cells / max(total_visited, 1)

    targets_found = int(np.array(state.targets_found).sum())
    all_found = targets_found == env.n_targets

    coordination = coverage - 0.5 * overlap_ratio

    return {
        'coverage': coverage,
        'overlap_ratio': overlap_ratio,
        'overlap_cells': int(overlap_cells),
        'first_find_step': first_find_step,
        'total_reward': total_reward,
        'targets_found': targets_found,
        'all_found': all_found,
        'episode_length': t + 1,
        'coordination_score': coordination,
        'grid_visited': grid_visited,
        'positions': np.array(all_positions),
    }


def random_policy(rng, state, env):
    return jax.random.randint(rng, (env.n_drones,), 0, 9)


def greedy_policy(rng, state, env):
    """Move toward highest probability cell, search when there."""
    gs = env.grid_size
    prob = np.array(state.prob_matrix)
    best_idx = np.argmax(prob)
    best_x, best_y = best_idx % gs, best_idx // gs

    actions = []
    for i in range(env.n_drones):
        x, y = int(state.drone_positions[i, 0]), int(state.drone_positions[i, 1])
        dx = np.sign(best_x - x)
        dy = np.sign(best_y - y)
        if dx == 0 and dy == 0:
            actions.append(8)  # SEARCH
        elif dx == -1 and dy == -1:
            actions.append(4)
        elif dx == -1 and dy == 0:
            actions.append(0)
        elif dx == -1 and dy == 1:
            actions.append(6)
        elif dx == 0 and dy == -1:
            actions.append(2)
        elif dx == 0 and dy == 1:
            actions.append(3)
        elif dx == 1 and dy == -1:
            actions.append(5)
        elif dx == 1 and dy == 0:
            actions.append(1)
        elif dx == 1 and dy == 1:
            actions.append(7)
        else:
            actions.append(8)
    return jnp.array(actions, dtype=jnp.int32)


def sweep_policy(rng, state, env):
    """Systematic left-to-right sweep, search every cell."""
    gs = env.grid_size
    actions = []
    for i in range(env.n_drones):
        x, y = int(state.drone_positions[i, 0]), int(state.drone_positions[i, 1])
        step = int(np.array(state.timestep))
        if step % 2 == 0:
            actions.append(8)  # Search
        else:
            actions.append(1)  # Move right
    return jnp.array(actions, dtype=jnp.int32)


def run_benchmark(n_episodes=100, output_dir='metrics_output'):
    os.makedirs(output_dir, exist_ok=True)

    env = DSSEJax(grid_size=7, n_drones=4, n_targets=2, timestep_limit=100, n_drones_to_rescue=2)

    policies = {
        'random': random_policy,
        'greedy': greedy_policy,
        'sweep': sweep_policy,
    }

    all_results = {}

    for policy_name, policy_fn in policies.items():
        print(f"\n=== {policy_name} policy ({n_episodes} episodes) ===")
        results = []

        for ep in range(n_episodes):
            rng = jax.random.PRNGKey(ep)
            metrics = run_episode_with_metrics(env, rng, policy_fn)
            results.append(metrics)

        # Aggregate
        coverages = [r['coverage'] for r in results]
        overlaps = [r['overlap_ratio'] for r in results]
        finds = [r['all_found'] for r in results]
        first_finds = [r['first_find_step'] for r in results if r['first_find_step'] is not None]
        rewards = [r['total_reward'] for r in results]
        coord_scores = [r['coordination_score'] for r in results]

        print(f"  Coverage:      {np.mean(coverages):.3f} +/- {np.std(coverages):.3f}")
        print(f"  Overlap ratio: {np.mean(overlaps):.3f} +/- {np.std(overlaps):.3f}")
        print(f"  Targets found: {np.mean(finds):.3f}")
        print(f"  Time to find:  {np.mean(first_finds):.1f} +/- {np.std(first_finds):.1f}" if first_finds else "  Time to find:  never")
        print(f"  Total reward:  {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
        print(f"  Coordination:  {np.mean(coord_scores):.3f} +/- {np.std(coord_scores):.3f}")

        all_results[policy_name] = results

        # Trajectory heatmap (from last episode)
        grid_visited = results[-1]['grid_visited']
        fig, axes = plt.subplots(1, env.n_drones + 1, figsize=(4 * (env.n_drones + 1), 4))

        for i in range(env.n_drones):
            axes[i].imshow(grid_visited[i], cmap='Blues', interpolation='nearest')
            axes[i].set_title(f'Drone {i}')

        # Combined
        combined = grid_visited.sum(axis=0)
        axes[-1].imshow(combined, cmap='YlOrRd', interpolation='nearest')
        axes[-1].set_title('Combined (overlap)')

        fig.suptitle(f'{policy_name} policy: per-drone visit heatmap', fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/heatmap_{policy_name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Comparison bar chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    policy_names = list(all_results.keys())

    metrics_to_plot = [
        ('Coverage', [np.mean([r['coverage'] for r in all_results[p]]) for p in policy_names]),
        ('Overlap', [np.mean([r['overlap_ratio'] for r in all_results[p]]) for p in policy_names]),
        ('Found rate', [np.mean([r['all_found'] for r in all_results[p]]) for p in policy_names]),
        ('Coordination', [np.mean([r['coordination_score'] for r in all_results[p]]) for p in policy_names]),
    ]

    for ax, (name, vals) in zip(axes, metrics_to_plot):
        bars = ax.bar(policy_names, vals, color=['#1565C0', '#FF8F00', '#4CAF50'])
        ax.set_title(name, fontweight='bold')
        ax.set_ylim(0, max(max(vals) * 1.2, 0.1))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', fontsize=9)

    fig.suptitle('DSSE Benchmark: Policy Comparison', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/policy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="metrics_output")
    args = parser.parse_args()
    run_benchmark(args.episodes, args.output)
