import argparse
import os
from typing import Dict, List

import jax
import jax.numpy as jnp

from envs import make_env
from envs.rddl.pizza_v2.PizzaV2MultiAgentViz import PizzaV2MultiAgentVisualizer


def _sample_random_actions(env, rng_key: jax.Array) -> Dict[str, jnp.ndarray]:
    """Sample one random discrete action per agent."""
    keys = jax.random.split(rng_key, len(env.agents))
    actions = {}
    for idx, agent in enumerate(env.agents):
        n_actions = int(env.action_space(agent).n)
        actions[agent] = jax.random.randint(keys[idx], shape=(), minval=0, maxval=n_actions, dtype=jnp.int32)
    return actions


def run_visualization_smoke_test(
    seed: int = 0,
    max_steps: int = 30,
    instance: str = "pizza_v2_instance_all.rddl",
    output_dir: str = "results/pizza_v2_visualization_smoke",
) -> None:
    """Run one random pizza_v2 episode and export visualization artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    env = make_env(
        "rddl/pizza_v2",
        {
            "render": False,
            "instance": instance,
            "render_name": "pizza_v2_smoke",
        },
    )

    # Avoid a Hydra runtime dependency in standalone scripts by attaching
    # the visualizer directly.
    env.env.set_visualizer(visualizer=PizzaV2MultiAgentVisualizer(env.env.model))

    rng = jax.random.PRNGKey(seed)
    obs, state = env.reset(rng)

    frames = []
    init_frame = env.render(state.env_state, save_frame=False)
    if init_frame is not None:
        frames.append(init_frame)

    done = False
    step = 0

    while not done and step < max_steps:
        rng, sample_key, step_key = jax.random.split(rng, 3)
        actions = _sample_random_actions(env, sample_key)

        obs, next_state, rewards, dones, info = env.step(step_key, state, actions)

        # The wrapper auto-resets at terminal states. Use pre_reset_state
        # to render the true terminal transition.
        vis_state = info.get("pre_reset_state", next_state)
        frame = env.render(vis_state.env_state, save_frame=False)
        if frame is not None:
            frames.append(frame)

        done = bool(dones["__all__"])
        state = next_state
        step += 1

    if len(frames) == 0:
        raise RuntimeError("No frames were rendered. Visualizer smoke test failed.")

    first_frame_path = os.path.join(output_dir, "pizza_v2_smoke_first_frame.png")
    gif_path = os.path.join(output_dir, "pizza_v2_smoke_episode.gif")

    frames[0].save(first_frame_path)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
    )

    print(f"Rendered {len(frames)} frames")
    print(f"Saved first frame: {first_frame_path}")
    print(f"Saved episode GIF: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pizza_v2 visualization smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--instance", type=str, default="pizza_v2_instance_all.rddl")
    parser.add_argument("--output-dir", type=str, default="results/pizza_v2_visualization_smoke")
    args = parser.parse_args()

    run_visualization_smoke_test(
        seed=args.seed,
        max_steps=args.max_steps,
        instance=args.instance,
        output_dir=args.output_dir,
    )
