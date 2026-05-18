import argparse
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from agents.bc.bc_lstm import BCLSTMConfig, BCLSTMNetwork, compute_bc_loss, BCLSTMAgent

LBF_CONFIG_NAMES = [
    "grid7_food3_nolevels",
    "grid7_food3_levels",
    "grid12_food6_nolevels",
    "grid12_food6_levels",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class BCDataSplits(NamedTuple):
    train_obs: jnp.ndarray
    train_actions: jnp.ndarray
    train_avail: jnp.ndarray
    train_mask: jnp.ndarray
    val_obs: jnp.ndarray
    val_actions: jnp.ndarray
    val_avail: jnp.ndarray
    val_mask: jnp.ndarray
    source: str


def load_data(data_dir: str):
    data_path = Path(data_dir)
    keys = ["obs", "actions", "avail", "mask"]

    from safetensors.numpy import load_file

    required = [f"{s}_{k}.safetensors" for s in ["train", "val"] for k in keys]
    missing = [f for f in required if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}")

    arrays = {}
    for split in ["train", "val"]:
        for key in keys:
            raw = load_file(str(data_path / f"{split}_{key}.safetensors"))
            arrays[f"{split}_{key}"] = jnp.array(raw["data"])
    return tuple(arrays[f"{s}_{k}"] for s in ["train", "val"] for k in keys)


def load_lbf_data(config_name: str,
                  seed: int,
                  val_fraction: float,
                  exclude_uncertain: bool) -> BCDataSplits:
    from human_data_processing.load_lbf_data import load_bc_data_padded

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1")

    data = load_bc_data_padded(
        config_name=config_name,
        exclude_uncertain=exclude_uncertain,
    )
    num_episodes = int(data.obs.shape[0])
    if num_episodes < 2:
        raise ValueError(
            f"LBF config {config_name} only has {num_episodes} episodes; "
            "need at least 2 for a train/val split"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    val_size = max(1, int(round(num_episodes * val_fraction)))
    val_size = min(val_size, num_episodes - 1)
    val_idx = jnp.array(perm[:val_size])
    train_idx = jnp.array(perm[val_size:])

    return BCDataSplits(
        train_obs=data.obs[train_idx],
        train_actions=data.actions[train_idx],
        train_avail=data.avail_actions[train_idx],
        train_mask=data.mask[train_idx],
        val_obs=data.obs[val_idx],
        val_actions=data.actions[val_idx],
        val_avail=data.avail_actions[val_idx],
        val_mask=data.mask[val_idx],
        source=f"lbf:{config_name}",
    )


def pad_obs(obs: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    obs_dim = int(obs.shape[-1])
    if obs_dim > target_dim:
        raise ValueError(f"Cannot pad obs_dim={obs_dim} down to {target_dim}")
    if obs_dim == target_dim:
        return obs
    return jnp.pad(obs, [(0, 0), (0, 0), (0, target_dim - obs_dim)])


def load_pooled_lbf_data(config_names: list[str],
                         seed: int,
                         val_fraction: float,
                         exclude_uncertain: bool) -> BCDataSplits:
    splits = [
        load_lbf_data(
            config_name=config_name,
            seed=seed + i,
            val_fraction=val_fraction,
            exclude_uncertain=exclude_uncertain,
        )
        for i, config_name in enumerate(config_names)
    ]
    obs_dim = max(int(split.train_obs.shape[-1]) for split in splits)

    return BCDataSplits(
        train_obs=jnp.concatenate(
            [pad_obs(split.train_obs, obs_dim) for split in splits], axis=0
        ),
        train_actions=jnp.concatenate([split.train_actions for split in splits], axis=0),
        train_avail=jnp.concatenate([split.train_avail for split in splits], axis=0),
        train_mask=jnp.concatenate([split.train_mask for split in splits], axis=0),
        val_obs=jnp.concatenate(
            [pad_obs(split.val_obs, obs_dim) for split in splits], axis=0
        ),
        val_actions=jnp.concatenate([split.val_actions for split in splits], axis=0),
        val_avail=jnp.concatenate([split.val_avail for split in splits], axis=0),
        val_mask=jnp.concatenate([split.val_mask for split in splits], axis=0),
        source=f"lbf_pooled:{','.join(config_names)}",
    )


def load_training_data(args) -> BCDataSplits:
    if args.lbf_config:
        config_names = args.lbf_config
        if config_names == ["all"]:
            config_names = LBF_CONFIG_NAMES
        if len(config_names) == 1:
            return load_lbf_data(
                config_name=config_names[0],
                seed=args.seed,
                val_fraction=args.val_fraction,
                exclude_uncertain=args.exclude_uncertain,
            )
        return load_pooled_lbf_data(
            config_names=config_names,
            seed=args.seed,
            val_fraction=args.val_fraction,
            exclude_uncertain=args.exclude_uncertain,
        )

    if not args.data_dir:
        raise ValueError("Either --data_dir or --lbf_config is required")

    (train_obs, train_actions, train_avail, train_mask,
     val_obs, val_actions, val_avail, val_mask) = load_data(args.data_dir)
    return BCDataSplits(
        train_obs=train_obs,
        train_actions=train_actions,
        train_avail=train_avail,
        train_mask=train_mask,
        val_obs=val_obs,
        val_actions=val_actions,
        val_avail=val_avail,
        val_mask=val_mask,
        source=f"files:{args.data_dir}",
    )


def count_unavailable_expert_actions(actions, avail, mask) -> int:
    picked = jnp.take_along_axis(avail, actions[..., None], axis=-1).squeeze(-1)
    return int(jnp.sum(mask & ~picked))


def remap_unavailable_actions_to_noop(actions, avail, mask) -> jnp.ndarray:
    picked = jnp.take_along_axis(avail, actions[..., None], axis=-1).squeeze(-1)
    unavailable = mask & ~picked
    return jnp.where(unavailable, jnp.zeros_like(actions), actions)


def train(args):
    rng = jax.random.PRNGKey(args.seed)

    defaults = {}
    if args.config:
        defaults = load_config(args.config)
        print(f"[train_bc] loaded config from {args.config}")

    data = load_training_data(args)
    train_obs = data.train_obs
    train_actions = data.train_actions
    train_avail = data.train_avail
    train_mask = data.train_mask
    val_obs = data.val_obs
    val_actions = data.val_actions
    val_avail = data.val_avail
    val_mask = data.val_mask

    obs_dim = int(train_obs.shape[-1])
    action_dim = int(train_avail.shape[-1])
    print(f"[train_bc] source={data.source}")
    print(f"[train_bc] obs_dim={obs_dim} action_dim={action_dim}")
    print(f"[train_bc] train: {train_obs.shape[0]} episodes, "
          f"val: {val_obs.shape[0]} episodes")
    print(
        "[train_bc] unavailable expert actions: "
        f"train={count_unavailable_expert_actions(train_actions, train_avail, train_mask)} "
        f"val={count_unavailable_expert_actions(val_actions, val_avail, val_mask)}"
    )
    if args.invalid_action_mode == "noop":
        train_actions = remap_unavailable_actions_to_noop(
            train_actions, train_avail, train_mask
        )
        val_actions = remap_unavailable_actions_to_noop(
            val_actions, val_avail, val_mask
        )
        print("[train_bc] remapped unavailable expert actions to NOOP")

    dropout = getattr(args, 'dropout', 0.0) or defaults.get("dropout", 0.0)
    config = BCLSTMConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        preprocess_dim=args.preprocess_dim or defaults.get("preprocess_dim", 256),
        lstm_dim=args.lstm_dim or defaults.get("lstm_dim", 128),
        postprocess_dim=args.postprocess_dim or defaults.get("postprocess_dim", 64),
        dropout_rate=dropout,
    )
    print(f"[train_bc] config: {config}")

    network = BCLSTMNetwork(
        action_dim=config.action_dim,
        preprocess_dim=config.preprocess_dim,
        lstm_dim=config.lstm_dim,
        postprocess_dim=config.postprocess_dim,
        dropout_rate=config.dropout_rate,
    )

    rng, init_rng = jax.random.split(rng)
    dummy_carry = (jnp.zeros((config.lstm_dim,)), jnp.zeros((config.lstm_dim,)))
    dummy_obs = jnp.zeros((config.obs_dim,))
    variables = network.init(init_rng, dummy_carry, dummy_obs)

    tx = optax.adam(args.lr)
    state = TrainState.create(
        apply_fn=network.apply, params=variables['params'], tx=tx)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"[train_bc] {param_count:,} parameters")

    batch_size = min(args.batch_size, train_obs.shape[0])
    best_val_acc = -1.0
    best_params = None

    @jax.jit
    def train_step(state, batch_obs, batch_actions, batch_avail, batch_mask):
        init_carry = (
            jnp.zeros((batch_obs.shape[0], config.lstm_dim)),
            jnp.zeros((batch_obs.shape[0], config.lstm_dim)),
        )
        def loss_fn(params):
            return compute_bc_loss(
                params, network, init_carry,
                batch_obs, batch_actions, batch_avail, batch_mask,
                mask_unavailable_actions=args.mask_unavailable_actions,
            )
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_accuracy(params, obs, actions, avail, mask):
        init_carry = (
            jnp.zeros((obs.shape[0], config.lstm_dim)),
            jnp.zeros((obs.shape[0], config.lstm_dim)),
        )
        batch_size, seq_len, _ = obs.shape
        def step_fn(carry, t):
            avail_t = avail[:, t, :]
            carry, logits = network.apply(
                {'params': params}, carry, obs[:, t, :])
            if args.mask_unavailable_actions:
                logits = jnp.where(avail_t, logits, -1e9)
            preds = jnp.argmax(logits, axis=-1)
            action_t = actions[:, t]
            if args.mask_unavailable_actions:
                action_available = jnp.take_along_axis(
                    avail_t, action_t[:, None], axis=-1
                ).squeeze(-1)
                valid = mask[:, t] & action_available
            else:
                valid = mask[:, t]
            correct = (preds == action_t) & valid
            return carry, (correct, valid)
        _, (all_correct, all_valid) = jax.lax.scan(
            step_fn, init_carry, jnp.arange(seq_len))
        return all_correct.sum() / jnp.maximum(all_valid.sum(), 1.0)

    for epoch in range(args.epochs):
        t0 = time.time()
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, train_obs.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, train_obs.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:
                continue
            state, loss = train_step(
                state, train_obs[idx], train_actions[idx],
                train_avail[idx], train_mask[idx],
            )
            epoch_loss += float(loss)
            n_batches += 1

        val_acc = float(eval_accuracy(
            state.params, val_obs, val_actions, val_avail, val_mask,
        ))
        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0

        print(f"  epoch {epoch+1:3d}/{args.epochs}: loss={avg_loss:.4f} "
              f"val_acc={val_acc:.3f} ({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = jax.tree.map(lambda x: x.copy(), state.params)

    print(f"\n[train_bc] best val accuracy: {best_val_acc:.4f}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    config_path = args.output.replace('.safetensors', '.yaml')
    with open(config_path, 'w') as f:
        yaml.dump({
            'obs_dim': config.obs_dim,
            'action_dim': config.action_dim,
            'preprocess_dim': config.preprocess_dim,
            'lstm_dim': config.lstm_dim,
            'postprocess_dim': config.postprocess_dim,
            'dropout_rate': config.dropout_rate,
            'val_accuracy': float(best_val_acc),
            'data_dir': args.data_dir,
            'lbf_config': args.lbf_config,
            'val_fraction': args.val_fraction,
            'exclude_uncertain': args.exclude_uncertain,
            'invalid_action_mode': args.invalid_action_mode,
            'mask_unavailable_actions': args.mask_unavailable_actions,
            'epochs': args.epochs,
            'lr': args.lr,
            'seed': args.seed,
        }, f)
    print(f"[train_bc] saved config to {config_path}")

    agent = BCLSTMAgent(config, params=best_params)
    agent.save_weights(args.output)
    print(f"[train_bc] saved weights to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train BC-LSTM from preprocessed safetensors data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_dir', default=None,
                        help="Directory with {train,val}_{obs,actions,avail,mask}.safetensors")
    parser.add_argument('--lbf_config', nargs='+', default=None,
                        choices=[
                            "all",
                            "grid7_food3_nolevels",
                            "grid7_food3_levels",
                            "grid12_food6_nolevels",
                            "grid12_food6_levels",
                        ],
                        help="Load a processed LBF padded dataset by config name")
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help="Validation fraction for --lbf_config episode split")
    parser.add_argument('--exclude_uncertain', action='store_true',
                        help="Drop LBF episodes with uncertain step-0 reconstruction")
    parser.add_argument('--invalid_action_mode', choices=["drop", "noop"],
                        default="drop",
                        help="How to handle expert actions marked unavailable")
    parser.add_argument('--no_mask_unavailable_actions',
                        dest='mask_unavailable_actions',
                        action='store_false',
                        help="Do not mask unavailable actions in BC loss/accuracy")
    parser.set_defaults(mask_unavailable_actions=True)
    parser.add_argument('--output', required=True,
                        help="Output path for .safetensors weights")
    parser.add_argument('--config', type=str, default=None,
                        help="YAML config with architecture defaults "
                             "(agents/bc/configs/*.yaml)")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preprocess_dim', type=int, default=None)
    parser.add_argument('--lstm_dim', type=int, default=None)
    parser.add_argument('--postprocess_dim', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout rate (0.0 = no dropout)")
    train(parser.parse_args())
