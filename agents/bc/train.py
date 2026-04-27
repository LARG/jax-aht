import argparse
import os
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from agents.bc.bc_lstm import BCLSTMConfig, BCLSTMNetwork, compute_bc_loss, BCLSTMAgent

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

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

def train(args):
    rng = jax.random.PRNGKey(args.seed)

    defaults = {}
    if args.config:
        defaults = load_config(args.config)
        print(f"[train_bc] loaded config from {args.config}")

    (train_obs, train_actions, train_avail, train_mask,
     val_obs, val_actions, val_avail, val_mask) = load_data(args.data_dir)

    obs_dim = int(train_obs.shape[-1])
    action_dim = int(train_avail.shape[-1])
    print(f"[train_bc] obs_dim={obs_dim} action_dim={action_dim}")
    print(f"[train_bc] train: {train_obs.shape[0]} episodes, "
          f"val: {val_obs.shape[0]} episodes")

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
    best_val_acc = 0.0
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
            carry, logits = network.apply(
                {'params': params}, carry, obs[:, t, :])
            preds = jnp.argmax(logits, axis=-1)
            correct = (preds == actions[:, t]) & mask[:, t]
            valid = mask[:, t]
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
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    config_path = args.output.replace('.safetensors', '.yaml')
    with open(config_path, 'w') as f:
        yaml.dump({
            'obs_dim': config.obs_dim,
            'action_dim': config.action_dim,
            'preprocess_dim': config.preprocess_dim,
            'lstm_dim': config.lstm_dim,
            'postprocess_dim': config.postprocess_dim,
            'val_accuracy': float(best_val_acc),
            'data_dir': args.data_dir,
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
    parser.add_argument('--data_dir', required=True,
                        help="Directory with {train,val}_{obs,actions,avail,mask}.safetensors")
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
