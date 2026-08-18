'''
Utilities for live (per-update-step) metric logging from inside jax.lax.scan loops.
'''

import numpy as np


def make_live_log_callback(pbar, num_parallel: int, logger=None,
                           loss_key_map=None, returns_prefix: str = "Train/"):
    '''Return a stateful io_callback that advances `pbar` and live-logs per-update metrics.

    It is called once per io_callback firing (i.e. num_parallel times per logical update
    step, where num_parallel = NUM_SEEDS * PARTNER_POP_SIZE for partner training or
    NUM_SEEDS for ego training). Once per logical step it:
      - averages each scalar metric over the num_parallel firings (i.e. over the
        seed / population vmap axes, matching the end-of-run aggregation), and
      - logs it under the end-of-run naming scheme with a monotonically increasing
        global train_step (which equals iter_idx * num_updates + local_step by
        construction, so curves stay continuous across open-ended iterations), and
      - advances `pbar` by one.

    Args:
        pbar:           tqdm progress bar, advanced once per logical update step.
        num_parallel:   number of io_callback firings per logical update step.
        logger:         logger with log_item()/commit(); if None, only the bar advances
                        (i.e. it acts as a progress-bar-only callback).
        loss_key_map:   dict mapping raw metric keys -> fully-qualified wandb tags, e.g.
                        {"value_loss_br": "Losses/BRValLoss"}. Keys absent from the map
                        (and not internal) are treated as rollout-return stats and logged
                        under returns_prefix.
        returns_prefix: prefix for return stats, e.g. "Train/Conf-Against-Ego_" or "Train/Ego/".
    '''
    loss_key_map = loss_key_map or {}
    _INTERNAL = ("update_steps", "returned_episode")
    state = {"count": 0, "accum": {}, "step": 0}

    def callback(*args):
        # Accumulate this firing's scalars (one vmap element) for later averaging.
        if logger is not None and args:
            for stat_name, val in args[0].items():
                if stat_name in _INTERNAL:
                    continue
                state["accum"][stat_name] = state["accum"].get(stat_name, 0.0) + float(np.asarray(val))
        state["count"] += 1
        if state["count"] >= num_parallel:
            if logger is not None and state["accum"]:
                step = state["step"]
                for stat_name, total in state["accum"].items():
                    tag = loss_key_map.get(stat_name, f"{returns_prefix}{stat_name}")
                    logger.log_item(tag, total / num_parallel, train_step=step, commit=False)
                logger.commit()
            state["accum"] = {}
            state["count"] = 0
            state["step"] += 1
            pbar.update(1)

    return callback
