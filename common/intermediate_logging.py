import numpy as np


def _as_int(value):
    return int(np.asarray(value).item())


def _mean_scalar(value):
    return float(np.asarray(value, dtype=np.float32).mean())


def _context_path(prefix, seed_idx=None, partner_idx=None, population_stage=None):
    parts = [prefix]
    if seed_idx is not None:
        parts.append(f"Seed_{_as_int(seed_idx)}")
    if partner_idx is not None:
        parts.append(f"Partner_{_as_int(partner_idx)}")
    if population_stage is not None:
        parts.append(f"PopulationStage_{_as_int(population_stage)}")
    return "/".join(parts)


def _log_scalars(logger, prefix, scalars, step):
    train_step = _as_int(step)
    for name, value in scalars.items():
        logger.log_item(f"{prefix}/{name}", float(value), train_step=train_step, commit=False)
    logger.commit()
    return np.int32(0)


def extract_masked_episode_stats(metric, metric_names):
    mask = np.asarray(metric["returned_episode"], dtype=bool)
    denom = int(mask.sum())
    stats = {}
    for metric_name in metric_names:
        values = np.asarray(metric[metric_name], dtype=np.float32)
        if denom == 0:
            stats[metric_name] = 0.0
        else:
            stats[metric_name] = float(np.where(mask, values, 0.0).sum() / denom)
    return stats


def log_ippo_intermediate_metrics(logger, metric, metric_names, update_step, seed_idx=None, partner_idx=None):
    prefix = _context_path("Train/Intermediate", seed_idx=seed_idx, partner_idx=partner_idx)
    stats = extract_masked_episode_stats(metric, metric_names)
    return _log_scalars(logger, prefix, stats, update_step)


def compute_sp_xp_returns(pair_returns, pop_size=None):
    pair_returns = np.asarray(pair_returns, dtype=np.float32)
    if pop_size is None:
        pop_size = int(round(np.sqrt(pair_returns.shape[0])))

    cross_product = np.meshgrid(np.arange(pop_size), np.arange(pop_size))
    agent_id_cartesian_product = np.stack([grid.ravel() for grid in cross_product], axis=-1)
    sp_mask = agent_id_cartesian_product[:, 0] == agent_id_cartesian_product[:, 1]

    sp_return = float(pair_returns[sp_mask].mean())
    xp_return = float(pair_returns[~sp_mask].mean()) if np.any(~sp_mask) else sp_return
    return sp_return, xp_return


def log_brdiv_intermediate_metrics(logger, metric, seed_idx):
    step = metric["update_steps"]
    loss_prefix = _context_path("Losses/Intermediate", seed_idx=seed_idx)
    loss_scalars = {
        "ConfPGLoss": _mean_scalar(metric["pg_loss_conf_agent"]),
        "BRPGLoss": _mean_scalar(metric["pg_loss_br_agent"]),
        "ConfValLoss": _mean_scalar(metric["value_loss_conf_agent"]),
        "BRValLoss": _mean_scalar(metric["value_loss_br_agent"]),
        "ConfEntropy": _mean_scalar(metric["entropy_conf"]),
        "BREntropy": _mean_scalar(metric["entropy_br"]),
    }
    _log_scalars(logger, loss_prefix, loss_scalars, step)

    eval_prefix = _context_path("Eval/Intermediate", seed_idx=seed_idx)
    pair_returns = metric["eval_ep_last_info"]["returned_episode_returns"]
    sp_return, xp_return = compute_sp_xp_returns(pair_returns)
    return _log_scalars(
        logger,
        eval_prefix,
        {
            "AvgSPReturnCurve": sp_return,
            "AvgXPReturnCurve": xp_return,
        },
        step,
    )


def log_lbrdiv_intermediate_metrics(logger, metric, seed_idx):
    step = metric["update_steps"]
    loss_prefix = _context_path("Losses/Intermediate", seed_idx=seed_idx)
    loss_scalars = {
        "ConfPGLoss": _mean_scalar(metric["pg_loss_conf_agent"]),
        "BRPGLoss": _mean_scalar(metric["pg_loss_br_agent"]),
        "ConfValLoss": _mean_scalar(metric["value_loss_conf_agent"]),
        "BRValLoss": _mean_scalar(metric["value_loss_br_agent"]),
        "ConfEntropy": _mean_scalar(metric["entropy_conf"]),
        "BREntropy": _mean_scalar(metric["entropy_br"]),
        "AvgLMsHorizontal": _mean_scalar(metric["lms_horizontal"]),
        "AvgLMsVertical": _mean_scalar(metric["lms_vertical"]),
    }
    _log_scalars(logger, loss_prefix, loss_scalars, step)

    eval_prefix = _context_path("Eval/Intermediate", seed_idx=seed_idx)
    pair_returns = metric["eval_ep_last_info"]["returned_episode_returns"]
    sp_return, xp_return = compute_sp_xp_returns(pair_returns)
    return _log_scalars(
        logger,
        eval_prefix,
        {
            "AvgSPReturnCurve": sp_return,
            "AvgXPReturnCurve": xp_return,
        },
        step,
    )


def compute_comedi_xp_return(xp_returns, valid_population_size):
    xp_returns = np.asarray(xp_returns, dtype=np.float32)
    valid_population_size = max(1, min(_as_int(valid_population_size), xp_returns.shape[0]))
    return float(xp_returns[:valid_population_size].mean())


def log_comedi_intermediate_metrics(logger, metric, xp_eval_returns, sp_eval_returns, seed_idx, population_stage):
    step = metric["update_steps"]
    loss_prefix = _context_path(
        "Losses/Intermediate",
        seed_idx=seed_idx,
        population_stage=population_stage,
    )
    loss_scalars = {
        "ConfPGLossSP": _mean_scalar(metric["pg_loss_conf_sp"]),
        "ConfPGLossXP": _mean_scalar(metric["pg_loss_conf_xp"]),
        "ConfPGLossMP": _mean_scalar(metric["pg_loss_conf_mp"]),
        "ConfValLossSP": _mean_scalar(metric["value_loss_conf_sp"]),
        "ConfValLossXP": _mean_scalar(metric["value_loss_conf_xp"]),
        "ConfValLossMP": _mean_scalar(metric["value_loss_conf_mp"]),
        "EntropySP": _mean_scalar(metric["entropy_conf_sp"]),
        "EntropyXP": _mean_scalar(metric["entropy_conf_xp"]),
        "EntropyMP": _mean_scalar(metric["entropy_conf_mp"]),
    }
    _log_scalars(logger, loss_prefix, loss_scalars, step)

    train_prefix = _context_path(
        "Train/Intermediate",
        seed_idx=seed_idx,
        population_stage=population_stage,
    )
    train_scalars = {
        "AverageRewardEgo": _mean_scalar(metric["average_rewards_ego"]),
        "AverageRewardBRSP": _mean_scalar(metric["average_rewards_br_sp"]),
        "AverageRewardBRMP2": _mean_scalar(metric["average_rewards_br_mp2"]),
    }
    _log_scalars(logger, train_prefix, train_scalars, step)

    eval_prefix = _context_path(
        "Eval/Intermediate",
        seed_idx=seed_idx,
        population_stage=population_stage,
    )
    xp_return = compute_comedi_xp_return(
        xp_eval_returns["returned_episode_returns"],
        valid_population_size=population_stage,
    )
    sp_return = _mean_scalar(sp_eval_returns["returned_episode_returns"])
    return _log_scalars(
        logger,
        eval_prefix,
        {
            "AvgSPReturnCurve": sp_return,
            "AvgXPReturnCurve": xp_return,
        },
        step,
    )
