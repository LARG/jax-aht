"""
Shared helpers for the benchmark study setup scripts.

Provides:
  - EASY_TASKS / HARD_TASKS: canonical task difficulty splits
  - TIMESTEP_TASK_MAP: source task -> target tasks for timestep transfer
  - resolve_algo_config(): Hydra-style defaults-chain config resolution
  - format_value() / format_timesteps(): YAML/CLI value formatting
  - compute_total_timesteps(): total training timestep budget from a resolved config
"""

import math
from pathlib import Path
import yaml


# ---------------------------------------------------------------------------
# Task difficulty classification
# ---------------------------------------------------------------------------

# Easy tasks: faster to train, lower complexity.
EASY_TASKS: list[str] = [
    "lbf",
    "overcooked-v1/cramped_room",
    "overcooked-v1/asymm_advantages",
]

# Hard tasks: slower to train, higher complexity.
HARD_TASKS: list[str] = [
    "overcooked-v1/coord_ring",
    "overcooked-v1/counter_circuit",
    "overcooked-v1/forced_coord",
]

# All benchmark tasks, easy then hard.
ALL_TASKS: list[str] = EASY_TASKS + HARD_TASKS

# Mapping from the canonical "source" task whose timestep settings should be
# transferred to each other task in the same difficulty tier.
#   source task -> [target tasks]
TIMESTEP_TASK_MAP: dict[str, list[str]] = {
    # Easy tier: LBF is the source.
    "lbf": [t for t in EASY_TASKS if t != "lbf"],
    # Hard tier: coord_ring is the source.
    "overcooked-v1/coord_ring": [t for t in HARD_TASKS if t != "overcooked-v1/coord_ring"],
}


# ---------------------------------------------------------------------------
# Config resolution (Hydra defaults chain)
# ---------------------------------------------------------------------------

def resolve_algo_config(config_path: Path, algo_configs_root: Path) -> dict:
    """Load an algorithm config with its Hydra defaults chain resolved (shallow merge).

    The @package directive is a comment and is ignored by yaml.safe_load.
    Later entries (including _self_) override earlier ones.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", [])
    merged: dict = {}

    for entry in defaults:
        if entry == "_self_":
            merged.update({k: v for k, v in raw.items() if k != "defaults"})
        else:
            # Hydra supports absolute package-rooted paths starting with '/'.
            # Strip the leading '/' so the path resolves relative to
            # algo_configs_root (e.g. "/algorithm/ppo_br/_base_" -> "ppo_br/_base_").
            entry_rel = entry.lstrip("/")
            if entry_rel.startswith("algorithm/"):
                entry_rel = entry_rel[len("algorithm/"):]
            ref = algo_configs_root / f"{entry_rel}.yaml"
            if ref.exists():
                merged.update(resolve_algo_config(ref, algo_configs_root))
            else:
                print(f"  WARNING: referenced config not found: {ref}")

    if not defaults:
        # No defaults block — use the file's own keys directly.
        merged = {k: v for k, v in raw.items() if k != "defaults"}

    return merged


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_value(val) -> str:
    """Format a hyperparameter value for YAML output.

    - booleans  -> true / false
    - integers  -> plain integer string
    - floats that are whole numbers and < 1e5 -> integer string
    - floats < 0.01 or >= 1e5 -> scientific notation (e.g. 1e-3, 1e8)
    - other floats -> :g format (e.g. 0.03, 0.5)
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e5:
            return str(int(val))
        if abs(val) < 0.01 or abs(val) >= 1e5:
            exp = math.floor(math.log10(abs(val)))
            mantissa = round(val / (10 ** exp), 10)
            if mantissa == int(mantissa):
                return f"{int(mantissa)}e{exp}"
            return f"{mantissa:.10g}e{exp}"
        return f"{val:g}"
    return str(val)


def format_timesteps(val: float) -> str:
    """Format a timestep value for use in CLI args.

    Values in [0.1, 100] are written as plain numbers (e.g. 9, 50, 0.5).
    Values outside that range use scientific notation (e.g. 1e8, 4.5e7, 6e6).
    """
    if val == 0:
        return "0"
    if 0.1 <= abs(val) <= 100:
        return str(int(val)) if val == int(val) else f"{val:g}"
    exp = math.floor(math.log10(abs(val)))
    mantissa = round(val / (10 ** exp), 10)
    if mantissa == int(mantissa):
        return f"{int(mantissa)}e{exp}"
    return f"{mantissa:.10g}e{exp}"


# ---------------------------------------------------------------------------
# Total timestep computation
# ---------------------------------------------------------------------------

# Keys that, if present in a resolved config, contribute the main timestep budget.
TIMESTEP_KEYS: list[str] = [
    "TOTAL_TIMESTEPS",
    "TOTAL_TIMESTEPS_PER_ITERATION",
    # ROTATE / open_ended_minimax specific
    "TIMESTEPS_PER_ITER_PARTNER",
    "TIMESTEPS_PER_ITER_EGO",
    "NUM_OPEN_ENDED_ITERS",
]


def compute_total_timesteps(config: dict, algo: str | None = None) -> float | None:
    """Compute the total number of environment timesteps for a resolved algorithm config.

    The formula depends on the algorithm family:

    Ego-agent training (e.g. ppo_ego, ppo_br, liam_ego, meliba_ego):
        TOTAL_TIMESTEPS

    Teammate generation / brdiv, lbrdiv:
        TOTAL_TIMESTEPS + ego_train_algorithm.TOTAL_TIMESTEPS

    Teammate generation / fcp:
        TOTAL_TIMESTEPS * PARTNER_POP_SIZE + ego_train_algorithm.TOTAL_TIMESTEPS
        (TOTAL_TIMESTEPS is per-partner; detected via ALG: fcp in the resolved config)

    Teammate generation / comedi:
        TOTAL_TIMESTEPS_PER_ITERATION * PARTNER_POP_SIZE + ego_train_algorithm.TOTAL_TIMESTEPS

    Open-ended training / COLE:
        PARTNER_POP_SIZE * TOTAL_TIMESTEPS_PER_ITERATION

    Open-ended training / ROTATE or open_ended_minimax:
        (TIMESTEPS_PER_ITER_PARTNER + TIMESTEPS_PER_ITER_EGO) * NUM_OPEN_ENDED_ITERS

    Open-ended training / paired:
        TOTAL_TIMESTEPS

    Parameters
    ----------
    config:
        Fully resolved algorithm config dict (from resolve_algo_config()).
    algo:
        Algorithm name (used to disambiguate COLE vs. other uses of
        TOTAL_TIMESTEPS_PER_ITERATION).  If None, the function infers the
        formula from the keys present in config.

    Returns
    -------
    Total timestep count as a float, or None if the formula cannot be applied.
    """
    ego_cfg = config.get("ego_train_algorithm", {}) or {}
    ego_ts = float(ego_cfg["TOTAL_TIMESTEPS"]) if "TOTAL_TIMESTEPS" in ego_cfg else 0.0

    # ---- ROTATE / open_ended_minimax ----
    if "TIMESTEPS_PER_ITER_PARTNER" in config and "TIMESTEPS_PER_ITER_EGO" in config:
        partner_ts = float(config["TIMESTEPS_PER_ITER_PARTNER"])
        ego_iter_ts = float(config["TIMESTEPS_PER_ITER_EGO"])
        num_iters = float(config.get("NUM_OPEN_ENDED_ITERS", 1))
        return (partner_ts + ego_iter_ts) * num_iters

    # ---- COLE (open_ended_training) ----
    if "TOTAL_TIMESTEPS_PER_ITERATION" in config and "PARTNER_POP_SIZE" in config:
        # Disambiguate: COLE uses pop * ts_per_iter; teammate_generation/comedi
        # does not multiply by pop_size.
        # We check ego_train_algorithm to detect teammate_generation algorithms.
        if ego_ts > 0:
            # Teammate generation / comedi: each partner runs TOTAL_TIMESTEPS_PER_ITERATION
            # timesteps of partner training, then ego training is added on top.
            pop_size = float(config.get("PARTNER_POP_SIZE", 1))
            total = float(config["TOTAL_TIMESTEPS_PER_ITERATION"]) * pop_size + ego_ts
            return total
        else:
            # Open-ended COLE: each partner runs a full TOTAL_TIMESTEPS_PER_ITERATION.
            pop_size = float(config["PARTNER_POP_SIZE"])
            ts_per_iter = float(config["TOTAL_TIMESTEPS_PER_ITERATION"])
            return pop_size * ts_per_iter

    # ---- TOTAL_TIMESTEPS (ego / teammate_gen / paired) ----
    if "TOTAL_TIMESTEPS" in config:
        total = float(config["TOTAL_TIMESTEPS"])
        if ego_ts > 0:
            # FCP trains a separate policy for each partner, so partner training
            # cost scales with population size.  ALG: fcp is set in _base_.yaml.
            if config.get("ALG") == "fcp":
                pop_size = float(config.get("PARTNER_POP_SIZE", 1))
                total = total * pop_size
            total += ego_ts
        return total

    return None


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------

def format_human(n: float) -> str:
    """Format a timestep count as a human-readable string (K / M / B)."""
    if n >= 1e9:
        return f"{n / 1e9:.3g}B"
    if n >= 1e6:
        return f"{n / 1e6:.3g}M"
    if n >= 1e3:
        return f"{n / 1e3:.3g}K"
    return str(int(n))


def parse_human_timesteps(s: str) -> float:
    """Parse a timestep string with optional K / M / B suffix.

    Examples: '130M' -> 1.3e8, '1.3B' -> 1.3e9, '500K' -> 5e5, '7000000' -> 7e6
    """
    s = s.strip()
    suffixes = {"k": 1e3, "m": 1e6, "b": 1e9}
    if s and s[-1].lower() in suffixes:
        return float(s[:-1]) * suffixes[s[-1].lower()]
    return float(s)


# ---------------------------------------------------------------------------
# Target parameter computation
# ---------------------------------------------------------------------------

def round_sig(x: float, sig: int = 3) -> float:
    """Round x to `sig` significant figures."""
    if x == 0:
        return 0.0
    d = math.ceil(math.log10(abs(x)))
    factor = 10 ** (sig - d)
    return round(x * factor) / factor


def compute_target_params(config: dict, target_total: float) -> dict[str, float | int]:
    """Return the minimal parameter changes to reach target_total timesteps.

    Priority rules:
      - ROTATE / open_ended_minimax : adjust NUM_OPEN_ENDED_ITERS and timesteps_per_iteration (integer)
      - FCP                         : adjust PARTNER_POP_SIZE (integer)
      - COLE                        : adjust TOTAL_TIMESTEPS_PER_ITERATION
      - CoMeDi                      : adjust TOTAL_TIMESTEPS_PER_ITERATION and PARTNER_POP_SIZE 
      - All others                  : adjust TOTAL_TIMESTEPS

    ego_train_algorithm.TOTAL_TIMESTEPS is always kept fixed.

    Returns a dict {param_name: new_python_value} (float or int).
    Returns {} if the target cannot be achieved (e.g. target < ego budget).
    """
    ego_cfg = config.get("ego_train_algorithm", {}) or {}
    ego_ts = float(ego_cfg.get("TOTAL_TIMESTEPS", 0))

    # ---- ROTATE / open_ended_minimax: scale both ts-per-iter and num-iters ----
    if "TIMESTEPS_PER_ITER_PARTNER" in config and "TIMESTEPS_PER_ITER_EGO" in config:
        partner_ts = float(config["TIMESTEPS_PER_ITER_PARTNER"])
        ego_iter_ts = float(config["TIMESTEPS_PER_ITER_EGO"])
        num_iters = float(config.get("NUM_OPEN_ENDED_ITERS", 1))
        current = (partner_ts + ego_iter_ts) * num_iters
        scale = target_total / current if current > 0 else 1.0
        sqrt_s = scale ** 0.5
        new_iters = max(1, round(num_iters * sqrt_s))
        return {
            "TIMESTEPS_PER_ITER_PARTNER": round_sig(partner_ts * sqrt_s),
            "TIMESTEPS_PER_ITER_EGO": round_sig(ego_iter_ts * sqrt_s),
            "NUM_OPEN_ENDED_ITERS": new_iters,
        }

    # ---- COLE (no ego budget): adjust TOTAL_TIMESTEPS_PER_ITERATION ----
    if "TOTAL_TIMESTEPS_PER_ITERATION" in config and "PARTNER_POP_SIZE" in config and ego_ts == 0:
        pop_size = float(config["PARTNER_POP_SIZE"])
        return {"TOTAL_TIMESTEPS_PER_ITERATION": round_sig(target_total / pop_size)}

    # ---- comedi (has ego budget): scale both TOTAL_TIMESTEPS_PER_ITERATION and PARTNER_POP_SIZE ----
    if "TOTAL_TIMESTEPS_PER_ITERATION" in config and ego_ts > 0:
        remaining = target_total - ego_ts
        if remaining <= 0:
            return {}
        pop_size = float(config.get("PARTNER_POP_SIZE", 1))
        current_partner = float(config["TOTAL_TIMESTEPS_PER_ITERATION"]) * pop_size
        scale = remaining / current_partner if current_partner > 0 else 1.0
        sqrt_s = scale ** 0.5
        new_pop = max(1, round(pop_size * sqrt_s))
        return {
            "TOTAL_TIMESTEPS_PER_ITERATION": round_sig(float(config["TOTAL_TIMESTEPS_PER_ITERATION"]) * sqrt_s),
            "PARTNER_POP_SIZE": new_pop,
        }

    # ---- FCP: adjust PARTNER_POP_SIZE ----
    if config.get("ALG") == "fcp":
        ts_per_partner = float(config["TOTAL_TIMESTEPS"])
        remaining = target_total - ego_ts
        if remaining <= 0 or ts_per_partner <= 0:
            return {}
        return {"PARTNER_POP_SIZE": max(1, round(remaining / ts_per_partner))}

    # ---- Default: adjust TOTAL_TIMESTEPS ----
    if "TOTAL_TIMESTEPS" in config:
        new_ts = target_total - ego_ts
        if new_ts <= 0:
            return {}
        return {"TOTAL_TIMESTEPS": new_ts}

    return {}
