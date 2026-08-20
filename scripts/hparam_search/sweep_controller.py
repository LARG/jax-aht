#!/usr/bin/env python
'''Custom locally-controlled wandb sweep driver.

Seeds a sweep with one run using the env's *current Hydra defaults*, then hands the
remainder of the sweep over to the search method declared in the sweep yaml (random,
for all the ROTATE sweeps). The defaults are read out of the Hydra configs, not
hardcoded here. See docs/wandb_sweep_usage.md 

Requires the `wandb[sweeps]` extra, which is not installed by default.

Usage
-----
    conda activate <env-name>

    # 1. create the sweep (locally controlled)
    PYTHONPATH=. python scripts/hparam_search/sweep_controller.py create \
        open_ended_training/param_sweep/rotate/lbf/lbf_12x12_param_sweep.yml

    # 2. run the controller -- MUST be in a screen/tmux, since nothing is scheduled
    #    while the controller is down. `create` prints the exact line to use.
    screen -S sweepctl
    PYTHONPATH=. python scripts/hparam_search/sweep_controller.py run <sweep_id>
    # detach with C-a d; reattach with `screen -r sweepctl`

    # 3. launch agents as usual, one per node
    bash scripts/hparam_search/run_hparam_sweep.sh <sweep_id>

'''
import argparse
import copy
import os
import sys
import time

import yaml

# scripts/hparam_search/sweep_controller.py -> repo root is three levels up.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_DIR = os.path.join(REPO_ROOT, "open_ended_training", "configs")
CONFIG_NAME = "base_config_oel"

DEFAULT_ENTITY = "aht-project"
DEFAULT_PROJECT = "aht-parameter-sweep"

# Sweep states in which the controller should keep scheduling. done() in wandb 0.19.9
# uses exactly this set, which is why a PAUSED sweep makes its loop exit.
ACTIVE_STATES = {"PENDING", "RUNNING", "PREEMPTING"}
# Paused: not finished, just suspended. Sleep and keep polling.
PAUSED_STATES = {"PAUSED"}
# Genuinely over. Exit.
TERMINAL_STATES = {"FINISHED", "CANCELED", "CANCELLED", "CRASHED", "KILLED"}


def _require_sweeps():
    '''Import the `sweeps` package with an actionable error if the extra is missing.'''
    try:
        import sweeps
    except ImportError:
        sys.exit(
            "wandb[sweeps] is not installed in this environment, so the local sweep "
            "controller cannot run.\n"
            "Install it with:\n"
            "    pip install 'wandb[sweeps]==0.19.9' 'narwhals==1.33.0'\n"
            "Both pins matter -- see docs/wandb_sweep_usage.md for why."
        )
    return sweeps


# --------------------------------------------------------------------------------------
# resolving the per-env defaults out of the Hydra configs
# --------------------------------------------------------------------------------------

def hydra_overrides_from_command(sweep_config):
    '''Pull the fixed Hydra overrides out of a sweep config's `command` block.

    Keeps `task=...`, `algorithm=...` and any other `key=value` entry, drops the wandb
    macros (`${env}`, `${program}`, `${args_no_hyphens}`) and the interpreter/program.
    '''
    overrides = []
    for entry in sweep_config.get("command", []):
        if not isinstance(entry, str):
            continue
        if entry.startswith("$") or entry.startswith("-"):
            continue
        if "=" not in entry:
            continue
        overrides.append(entry)
    return overrides


def resolve_defaults(sweep_config):
    '''Compose the Hydra config this sweep trains with, and read each swept parameter.

    Returns {param_name: default_value} for every parameter in the sweep, where the
    parameter name is used as a dotted path into the composed config (e.g.
    `algorithm.LAMBDA_1`).
    '''
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = hydra_overrides_from_command(sweep_config)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    defaults = {}
    for name in sweep_config.get("parameters", {}):
        value = OmegaConf.select(cfg, name)
        if value is None:
            sys.exit(
                f"Swept parameter {name!r} does not resolve in the Hydra config composed "
                f"with overrides {overrides}. Cannot determine its default."
            )
        defaults[name] = value
    return defaults


def _num(value):
    '''Best-effort float conversion; None if the value is not numeric.'''
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values_equal(a, b, rel_tol=1e-9):
    '''Compare two config values, numerically when possible.'''
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= rel_tol * max(1.0, abs(na), abs(nb))
    return a == b


def _normalize(value):
    '''Turn a numeric-looking string into an actual number.

    PyYAML follows YAML 1.1, where scientific notation is only a float if it has both a
    decimal point and a signed exponent -- so `LR: values: [1e-5, 5e-5, 1e-4, 5e-4]`
    parses as a list of *strings*, while `0.001` in the same list parses as a float.
    This only matters for the values this script schedules directly. Anything the wandb
    backend creates -- every cloud-controlled sweep run, and the runs the backend creates
    from this controller's search suggestions -- is coerced to a float server-side, so
    normalising here just makes the seeded run match the rest of the sweep.
    '''
    if isinstance(value, str):
        number = _num(value)
        if number is not None:
            return number
    return value


def snap_to_grid(defaults, sweep_config):
    '''Replace each default with the equal entry from its sweep grid, where there is one.

    The grids hold ints (`LAMBDA_1: [0, 1, ... 10]`) while the Hydra defaults are floats
    (`LAMBDA_1: 7.0`). Same number, different literal -- scheduling the grid's own
    literal keeps the seeded run indistinguishable from a searched one in the UI, and
    keeps the restart-safety check comparing like with like.
    '''
    snapped = {}
    for name, default in defaults.items():
        spec = sweep_config["parameters"][name]
        grid = spec.get("values")
        if grid is None:
            snapped[name] = _normalize(default)
            continue
        match = [v for v in grid if values_equal(v, default)]
        if match:
            snapped[name] = _normalize(match[0])
        else:
            print(
                f"  WARNING: default {name}={default!r} is not in this sweep's grid "
                f"{grid}. Seeding with the default anyway."
            )
            snapped[name] = _normalize(default)
    return snapped


# --------------------------------------------------------------------------------------
# reading swept parameters back off an existing run
# --------------------------------------------------------------------------------------

def _unwrap(node):
    '''wandb stores config entries as {"value": ..., "desc": ...}; unwrap one level.'''
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def run_param_value(run_config, dotted_name):
    '''Read a swept parameter out of a run's logged config.

    Handles both shapes a Hydra-style dotted sweep parameter can take: a literal flat
    key (`"algorithm.LAMBDA_1"`, how the sweep injects it) and a nested path
    (`config["algorithm"]["LAMBDA_1"]`, how run.py logs the composed config).
    '''
    if dotted_name in run_config:
        return _unwrap(run_config[dotted_name])

    node = run_config
    for part in dotted_name.split("."):
        node = _unwrap(node)
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return _unwrap(node)


def run_matches(run, defaults):
    '''True if every swept parameter on `run` equals the default.'''
    config = run.config or {}
    for name, default in defaults.items():
        value = run_param_value(config, name)
        if value is None or not values_equal(value, default):
            return False
    return True


class _ApiRun:
    '''Minimal stand-in exposing the attributes run_matches() and callers need.'''

    def __init__(self, run):
        self.name = run.id
        self.state = run.state
        self.config = run.config


def all_sweep_runs(entity, project, sweep_id, fallback):
    '''Every run in the sweep, via the public API.

    _WandbController reads its run list from the sweep GraphQL query, which returns only
    a recent window -- on a sweep with 44 runs it reported 10. That is harmless for random
    search, but it silently breaks the "has the default already been run?" check below,
    which then re-seeds a duplicate on every restart. Falls back to the controller's own
    list if the API is unreachable.
    '''
    try:
        import wandb
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        return [_ApiRun(r) for r in sweep.runs]
    except Exception as exc:
        print(f"  WARNING: could not list the sweep's runs via the API ({exc}); "
              "falling back to the controller's partial view, which may re-seed.")
        return fallback or []


def find_default_run(runs, defaults):
    '''Return the first existing run whose swept params match the defaults, if any.'''
    for run in runs or []:
        if run_matches(run, defaults):
            return run
    return None


# --------------------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------------------

def cmd_create(args):
    import wandb

    with open(args.sweep_yaml) as f:
        config = yaml.safe_load(f)

    # Inject the local-controller flag into an in-memory copy only. The committed yaml
    # stays cloud-controlled so `wandb sweep <yaml>` launches are unaffected.
    config = copy.deepcopy(config)
    existing = config.get("controller", {}).get("type")
    if existing not in (None, "local"):
        sys.exit(f"{args.sweep_yaml} already declares controller.type={existing!r}.")
    config.setdefault("controller", {})["type"] = "local"

    # The yaml's own entity/project win over the module defaults; --entity/--project
    # override both. (Getting this order wrong silently creates the sweep in
    # aht-parameter-sweep even when the yaml names a scratch project.)
    entity = args.entity or config.get("entity") or DEFAULT_ENTITY
    project = args.project or config.get("project") or DEFAULT_PROJECT

    # Resolve the defaults now, so a broken command block fails here and not after the
    # sweep has been created.
    defaults = snap_to_grid(resolve_defaults(config), config)
    print(f"Hydra defaults that will seed this sweep ({args.sweep_yaml}):")
    for name, value in defaults.items():
        print(f"  {name} = {value!r}")

    sweep_id = wandb.sweep(config, entity=entity, project=project)
    print(f"\nCreated locally-controlled sweep: {entity}/{project}/{sweep_id}")
    qualifier = ("" if (entity, project) == (DEFAULT_ENTITY, DEFAULT_PROJECT)
                 else f" --entity {entity} --project {project}")
    print("Next, in a screen:")
    print(f"  PYTHONPATH=. python scripts/hparam_search/sweep_controller.py run {sweep_id}{qualifier}")
    print("Then, run the param sweep on a compute node:")
    print(f"  bash scripts/hparam_search/run_hparam_sweep.sh {sweep_id}")
    return sweep_id


def split_sweep_id(sweep_id, entity, project):
    '''Accept either a bare sweep id or a fully qualified entity/project/id.

    Returns (entity, project, bare_id). They have to stay separate: unlike `wandb agent`,
    `wandb.controller()` does not parse a qualified id -- it passes the string straight
    to InternalApi.sweep(), which then fails with "name required for project query"
    because it reads entity/project from the environment instead.
    '''
    if sweep_id.count("/") == 2:
        return tuple(sweep_id.split("/"))
    if "/" in sweep_id:
        sys.exit(f"Malformed sweep id {sweep_id!r}: use <id> or <entity>/<project>/<id>.")
    return entity, project, sweep_id


def sweep_state(tuner):
    return (tuner._sweep_obj or {}).get("state", "").upper()


def outstanding_entries(tuner):
    '''The suggestions the controller has written but the backend has not consumed.'''
    return (tuner._controller or {}).get("schedule") or []


def describe_pending(tuner, schedule_id):
    '''Best-effort explanation of what an outstanding suggestion is waiting for.'''
    scheduled = (tuner._scheduler or {}).get("scheduled") or []
    entry = next((s for s in scheduled if s.get("id") == schedule_id), None)
    if entry is None:
        return "the backend has not created a run for it yet (no agent has picked it up)"
    runid = entry.get("runid")
    run = (tuner._sweep_runs_map or {}).get(runid)
    if run is None:
        return f"run {runid} is not in the sweep's run list (deleted, or never created)"
    return (f"run {runid} is state={run.state!r} with "
            f"{'no' if not run.summary_metrics else 'some'} summary metrics")


def is_wedged(tuner, schedule_id):
    '''True only for a suggestion that can never clear on its own.

    An entry the backend has not turned into a run yet is *not* wedged -- it is simply
    waiting for a free agent, which is the normal steady state once the pool is full.
    The wedged case is an entry whose run exists but will never report: it has vanished,
    or it is stuck in `pending` with no summary metrics because the process died before
    reaching wandb.init.
    '''
    scheduled = (tuner._scheduler or {}).get("scheduled") or []
    entry = next((s for s in scheduled if s.get("id") == schedule_id), None)
    if entry is None:
        return False
    run = (tuner._sweep_runs_map or {}).get(entry.get("runid"))
    if run is None:
        return True
    return run.state == "pending" and not run.summary_metrics


def drop_entries(tuner, stale_ids):
    '''Drop wedged suggestions so the pool can refill.

    wandb only clears a suggestion once _parse_scheduled() sees its run leave the
    `pending` state or report a summary metric. A run that dies before either -- crashing
    during init, OOM, an evicted node -- leaves the entry in place forever, permanently
    consuming one of the pool's slots. With the stock one-slot pool that wedges the sweep
    outright; with a larger pool it silently shrinks the pool instead.

    Restarting the controller clears these only because _start_if_not_started() resets
    self._controller to {} and syncs it. This does the same thing in-process.

    Dropping a suggestion never loses work: an entry the backend never consumed is simply
    replaced on the next step, and the search re-derives its next suggestion from the
    sweep's run history regardless.
    '''
    kept = [e for e in outstanding_entries(tuner) if e.get("id") not in stale_ids]
    tuner._controller["schedule"] = kept
    tuner._sweep_object_sync_to_backend()


def schedule_batch(tuner, target):
    '''Top the outstanding-suggestion pool up to `target` entries.

    _WandbController.schedule() hard-caps the pool at one entry ("only schedule one run
    at a time (for now)"), so with several agents per sweep they serialise: agent N+1 sits
    idle until agent N's run reaches wandb.init and flips out of `pending`, which for a
    jax-aht run is the better part of a minute of imports and Hydra composition. That is
    the multi-minute wait new agents see.

    The one-at-a-time rule is documented as an implementation detail, not part of the
    protocol -- both controller.schedule and scheduler.scheduled are lists, and
    _parse_scheduled() already matches entries by id -- so writing several entries and
    syncing once is well-formed. Returns the number of new suggestions added.

    Only safe for methods that sample independently. search() reads the same run history
    for every call within one step, so grid/bayes would hand back duplicate suggestions;
    those stay capped at one per step.
    '''
    from wandb.wandb_controller import _id_generator

    method = (tuner.sweep_config or {}).get("method", "random")
    if method != "random":
        target = 1

    entries = list(outstanding_entries(tuner))
    added = 0
    while len(entries) < target:
        suggestion = tuner.search()
        if suggestion is None:
            # schedule(None) is how the backend is told the search space is exhausted.
            entries.append({"id": _id_generator(), "data": {"args": None}})
            added += 1
            break
        entries.append(
            {"id": _id_generator(), "data": {"args": suggestion.config}}
        )
        added += 1

    if added:
        tuner._controller["schedule"] = entries
        tuner._sweep_object_sync_to_backend()
    return added


def seed_defaults(tuner, defaults, force=False, runs=None):
    '''Schedule one run at the Hydra defaults, unless one already exists.

    Returns True if a run was scheduled.
    '''
    sweeps = _require_sweeps()

    existing = find_default_run(tuner._sweep_runs if runs is None else runs, defaults)
    if existing is not None and not force:
        print(f"Default-hyperparameter run already present ({existing.name}, "
              f"state={existing.state}); not seeding again.")
        return False

    # schedule() is a no-op while an earlier suggestion is still pending, which would
    # silently drop the seed. _step() has just drained anything already picked up, so a
    # leftover entry here means the backend has not consumed the previous suggestion yet.
    if tuner._controller.get("schedule"):
        print("A suggestion is still pending with the backend; deferring the seed.")
        return False

    run = sweeps.SweepRun(config={k: {"value": v} for k, v in defaults.items()})
    tuner.schedule(run)
    print("Seeded the sweep with the Hydra defaults:")
    for name, value in defaults.items():
        print(f"  {name} = {value!r}")
    return True


def cmd_run(args):
    import wandb

    _require_sweeps()
    entity, project, sweep_id = split_sweep_id(
        args.sweep_id, args.entity or DEFAULT_ENTITY, args.project or DEFAULT_PROJECT
    )
    print(f"Attaching local controller to {entity}/{project}/{sweep_id}")

    tuner = wandb.controller(sweep_id, entity=entity, project=project)
    # Pulls the sweep object, its config and every existing run off the backend, so all
    # controller state is rebuilt from scratch -- this is what makes restarts safe.
    tuner._step()

    defaults = snap_to_grid(resolve_defaults(tuner.sweep_config), tuner.sweep_config)
    runs = all_sweep_runs(entity, project, sweep_id, tuner._sweep_runs)
    print(f"Sweep has {len(runs)} runs (the controller itself sees "
          f"{len(tuner._sweep_runs or [])}).")
    seed_defaults(tuner, defaults, force=args.force_seed, runs=runs)

    if args.seed_only:
        print("--seed-only given; not entering the scheduling loop.")
        return

    print(f"Entering scheduling loop (poll every {args.poll_interval}s, "
          f"{args.max_pending} suggestions kept outstanding, stuck-suggestion timeout "
          f"{args.schedule_timeout}s). "
          "Ctrl-C to stop; restarting resumes without re-seeding.")
    # id -> monotonic time first seen outstanding. See drop_entries().
    watched = {}
    while True:
        state = sweep_state(tuner)

        if state in TERMINAL_STATES:
            print(f"Sweep state is {state}; controller exiting.")
            return
        if state in PAUSED_STATES:
            print(f"Sweep is {state}; holding (not exiting). Resume it in the UI to "
                  "continue scheduling.")
            time.sleep(args.poll_interval)
            tuner._step()
            continue
        if state not in ACTIVE_STATES:
            print(f"WARNING: unrecognised sweep state {state!r}; continuing to poll "
                  "rather than exiting.")
            time.sleep(args.poll_interval)
            tuner._step()
            continue

        # Body of _WandbController.step(), minus the _step() refresh we drive ourselves.
        # schedule() is a no-op while an earlier suggestion (including the seed) is still
        # pending, so this cannot overwrite the seeded run. schedule(None) is how the
        # backend is told the search space is exhausted, so it is deliberately still
        # called when search() comes back empty.
        # Keep several suggestions queued so an agent that finishes -- or one that has
        # only just started -- finds work waiting instead of sitting idle for a poll
        # cycle plus however long the previous run takes to reach wandb.init.
        schedule_batch(tuner, args.max_pending)

        to_stop = tuner.stopping()
        if to_stop:
            tuner.stop_runs(to_stop)

        tuner.print_status()
        time.sleep(args.poll_interval)
        tuner._step()

        # _step() has just dropped every suggestion the backend consumed. Anything still
        # here is outstanding; an id that stays outstanding too long is wedged.
        now = time.monotonic()
        live = {e.get("id") for e in outstanding_entries(tuner)}
        watched = {k: v for k, v in watched.items() if k in live}
        for schedule_id in live:
            watched.setdefault(schedule_id, now)

        stale = {k for k, seen in watched.items()
                 if now - seen > args.schedule_timeout and is_wedged(tuner, k)}
        if stale:
            for schedule_id in stale:
                waited = int(now - watched[schedule_id])
                print(f"WARNING: suggestion {schedule_id} outstanding for {waited}s: "
                      f"{describe_pending(tuner, schedule_id)}. Dropping it.")
            drop_entries(tuner, stale)
            watched = {k: v for k, v in watched.items() if k not in stale}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    # Default to None, not to DEFAULT_*, so `create` can tell "not specified" from an
    # explicit choice and fall back to the sweep yaml's own entity/project.
    common.add_argument("--entity", default=None,
                        help=f"overrides the sweep yaml (default: {DEFAULT_ENTITY})")
    common.add_argument("--project", default=None,
                        help=f"overrides the sweep yaml (default: {DEFAULT_PROJECT})")

    p_create = sub.add_parser(
        "create", parents=[common],
        help="create a locally-controlled sweep from a committed sweep yaml",
    )
    p_create.add_argument("sweep_yaml")
    p_create.set_defaults(func=cmd_create)

    p_run = sub.add_parser(
        "run", parents=[common],
        help="seed with the Hydra defaults, then drive the sweep",
    )
    p_run.add_argument("sweep_id", help="bare sweep id, or entity/project/id")
    p_run.add_argument("--poll-interval", type=float, default=15.0,
                       help="seconds between controller steps (default: 15)")
    p_run.add_argument("--force-seed", action="store_true",
                       help="schedule the defaults even if a matching run already exists")
    p_run.add_argument("--max-pending", type=int, default=4,
                       help="how many suggestions to keep queued for agents to pick up "
                            "(default: 4). Set it to roughly the number of agents on "
                            "this sweep. wandb's own controller hard-caps this at 1, "
                            "which makes agents wait for each other to start.")
    p_run.add_argument("--schedule-timeout", type=float, default=600.0,
                       help="seconds an unstarted suggestion may stay outstanding before "
                            "the controller clears it (default: 600). Works around a "
                            "wandb bug where a run that crashes before leaving 'pending' "
                            "wedges scheduling permanently.")
    p_run.add_argument("--seed-only", action="store_true",
                       help="seed and exit, without entering the scheduling loop")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
