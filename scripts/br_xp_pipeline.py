#!/usr/bin/env python3
"""Run BR -> XP -> plotting pipeline with status + manifest tracking."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from evaluation.generate_heldout_br_config_from_logs import generate_config_from_logs


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def shell_out(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def safe_task_name(task: str) -> str:
    return task.replace("/", "_")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def flatten_tasks(registry: DictConfig) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for _, domain_cfg in registry["domains"].items():
        for task_cfg in domain_cfg["tasks"]:
            tasks.append(
                {
                    "task": str(task_cfg["task"]),
                    "title": str(task_cfg.get("title", task_cfg["task"])),
                }
            )
    return tasks


def latest_timestamp_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_timestamp_dir_with_file(base_dir: Path, filename: str) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted(
        [p for p in base_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / filename).exists():
            return candidate
    return None


@dataclass
class CmdRecord:
    stage: str
    cmd: list[str]
    returncode: int
    started_at: str
    finished_at: str


class PipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.registry_path = (self.repo_root / args.registry).resolve()

        self.registry = OmegaConf.load(self.registry_path)
        if not isinstance(self.registry, DictConfig):
            raise ValueError(f"Invalid registry: {self.registry_path}")

        self.run_id = args.run_id or default_run_id()
        output_root = Path(self.registry["pipeline"]["output_root"])
        self.run_dir = (self.repo_root / output_root / self.run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.status_path = self.run_dir / "status.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.log_path = self.run_dir / "pipeline.log"

        self.command_records: list[CmdRecord] = []
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "pipeline_name": str(self.registry["pipeline"]["name"]),
            "started_at": utc_now(),
            "finished_at": None,
            "repo_root": str(self.repo_root),
            "registry": str(self.registry_path),
            "git": {
                "branch": shell_out(["git", "-C", str(self.repo_root), "rev-parse", "--abbrev-ref", "HEAD"]),
                "commit": shell_out(["git", "-C", str(self.repo_root), "rev-parse", "HEAD"]),
            },
            "inputs": {
                "gpu_list": args.gpu_list,
                "parallel_jobs": args.parallel_jobs,
                "total_timesteps": args.total_timesteps,
                "xp_config": args.xp_config,
                "logs_glob": args.logs_glob,
                "dry_run": args.dry_run,
            },
            "artifacts": {},
            "commands": [],
        }

        self.status: dict[str, Any] = {
            "run_id": self.run_id,
            "state": "running",
            "started_at": self.manifest["started_at"],
            "finished_at": None,
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "log_path": str(self.log_path),
            "manifest_path": str(self.manifest_path),
            "stages": {
                "br": "pending",
                "ingest": "pending",
                "xp": "pending",
                "analysis": "pending",
                "plots": "pending",
            },
            "last_message": "initialized",
        }

        self._write_status("initialized")

    def _write_status(self, message: str) -> None:
        self.status["last_message"] = message
        self.status_path.write_text(json.dumps(self.status, indent=2, sort_keys=True), encoding="utf-8")

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _run_cmd(self, stage: str, cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
        started_at = utc_now()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{started_at}] STAGE={stage} CMD={' '.join(cmd)}\n")

        if self.args.dry_run:
            finished_at = utc_now()
            self.command_records.append(
                CmdRecord(stage=stage, cmd=cmd, returncode=0, started_at=started_at, finished_at=finished_at)
            )
            return

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        with self.log_path.open("a", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )

        finished_at = utc_now()
        self.command_records.append(
            CmdRecord(
                stage=stage,
                cmd=cmd,
                returncode=proc.returncode,
                started_at=started_at,
                finished_at=finished_at,
            )
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed in stage '{stage}' with exit code {proc.returncode}: {' '.join(cmd)}"
            )

    def run(self) -> None:
        try:
            if self.args.skip_br:
                self.status["stages"]["br"] = "skipped"
                self._write_status("skipped BR stage")
            else:
                self.run_br_stage()

            if self.args.skip_ingest:
                self.status["stages"]["ingest"] = "skipped"
                self._write_status("skipped ingest stage")
            else:
                self.run_ingest_stage()

            if self.args.skip_xp:
                self.status["stages"]["xp"] = "skipped"
                self._write_status("skipped XP stage")
            else:
                self.run_xp_stage()

            analysis_script = self.repo_root / "evaluation/analyze_xp_matrix.py"
            if self.args.skip_analysis:
                self.status["stages"]["analysis"] = "skipped"
                self._write_status("skipped analysis stage")
            elif not analysis_script.exists():
                self.status["stages"]["analysis"] = "skipped"
                self._write_status(
                    "skipped analysis stage (evaluation/analyze_xp_matrix.py not present)"
                )
            else:
                self.run_analysis_stage()

            if self.args.skip_plots:
                self.status["stages"]["plots"] = "skipped"
                self._write_status("skipped plots stage")
            else:
                self.run_plot_stage()

            self.status["state"] = "completed"
            self.status["finished_at"] = utc_now()
            self.manifest["finished_at"] = self.status["finished_at"]
            self._finalize_records()
            self._write_manifest()
            self._write_status("pipeline completed")
        except Exception as exc:
            self.status["state"] = "failed"
            self.status["finished_at"] = utc_now()
            self.manifest["finished_at"] = self.status["finished_at"]
            self.manifest["error"] = str(exc)
            self._finalize_records()
            self._write_manifest()
            self._write_status(f"failed: {exc}")
            raise

    def _finalize_records(self) -> None:
        self.manifest["commands"] = [
            {
                "stage": record.stage,
                "cmd": record.cmd,
                "returncode": record.returncode,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
            }
            for record in self.command_records
        ]

    def run_br_stage(self) -> None:
        self.status["stages"]["br"] = "running"
        self._write_status("running BR scripts")

        env = {
            "GPU_LIST": self.args.gpu_list,
            "PARALLEL_JOBS": str(self.args.parallel_jobs),
            "TOTAL_TIMESTEPS": str(self.args.total_timesteps),
        }

        scripts_run: list[str] = []
        for _, domain_cfg in self.registry["domains"].items():
            for script_rel in domain_cfg["br_scripts"]:
                script = str(script_rel)
                self._run_cmd("br", ["bash", script], extra_env=env)
                scripts_run.append(script)

        self.manifest["artifacts"]["br"] = {
            "scripts_run": scripts_run,
            "checkpoint_log_glob": self.args.logs_glob,
        }

        self.status["stages"]["br"] = "completed"
        self._write_status("completed BR scripts")

    def run_ingest_stage(self) -> None:
        self.status["stages"]["ingest"] = "running"
        self._write_status("generating global_heldout_br_generated from logs")

        base_cfg = self.repo_root / "evaluation/configs/global_heldout_br.yaml"
        out_cfg = self.repo_root / "evaluation/configs/global_heldout_br_generated.yaml"
        log_files = sorted(self.repo_root.glob(self.args.logs_glob))

        if self.args.dry_run:
            summary = {
                "base_config": str(base_cfg),
                "output_config": str(out_cfg),
                "log_files": [str(p) for p in log_files],
                "job_records_seen": 0,
                "updated_paths": 0,
                "ignored_not_found": 0,
                "unknown_job_keys": [],
                "missing_expected_job_keys": [],
                "dry_run": True,
            }
        else:
            summary = generate_config_from_logs(
                base_config_path=base_cfg,
                output_config_path=out_cfg,
                log_files=log_files,
            )

        self.manifest["artifacts"]["ingest"] = summary

        self.status["stages"]["ingest"] = "completed"
        self._write_status("generated global_heldout_br_generated")

    def run_xp_stage(self) -> None:
        self.status["stages"]["xp"] = "running"
        self._write_status("running heldout XP evaluations")

        xp_config = self.args.xp_config or str(self.registry["pipeline"]["default_xp_config"])
        label_prefix = str(self.registry["pipeline"]["label_prefix"])
        csv_name = str(self.registry["pipeline"]["xp_csv_name"])

        xp_outputs: list[dict[str, str]] = []
        task_defs = flatten_tasks(self.registry)

        for task_def in task_defs:
            task = task_def["task"]
            label = f"{label_prefix}{safe_task_name(task)}"

            self._run_cmd(
                "xp",
                [
                    "python3",
                    "evaluation/run.py",
                    "--config-name",
                    xp_config,
                    f"task={task}",
                    f"label={label}",
                    "xp_matrix_outputs.save_heatmap=true",
                ],
            )

            xp_root = self.repo_root / "results" / task / "heldout_xp_matrix" / label
            latest_dir = latest_timestamp_dir(xp_root)
            if latest_dir is None:
                raise RuntimeError(f"No XP output directory found for task={task} label={label}")

            csv_path = latest_dir / csv_name
            if not csv_path.exists():
                raise RuntimeError(f"Missing XP CSV: {csv_path}")

            self._run_cmd(
                "xp",
                [
                    "python3",
                    "evaluation/plot_xp_csv_tsne.py",
                    str(latest_dir),
                    "--publication",
                    "--embedding",
                    "cols",
                ],
            )

            tsne_col_plot = latest_dir / f"{csv_path.stem}_cols_tsne.png"
            xp_outputs.append(
                {
                    "task": task,
                    "title": task_def["title"],
                    "label": label,
                    "xp_dir": str(latest_dir),
                    "xp_csv": str(csv_path),
                    "tsne_col_plot": str(tsne_col_plot),
                }
            )

        self.manifest["artifacts"]["xp"] = {
            "xp_config": xp_config,
            "outputs": xp_outputs,
        }

        self.status["stages"]["xp"] = "completed"
        self._write_status("completed heldout XP evaluations")

    def run_analysis_stage(self) -> None:
        self.status["stages"]["analysis"] = "running"
        self._write_status("running teammate quality/diversity analysis")

        xp_artifacts = self.manifest.get("artifacts", {}).get("xp", {}).get("outputs", [])
        if not xp_artifacts:
            label_prefix = str(self.registry["pipeline"]["label_prefix"])
            csv_name = str(self.registry["pipeline"]["xp_csv_name"])
            xp_artifacts = []
            for task_def in flatten_tasks(self.registry):
                task = task_def["task"]
                label = f"{label_prefix}{safe_task_name(task)}"
                xp_root = self.repo_root / "results" / task / "heldout_xp_matrix" / label
                latest_dir = latest_timestamp_dir_with_file(xp_root, csv_name)
                if latest_dir is None:
                    if self.args.allow_missing_xp:
                        continue
                    raise RuntimeError(f"No XP output directory found for task={task} label={label}")
                csv_path = latest_dir / csv_name
                if not csv_path.exists():
                    if self.args.allow_missing_xp:
                        continue
                    raise RuntimeError(f"Missing XP CSV: {csv_path}")
                xp_artifacts.append({"task": task, "title": task_def["title"], "xp_dir": str(latest_dir), "xp_csv": str(csv_path)})

        if not xp_artifacts:
            raise RuntimeError("No XP CSV artifacts available for analysis stage.")

        analysis_outputs: list[dict[str, str]] = []
        for entry in xp_artifacts:
            task = str(entry["task"])
            latest_dir = Path(entry["xp_dir"])
            csv_path = Path(entry["xp_csv"])
            if not csv_path.exists():
                if self.args.allow_missing_xp:
                    continue
                raise RuntimeError(f"Missing XP CSV: {csv_path}")

            heatmap_png = latest_dir / f"{csv_path.stem}_heatmap.png"
            row_tsne_png = latest_dir / f"{csv_path.stem}_rows_tsne.png"
            analysis_dir = latest_dir / f"{csv_path.stem}_analysis"
            summary_json = analysis_dir / "summary.json"

            self._run_cmd(
                "analysis",
                [
                    "python3",
                    "evaluation/plot_xp_csv_heatmap.py",
                    str(csv_path),
                    "--title",
                    f"XP Matrix: {task}",
                    "--out",
                    str(heatmap_png),
                    "--no-annot",
                ],
            )

            self._run_cmd(
                "analysis",
                [
                    "python3",
                    "evaluation/plot_xp_csv_tsne.py",
                    str(latest_dir),
                    "--publication",
                    "--embedding",
                    "rows",
                ],
            )

            self._run_cmd(
                "analysis",
                [
                    "python3",
                    "evaluation/analyze_xp_matrix.py",
                    str(csv_path),
                    "--out-dir",
                    str(analysis_dir),
                ],
            )

            analysis_outputs.append(
                {
                    "task": task,
                    "xp_csv": str(csv_path),
                    "heatmap": str(heatmap_png),
                    "row_tsne": str(row_tsne_png),
                    "analysis_summary": str(summary_json),
                }
            )

        self.manifest["artifacts"]["analysis"] = {
            "outputs": analysis_outputs,
        }

        self.status["stages"]["analysis"] = "completed"
        self._write_status("completed teammate quality/diversity analysis")

    def run_plot_stage(self) -> None:
        self.status["stages"]["plots"] = "running"
        self._write_status("creating publication meta-figure")

        xp_artifacts = self.manifest.get("artifacts", {}).get("xp", {}).get("outputs", [])
        if not xp_artifacts:
            label_prefix = str(self.registry["pipeline"]["label_prefix"])
            csv_name = str(self.registry["pipeline"]["xp_csv_name"])
            xp_artifacts = []
            for task_def in flatten_tasks(self.registry):
                task = task_def["task"]
                label = f"{label_prefix}{safe_task_name(task)}"
                xp_root = self.repo_root / "results" / task / "heldout_xp_matrix" / label
                latest_dir = latest_timestamp_dir_with_file(xp_root, csv_name)
                if latest_dir is None:
                    if self.args.allow_missing_xp:
                        continue
                    raise RuntimeError(f"No XP output directory found for task={task} label={label}")
                csv_path = latest_dir / csv_name
                if not csv_path.exists():
                    if self.args.allow_missing_xp:
                        continue
                    raise RuntimeError(f"Missing XP CSV: {csv_path}")
                xp_artifacts.append({"task": task, "title": task_def["title"], "xp_csv": str(csv_path)})

        if not xp_artifacts:
            raise RuntimeError("No XP CSV artifacts available for plot stage.")

        csv_paths = [entry["xp_csv"] for entry in xp_artifacts]
        titles = [entry["title"] for entry in xp_artifacts]

        meta_cfg = self.registry["meta_figure"]
        out_png = self.repo_root / str(meta_cfg["output_png"])
        out_png.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3",
            "evaluation/plot_tsne_meta_figure.py",
            *csv_paths,
            "--titles",
            *titles,
            "--out",
            str(out_png),
            "--preset",
            str(meta_cfg["preset"]),
            "--density-backdrop",
            str(meta_cfg["density_backdrop"]),
            "--global-xlabel",
            str(meta_cfg["global_xlabel"]),
            "--global-ylabel",
            str(meta_cfg["global_ylabel"]),
        ]

        if bool(meta_cfg.get("density_contours", False)):
            cmd.append("--density-contours")
        if bool(meta_cfg.get("no_subplot_axis_ticks", False)):
            cmd.append("--no-subplot-axis-ticks")
        if bool(meta_cfg.get("hide_subplot_axis_labels", False)):
            cmd.append("--hide-subplot-axis-labels")

        self._run_cmd("plots", cmd)

        self.manifest["artifacts"]["plots"] = {
            "meta_plot": str(out_png),
            "input_csvs": csv_paths,
            "titles": titles,
        }

        self.status["stages"]["plots"] = "completed"
        self._write_status("created publication meta-figure")


def cmd_run(args: argparse.Namespace) -> int:
    runner = PipelineRunner(args)
    runner.run()
    print(f"run_id={runner.run_id}")
    print(f"status={runner.status_path}")
    print(f"manifest={runner.manifest_path}")
    print(f"log={runner.log_path}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    registry_path = (repo_root / args.registry).resolve()
    registry = OmegaConf.load(registry_path)

    run_id = args.run_id or default_run_id()
    output_root = repo_root / str(registry["pipeline"]["output_root"])
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    launch_log = run_dir / "launcher.log"
    launcher_meta = run_dir / "launcher.json"

    cmd: list[str] = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--repo-root",
        str(repo_root),
        "--registry",
        str(args.registry),
        "--run-id",
        run_id,
        "--gpu-list",
        str(args.gpu_list),
        "--parallel-jobs",
        str(args.parallel_jobs),
        "--total-timesteps",
        str(args.total_timesteps),
        "--logs-glob",
        str(args.logs_glob),
    ]

    if args.xp_config is not None:
        cmd.extend(["--xp-config", str(args.xp_config)])
    if args.skip_br:
        cmd.append("--skip-br")
    if args.skip_ingest:
        cmd.append("--skip-ingest")
    if args.skip_xp:
        cmd.append("--skip-xp")
    if args.skip_analysis:
        cmd.append("--skip-analysis")
    if args.skip_plots:
        cmd.append("--skip-plots")
    if args.allow_missing_xp:
        cmd.append("--allow-missing-xp")
    if args.dry_run:
        cmd.append("--dry-run")

    with launch_log.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] launching detached pipeline\n")
        f.write(f"CMD={' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env=os.environ.copy(),
        )

    launcher_meta.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": proc.pid,
                "launched_at": utc_now(),
                "repo_root": str(repo_root),
                "registry": str(registry_path),
                "command": cmd,
                "launch_log": str(launch_log),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"run_id={run_id}")
    print(f"pid={proc.pid}")
    print(f"run_dir={run_dir}")
    print(f"launch_log={launch_log}")
    print(f"launcher_meta={launcher_meta}")
    print("status_cmd=python3 scripts/br_xp_pipeline.py status --run-id " + run_id)
    return 0


def resolve_status_path(repo_root: Path, registry_path: Path, run_id: str | None) -> Path:
    registry = OmegaConf.load(registry_path)
    output_root = repo_root / str(registry["pipeline"]["output_root"])
    if run_id:
        status_path = output_root / run_id / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(f"No status file for run_id={run_id}: {status_path}")
        return status_path

    candidates = sorted(output_root.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No status files under: {output_root}")
    return candidates[0]


def resolve_output_root(repo_root: Path, registry_path: Path) -> Path:
    registry = OmegaConf.load(registry_path)
    return repo_root / str(registry["pipeline"]["output_root"])


def cmd_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    registry_path = (repo_root / args.registry).resolve()
    try:
        status_path = resolve_status_path(repo_root, registry_path, args.run_id)
    except FileNotFoundError:
        if args.run_id:
            output_root = resolve_output_root(repo_root, registry_path)
            launcher_meta = output_root / args.run_id / "launcher.json"
            if launcher_meta.exists():
                launch = json.loads(launcher_meta.read_text(encoding="utf-8"))
                pid = int(launch.get("pid", -1))
                pending_status = {
                    "run_id": launch.get("run_id", args.run_id),
                    "state": "launching",
                    "pid": pid,
                    "process_alive": pid_alive(pid) if pid > 0 else None,
                    "launch_log": launch.get("launch_log"),
                    "launched_at": launch.get("launched_at"),
                    "note": "status.json not written yet; pipeline process has been launched.",
                }
                if args.json:
                    print(json.dumps(pending_status, indent=2, sort_keys=True))
                    return 0
                print(f"run_id: {pending_status['run_id']}")
                print(f"state: {pending_status['state']}")
                print(f"pid: {pending_status['pid']}")
                print(f"process_alive: {pending_status['process_alive']}")
                print(f"launched_at: {pending_status['launched_at']}")
                print(f"launch_log: {pending_status['launch_log']}")
                print(f"note: {pending_status['note']}")
                return 0
        raise

    status = json.loads(status_path.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    print(f"run_id: {status.get('run_id')}")
    print(f"state: {status.get('state')}")
    print(f"started_at: {status.get('started_at')}")
    print(f"finished_at: {status.get('finished_at')}")
    print(f"run_dir: {status.get('run_dir')}")
    print(f"log_path: {status.get('log_path')}")
    print(f"manifest_path: {status.get('manifest_path')}")
    if isinstance(status.get("pid"), int):
        pid = int(status["pid"])
        print(f"pid: {pid}")
        print(f"process_alive: {pid_alive(pid)}")
    print("stages:")
    for stage, state in status.get("stages", {}).items():
        print(f"  - {stage}: {state}")
    print(f"last_message: {status.get('last_message')}")
    return 0


def add_pipeline_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--registry",
        default="evaluation/configs/pipeline_task_registry.yaml",
        help="Registry config relative to repo root.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu-list", default="1,2")
    parser.add_argument("--parallel-jobs", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--xp-config", default=None)
    parser.add_argument("--logs-glob", default="*_br_checkpoints_gpu*.txt")
    parser.add_argument("--skip-br", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-xp", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--allow-missing-xp",
        action="store_true",
        help="Skip tasks with missing XP CSV artifacts when reconstructing analysis/plot inputs.",
    )
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run_p = sub.add_parser("run", help="Run BR -> XP -> plot pipeline.")
    add_pipeline_run_args(run_p)
    run_p.set_defaults(func=cmd_run)

    launch_p = sub.add_parser(
        "launch",
        help="Launch pipeline in detached mode (survives SSH disconnect).",
    )
    add_pipeline_run_args(launch_p)
    launch_p.set_defaults(func=cmd_launch)

    status_p = sub.add_parser("status", help="Show pipeline status for a run id (or latest).")
    status_p.add_argument("--repo-root", default=".")
    status_p.add_argument(
        "--registry",
        default="evaluation/configs/pipeline_task_registry.yaml",
        help="Registry config relative to repo root.",
    )
    status_p.add_argument("--run-id", default=None)
    status_p.add_argument("--json", action="store_true")
    status_p.set_defaults(func=cmd_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

