# Trajectory cache: record raw per-step Overcooked transitions to a .npz per
# (task, teammate) and replay them on CPU so features can be recomputed
# without re-running rollouts.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("compute_pd")

CACHE_FORMAT_VERSION = 2


def _unwrap(s):
    while hasattr(s, "env_state"):
        s = s.env_state
    return s


class _ReplayState:
    """Duck-typed Overcooked env state exposing only what the updates read."""

    __slots__ = ("agent_inv", "agent_pos", "agent_dir", "maze_map", "wall_map")

    def __init__(self, agent_inv, agent_pos, agent_dir, maze_map, wall_map):
        self.agent_inv = agent_inv
        self.agent_pos = agent_pos
        self.agent_dir = agent_dir
        self.maze_map = maze_map
        self.wall_map = wall_map


class OvercookedTrajRecorder:
    """Accumulate raw per-step Overcooked transitions for one (task, teammate)."""

    def __init__(self, agent: str, task: str, seed: int):
        self.agent = agent
        self.task = task
        self.seed = seed
        self._ep_lengths: List[int] = []
        self._cur = 0
        self.act: List[np.ndarray] = []
        self.rew: List[np.ndarray] = []
        self.inv_pre: List[np.ndarray] = []
        self.inv_post: List[np.ndarray] = []
        self.pos_pre: List[np.ndarray] = []
        self.pos_post: List[np.ndarray] = []
        self.dir_pre: List[np.ndarray] = []
        self.mmw_pre: List[np.ndarray] = []
        self.wall_map: Optional[np.ndarray] = None
        self.mm_template: Optional[np.ndarray] = None
        self.pad: int = 0
        self.H: int = 0
        self.W: int = 0

    def record_step(self, state_pre, state_post, a0: int, a1: int, r0: float, r1: float) -> None:
        sp, sq = _unwrap(state_pre), _unwrap(state_post)
        mm = np.asarray(sp.maze_map)
        wm = np.asarray(sp.wall_map)
        pad = (mm.shape[0] - wm.shape[0]) // 2
        H, W = int(wm.shape[0]), int(wm.shape[1])

        assert mm.min() >= 0 and mm.max() < 256, "maze_map does not fit in uint8"
        if self.wall_map is None:
            self.wall_map = wm.astype(bool)
            self.mm_template = mm.astype(np.uint8)
            self.pad, self.H, self.W = pad, H, W
        else:
            # Static-layout invariants; fail loudly rather than write a lossy cache.
            assert (pad, H, W) == (self.pad, self.H, self.W), "layout geometry changed mid-run"
            assert np.array_equal(wm.astype(bool), self.wall_map), "wall_map changed mid-run"
            ring = self.mm_template.copy()
            ring[pad:pad + H, pad:pad + W, :] = mm[pad:pad + H, pad:pad + W, :]
            assert np.array_equal(ring, mm.astype(np.uint8)), (
                "maze_map changed OUTSIDE the walkable window; cache would be lossy"
            )

        self.act.append(np.array([a0, a1], dtype=np.int8))
        self.rew.append(np.array([r0, r1], dtype=np.float32))
        self.inv_pre.append(np.asarray(sp.agent_inv).reshape(-1).astype(np.int16))
        self.inv_post.append(np.asarray(sq.agent_inv).reshape(-1).astype(np.int16))
        self.pos_pre.append(np.asarray(sp.agent_pos).reshape(-1, 2).astype(np.int16))
        self.pos_post.append(np.asarray(sq.agent_pos).reshape(-1, 2).astype(np.int16))
        self.dir_pre.append(np.asarray(sp.agent_dir).reshape(-1, 2).astype(np.int16))
        self.mmw_pre.append(mm[pad:pad + H, pad:pad + W, :].astype(np.uint8))
        self._cur += 1

    def end_episode(self) -> None:
        self._ep_lengths.append(self._cur)
        self._cur = 0

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if self.wall_map is None:
            raise ValueError(f"no steps recorded for {self.agent}")
        ep_ptr = np.concatenate([[0], np.cumsum(self._ep_lengths)]).astype(np.int32)
        return dict(
            version=np.int32(CACHE_FORMAT_VERSION),
            ep_ptr=ep_ptr,
            act=np.stack(self.act),
            rew=np.stack(self.rew),
            inv_pre=np.stack(self.inv_pre),
            inv_post=np.stack(self.inv_post),
            pos_pre=np.stack(self.pos_pre),
            pos_post=np.stack(self.pos_post),
            dir_pre=np.stack(self.dir_pre),
            mmw_pre=np.stack(self.mmw_pre),
            wall_map=self.wall_map,
            mm_template=self.mm_template,
            pad=np.int32(self.pad),
            H=np.int32(self.H),
            W=np.int32(self.W),
            agent=np.array(self.agent),
            task=np.array(self.task),
            seed=np.int32(self.seed),
        )

    def save(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_arrays())
        return path.stat().st_size


def cache_path(cache_root: Path, task: str, agent: str) -> Path:
    safe = (
        str(agent).replace(" ", "_").replace("(", "").replace(")", "")
        .replace(",", "").replace("/", "_")
    )
    return Path(cache_root) / str(task).replace("/", "_") / f"{safe}.npz"


class CachedEpisode:
    """One episode's worth of raw arrays, sliced out of a cache file."""

    __slots__ = ("act", "rew", "inv_pre", "inv_post", "pos_pre", "pos_post",
                 "dir_pre", "mmw_pre", "wall_map", "mm_template", "pad", "H", "W")

    def __init__(self, blob, lo: int, hi: int):
        self.act = blob["act"][lo:hi]
        self.rew = blob["rew"][lo:hi]
        self.inv_pre = blob["inv_pre"][lo:hi]
        self.inv_post = blob["inv_post"][lo:hi]
        self.pos_pre = blob["pos_pre"][lo:hi]
        self.pos_post = blob["pos_post"][lo:hi]
        self.dir_pre = blob["dir_pre"][lo:hi]
        self.mmw_pre = blob["mmw_pre"][lo:hi]
        self.wall_map = blob["wall_map"]
        self.mm_template = blob["mm_template"]
        self.pad = int(blob["pad"])
        self.H = int(blob["H"])
        self.W = int(blob["W"])

    def __len__(self) -> int:
        return int(self.act.shape[0])

    def maze_map(self, t: int) -> np.ndarray:
        mm = self.mm_template.copy()
        p, H, W = self.pad, self.H, self.W
        mm[p:p + H, p:p + W, :] = self.mmw_pre[t]
        return mm

    def states(self, t: int):
        pre = _ReplayState(self.inv_pre[t], self.pos_pre[t], self.dir_pre[t],
                           self.maze_map(t), self.wall_map)
        post = _ReplayState(self.inv_post[t], self.pos_post[t], self.dir_pre[t],
                            self.maze_map(t), self.wall_map)
        return pre, post


def load_episodes(path: Path) -> List[CachedEpisode]:
    blob = dict(np.load(path, allow_pickle=False))
    ver = int(blob.get("version", 0))
    if ver != CACHE_FORMAT_VERSION:
        raise ValueError(f"{path}: cache format v{ver}, expected v{CACHE_FORMAT_VERSION}")
    ptr = blob["ep_ptr"]
    return [CachedEpisode(blob, int(ptr[i]), int(ptr[i + 1])) for i in range(len(ptr) - 1)]


def replay_counts(path: Path) -> List[Any]:
    """Rebuild per-episode OvercookedEpisodeCounts from a cache file by replaying `overcooked_step_update`."""
    from scripts.population_diversity.pd_overcooked_features import (
        OvercookedEpisodeCounts,
        overcooked_step_update,
    )

    out = []
    for ep in load_episodes(path):
        counts = OvercookedEpisodeCounts()
        ep_return = 0.0
        for t in range(len(ep)):
            pre, post = ep.states(t)
            a0, a1 = int(ep.act[t][0]), int(ep.act[t][1])
            r0, r1 = float(ep.rew[t][0]), float(ep.rew[t][1])
            overcooked_step_update(counts, a0, a1, r0, r1, pre, post, {})
            ep_return += r0
        counts.final_score = ep_return
        counts.episode_length = len(ep)
        out.append(counts)
    return out


def save_batched_traj(traj: Dict[str, Any], agent: str, task: str, seed: int, path: Path) -> int:
    """Persist the stacked output of `rollout_two_policy_batched` as a cache file."""
    E, T = traj["act"].shape[0], traj["act"].shape[1]
    pad, H, W = int(traj["pad"]), int(traj["H"]), int(traj["W"])

    mm_full = traj["mm_full_pre"]
    template = mm_full[0, 0].copy()
    rebuilt = np.broadcast_to(template, mm_full.shape).copy()
    rebuilt[:, :, pad:pad + H, pad:pad + W, :] = traj["mmw_pre"]
    assert np.array_equal(rebuilt, mm_full), (
        "maze_map changed OUTSIDE the walkable window; the window-only cache would be lossy"
    )

    # Fixed horizon: every episode must end exactly at T.
    done = traj["done"]
    assert bool(done[:, -1].all()) and not bool(done[:, :-1].any()), (
        "episodes did not all terminate exactly at max_steps; fixed-length cache invalid"
    )

    def flat(k, dtype=None):
        a = traj[k].reshape((E * T,) + traj[k].shape[2:])
        return a if dtype is None else a.astype(dtype)

    arrays = dict(
        version=np.int32(CACHE_FORMAT_VERSION),
        ep_ptr=(np.arange(E + 1) * T).astype(np.int32),
        act=flat("act", np.int8),
        rew=flat("rew", np.float32),
        inv_pre=flat("inv_pre", np.int16),
        inv_post=flat("inv_post", np.int16),
        pos_pre=flat("pos_pre", np.int16),
        pos_post=flat("pos_post", np.int16),
        dir_pre=flat("dir_pre", np.int16),
        mmw_pre=flat("mmw_pre", np.uint8),
        wall_map=traj["wall_map"],
        mm_template=template,
        pad=np.int32(pad), H=np.int32(H), W=np.int32(W),
        agent=np.array(agent), task=np.array(task), seed=np.int32(seed),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size
