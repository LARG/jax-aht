"""Overcooked feature computation from recorded trajectories: per-step event counters,
per-episode feature vectors, partner-contingent statistics, and counter-cell support."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from scripts.population_diversity.pd_events import (
    OVERCOOKED_FEATURE_NAMES,
    OVERCOOKED_SHAPED_INFOS,
)

_OC_UP, _OC_DOWN, _OC_RIGHT, _OC_LEFT, _OC_STAY, _OC_INTERACT = range(6)


_OC_RESOURCE_KINDS = ("pot", "onion_pile", "plate_pile", "goal")

# overcooked object codes (jaxmarl OBJECT_TO_INDEX)
_OC_EMPTY, _OC_WALL, _OC_ONION, _OC_ONION_PILE = 1, 2, 3, 4
_OC_PLATE, _OC_PLATE_PILE, _OC_GOAL, _OC_POT, _OC_DISH, _OC_AGENT = 5, 6, 7, 8, 9, 10
_OC_POT_EMPTY = 23  # pot status when no onions are in the pot

# DIR_TO_VEC, indexed by move action (up/down/right/left == 0/1/2/3).
_OC_DIR_TO_VEC = ((0, -1), (0, 1), (1, 0), (-1, 0))

# counter-mediated events, split into "put an item down" and "take an item back".
_OC_PUT_ON_COUNTER = {
    "put_onion_on_X": _OC_ONION,
    "put_dish_on_X": _OC_PLATE,
    "put_soup_on_X": _OC_DISH,
}
_OC_PICKUP_FROM_COUNTER = ("pickup_onion_from_X", "pickup_dish_from_X", "pickup_soup_from_X")

_OC_ITEM_FAMILY = {_OC_ONION: 0, _OC_PLATE: 1, _OC_DISH: 1}
_OC_MI_MIN_PLACEMENTS = 4  # below this the plug-in MI is dominated by its own bias


@dataclass
class OvercookedEpisodeCounts:
    """per-episode overcooked shaped_infos event counts."""
    counts_by_event: Dict[str, int] = field(default_factory=lambda: {ev: 0 for ev in OVERCOOKED_SHAPED_INFOS})
    n_steps: int = 0
    n_handoff_given: int = 0
    n_handoff_received: int = 0
    n_self_retrieve: int = 0
    n_region_NW: int = 0
    n_region_NE: int = 0
    n_region_SW: int = 0
    n_region_SE: int = 0
    n_interact_into_partner: int = 0
    n_blocked_partner: int = 0
    partner_dist_sum: float = 0.0
    dist_norm: int = 0  # (height + width) of the walkable grid
    grid_h: int = 0     # height of the walkable grid (for normalizing cell positions)
    grid_w: int = 0     # width of the walkable grid
    dwell_steps: list = field(default_factory=list)  # sit-steps of each tracked-agent drop that got picked up
    ang_signed_sum: float = 0.0       # signed angular displacement around the layout centroid
    ang_abs_sum: float = 0.0
    # per-resource-kind, per-instance usage counts by the tracked agent (kind -> pos -> uses)
    resource_usage: Dict[str, Dict[Tuple[int, int], int]] = field(
        default_factory=lambda: {k: {} for k in _OC_RESOURCE_KINDS}
    )
    # same, for the PARTNER (only used relative to the tracked agent's usage: *_sharing)
    resource_usage_partner: Dict[str, Dict[Tuple[int, int], int]] = field(
        default_factory=lambda: {k: {} for k in _OC_RESOURCE_KINDS}
    )
    counter_ledger: Dict[Tuple[int, int], Tuple[int, int, int]] = field(default_factory=dict)
    # counter position -> number of placements by the tracked agent
    placements_by_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)
    placements_by_pos_item: Dict[Tuple[Tuple[int, int], int], int] = field(default_factory=dict)
    visited_self: set = field(default_factory=set)
    visited_partner: set = field(default_factory=set)
    # canonical (sorted by (y, x)) instance positions per resource kind, cached at step 0
    resource_pos: Optional[Dict[str, List[Tuple[int, int]]]] = None
    final_score: float = 0.0
    episode_length: int = 0


def overcooked_episode_to_vector(
    counts: OvercookedEpisodeCounts, return_norm: float = 0.0, length_norm: float = 0.0
) -> np.ndarray:
    """per-episode overcooked counts to the theta contribution vector."""
    ev = counts.counts_by_event
    feats = [float(ev.get(name, 0)) for name in OVERCOOKED_SHAPED_INFOS]

    ep_len = max(1, int(counts.n_steps) or int(counts.episode_length) or 1)

    # items the tracked agent left on a counter that were never picked back up
    dead_drop = sum(1 for placer, _item, _t in counts.counter_ledger.values() if placer == 1)

    # how spread out the tracked agent's counter usage was
    placement_counts = list(counts.placements_by_pos.values())
    total_placements = sum(placement_counts)
    counter_entropy = 0.0
    if total_placements >= 2 and len(placement_counts) >= 2:
        entropy = 0.0
        for n in placement_counts:
            p = n / total_placements
            if p > 0.0:
                entropy -= p * np.log(p)
        counter_entropy = entropy / np.log(len(placement_counts))

    modal_counter_frac = 0.0
    if total_placements > 0:
        modal_counter_frac = max(counts.placements_by_pos.values()) / total_placements

    norm_x = max(1, (counts.grid_w or 1) - 1)
    norm_y = max(1, (counts.grid_h or 1) - 1)

    def _modal_cell(items: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
        """most-placed cell restricted to `items`; None if no such placement."""
        agg: Dict[Tuple[int, int], int] = {}
        for (pos, item), n in counts.placements_by_pos_item.items():
            if item in items:
                agg[pos] = agg.get(pos, 0) + n
        if not agg:
            return None
        # tie-break on lowest (y, x)
        return min(agg.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))[0]

    _ONION_ITEMS = (_OC_ONION,)
    _DISH_ITEMS = (_OC_PLATE, _OC_DISH)  # empty plate and plated soup
    onion_cell = _modal_cell(_ONION_ITEMS)
    dish_cell = _modal_cell(_DISH_ITEMS)
    onion_counter_x = float(onion_cell[0]) / norm_x if onion_cell else 0.0
    onion_counter_y = float(onion_cell[1]) / norm_y if onion_cell else 0.0
    dish_counter_x = float(dish_cell[0]) / norm_x if dish_cell else 0.0
    dish_counter_y = float(dish_cell[1]) / norm_y if dish_cell else 0.0

    counter_item_mi = 0.0
    if counts.placements_by_pos_item:
        cells = sorted({pos for pos, _ in counts.placements_by_pos_item})
        fams = sorted({_OC_ITEM_FAMILY.get(item, 0) for _, item in counts.placements_by_pos_item})
        n_pl = float(sum(counts.placements_by_pos_item.values()))
        if len(cells) >= 2 and len(fams) >= 2 and n_pl >= _OC_MI_MIN_PLACEMENTS:
            joint = np.zeros((len(cells), len(fams)), dtype=np.float64)
            ci = {c: i for i, c in enumerate(cells)}
            fi = {f: i for i, f in enumerate(fams)}
            for (pos, item), n in counts.placements_by_pos_item.items():
                joint[ci[pos], fi[_OC_ITEM_FAMILY.get(item, 0)]] += n

            def _H(cnt: np.ndarray) -> float:
                p = cnt[cnt > 0] / n_pl
                return float(-(p * np.log(p)).sum()) + (p.size - 1) / (2.0 * n_pl)

            mi = _H(joint.sum(axis=1)) + _H(joint.sum(axis=0)) - _H(joint.reshape(-1))
            denom = np.log(min(len(cells), len(fams)))
            counter_item_mi = float(np.clip(mi / denom, 0.0, 1.0)) if denom > 0 else 0.0

    region_fracs = [
        counts.n_region_NW / ep_len,
        counts.n_region_NE / ep_len,
        counts.n_region_SW / ep_len,
    ]

    union = counts.visited_self | counts.visited_partner
    territory_overlap = (
        len(counts.visited_self & counts.visited_partner) / len(union) if union else 0.0
    )

    n_onion = (
        ev.get("pickup_onion_from_O", 0)
        + ev.get("pickup_onion_from_X", 0)
        + ev.get("put_onion_on_X", 0)
        + ev.get("PLACEMENT_IN_POT", 0)
    )
    n_plate = (
        ev.get("pickup_dish_from_D", 0)
        + ev.get("pickup_dish_from_X", 0)
        + ev.get("put_dish_on_X", 0)
        + ev.get("SOUP_PICKUP", 0)
        + ev.get("pickup_soup_from_X", 0)
        + ev.get("put_soup_on_X", 0)
        + ev.get("delivery", 0)
    )
    n_role = n_onion + n_plate
    onion_role_frac = (n_onion / n_role) if n_role > 0 else 0.0
    pot_role_frac = (ev.get("PLACEMENT_IN_POT", 0) / n_role) if n_role > 0 else 0.0
    plating_role_frac = (
        (ev.get("pickup_dish_from_D", 0) + ev.get("pickup_dish_from_X", 0)
         + ev.get("SOUP_PICKUP", 0)) / n_role
    ) if n_role > 0 else 0.0

    mean_partner_distance = (counts.partner_dist_sum / ep_len) / max(1, counts.dist_norm)

    remaining = [
        max(0, ep_len - t) for placer, _item, t in counts.counter_ledger.values() if placer == 1
    ]
    dwell_all = counts.dwell_steps + remaining
    counter_item_dwell = float(np.median(dwell_all)) / ep_len if dwell_all else 0.0

    circulation_direction = (
        counts.ang_signed_sum / counts.ang_abs_sum if counts.ang_abs_sum > 0 else 0.0
    )

    def _usage_entropy(kind: str) -> float:
        # 0.0 when the layout has < 2 instances OR the agent never used the kind
        positions = (counts.resource_pos or {}).get(kind, [])
        usage = counts.resource_usage.get(kind, {})
        total = sum(usage.values())
        if len(positions) < 2 or total <= 0:
            return 0.0
        entropy = 0.0
        for pos in positions:
            p = usage.get(pos, 0) / total
            if p > 0.0:
                entropy -= p * np.log(p)
        return entropy / np.log(len(positions))

    def _usage_sharing(kind: str) -> float:
        positions = (counts.resource_pos or {}).get(kind, [])
        mine = counts.resource_usage.get(kind, {})
        theirs = counts.resource_usage_partner.get(kind, {})
        tm, tt = sum(mine.values()), sum(theirs.values())
        if len(positions) < 2 or tm <= 0 or tt <= 0:
            return 0.0
        return float(sum(min(mine.get(pos, 0) / tm, theirs.get(pos, 0) / tt) for pos in positions))

    def _modal_resource(kind: str) -> Tuple[float, float]:
        positions = (counts.resource_pos or {}).get(kind, [])
        usage = counts.resource_usage.get(kind, {})
        if len(positions) < 2 or not usage:
            return 0.0, 0.0
        pos = min(usage.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))[0]
        return float(pos[0]) / norm_x, float(pos[1]) / norm_y

    pot_usage_entropy = _usage_entropy("pot")
    modal_pot_x, modal_pot_y = _modal_resource("pot")
    modal_onion_pile_x, modal_onion_pile_y = _modal_resource("onion_pile")
    modal_plate_pile_x, modal_plate_pile_y = _modal_resource("plate_pile")
    modal_goal_x, modal_goal_y = _modal_resource("goal")
    pot_sharing = _usage_sharing("pot")
    onion_pile_sharing = _usage_sharing("onion_pile")
    plate_pile_sharing = _usage_sharing("plate_pile")
    goal_sharing = _usage_sharing("goal")
    onion_pile_usage_entropy = _usage_entropy("onion_pile")
    plate_pile_usage_entropy = _usage_entropy("plate_pile")
    goal_usage_entropy = _usage_entropy("goal")

    feats += [
        float(counts.n_handoff_given),
        float(counts.n_handoff_received),
        float(dead_drop),
        float(counts.n_self_retrieve),
        float(counter_entropy),
        float(modal_counter_frac),
        float(counter_item_mi),
        float(onion_counter_x),
        float(onion_counter_y),
        float(dish_counter_x),
        float(dish_counter_y),
        *[float(x) for x in region_fracs],
        float(territory_overlap),
        float(onion_role_frac),
        float(mean_partner_distance),
        float(counts.n_interact_into_partner),
        float(counts.n_blocked_partner),
        float(counter_item_dwell),
        float(circulation_direction),
        float(pot_usage_entropy),
        float(onion_pile_usage_entropy),
        float(plate_pile_usage_entropy),
        float(goal_usage_entropy),
        float(pot_role_frac),
        float(plating_role_frac),
        float(modal_pot_x),
        float(modal_pot_y),
        float(modal_onion_pile_x),
        float(modal_onion_pile_y),
        float(modal_plate_pile_x),
        float(modal_plate_pile_y),
        float(modal_goal_x),
        float(modal_goal_y),
        float(pot_sharing),
        float(onion_pile_sharing),
        float(plate_pile_sharing),
        float(goal_sharing),
    ]
    assert len(feats) == len(OVERCOOKED_FEATURE_NAMES), (
        f"overcooked feature vector length {len(feats)} != {len(OVERCOOKED_FEATURE_NAMES)} names"
    )
    return np.array(feats, dtype=np.float64)


def _oc_scan_resources(mm: np.ndarray, pad: int, H: int, W: int) -> Dict[str, List[Tuple[int, int]]]:
    """positions (x, y) of each static resource, in canonical (y, x)-sorted order."""
    code_to_kind = {
        _OC_POT: "pot",
        _OC_ONION_PILE: "onion_pile",
        _OC_PLATE_PILE: "plate_pile",
        _OC_GOAL: "goal",
    }
    out: Dict[str, List[Tuple[int, int]]] = {k: [] for k in _OC_RESOURCE_KINDS}
    for y in range(H):
        for x in range(W):
            py, px = pad + y, pad + x
            if not (0 <= py < mm.shape[0] and 0 <= px < mm.shape[1]):
                continue
            kind = code_to_kind.get(int(mm[py, px, 0]))
            if kind is not None:
                out[kind].append((x, y))  # scanned in (y, x) order already
    return out


def _oc_classify_event(inv0: int, inv1: int, obj: int) -> Optional[str]:
    """map an inventory transition + faced-cell object to a shaped_infos event name."""
    if inv0 == inv1:
        return None
    if inv0 == _OC_EMPTY and inv1 == _OC_ONION:
        return "pickup_onion_from_O" if obj == _OC_ONION_PILE else "pickup_onion_from_X"
    if inv0 == _OC_EMPTY and inv1 == _OC_PLATE:
        return "pickup_dish_from_D" if obj == _OC_PLATE_PILE else "pickup_dish_from_X"
    if inv0 == _OC_EMPTY and inv1 == _OC_DISH:
        return "pickup_soup_from_X"
    if inv0 == _OC_PLATE and inv1 == _OC_DISH:
        return "SOUP_PICKUP"
    if inv0 == _OC_ONION and inv1 == _OC_EMPTY:
        return "PLACEMENT_IN_POT" if obj == _OC_POT else "put_onion_on_X"
    if inv0 == _OC_PLATE and inv1 == _OC_EMPTY:
        return "put_dish_on_X"
    if inv0 == _OC_DISH and inv1 == _OC_EMPTY:
        return "delivery" if obj == _OC_GOAL else "put_soup_on_X"
    return None


def overcooked_step_update(
    counts: OvercookedEpisodeCounts,
    a0: int,
    a1: int,
    r0: float,
    r1: float,
    state_pre,
    state_post,
    info: Dict,
) -> None:
    """reconstruct shaped_infos events from overcooked state transitions."""
    EMPTY, WALL, ONION, ONION_PILE, PLATE, PLATE_PILE, GOAL, POT, DISH = (
        _OC_EMPTY, _OC_WALL, _OC_ONION, _OC_ONION_PILE, _OC_PLATE,
        _OC_PLATE_PILE, _OC_GOAL, _OC_POT, _OC_DISH)
    STAY_A, INTERACT_A = 4, 5
    POT_EMPTY = _OC_POT_EMPTY

    def _unwrap(s):
        while hasattr(s, "env_state"):
            s = s.env_state
        return s

    sp, sq = _unwrap(state_pre), _unwrap(state_post)
    try:
        inv_pre = np.asarray(sp.agent_inv).reshape(-1)
        inv_post = np.asarray(sq.agent_inv).reshape(-1)
        pos_pre = np.asarray(sp.agent_pos).reshape(-1, 2)
        pos_post = np.asarray(sq.agent_pos).reshape(-1, 2)
        adir = np.asarray(sp.agent_dir).reshape(-1, 2)
        mm = np.asarray(sp.maze_map)
        wm = np.asarray(sp.wall_map)
    except AttributeError:
        return  # not an overcooked state, nothing to reconstruct

    pad = (mm.shape[0] - wm.shape[0]) // 2
    ev = counts.counts_by_event
    H, W = int(wm.shape[0]), int(wm.shape[1])

    i = 1
    action = int(a1)
    inv0, inv1 = int(inv_pre[i]), int(inv_post[i])
    fx = int(pos_pre[i][0] + adir[i][0])
    fy = int(pos_pre[i][1] + adir[i][1])
    on_grid = (0 <= fx < W) and (0 <= fy < H)
    py, px = pad + fy, pad + fx
    obj = int(mm[py, px, 0]) if (0 <= py < mm.shape[0] and 0 <= px < mm.shape[1]) else EMPTY
    is_table = bool(wm[fy, fx]) if on_grid else False
    moved = not np.array_equal(pos_pre[i], pos_post[i])

    _overcooked_derived_update(
        counts, a0, a1, inv_pre, inv_post, pos_pre, pos_post, adir, mm, pad, H, W, wm,
    )

    if inv0 != inv1:
        # interact succeeded, classify by inventory delta + faced cell
        if inv0 == EMPTY and inv1 == ONION:
            ev["pickup_onion_from_O" if obj == ONION_PILE else "pickup_onion_from_X"] += 1
        elif inv0 == EMPTY and inv1 == PLATE:
            ev["pickup_dish_from_D" if obj == PLATE_PILE else "pickup_dish_from_X"] += 1
        elif inv0 == EMPTY and inv1 == DISH:
            ev["pickup_soup_from_X"] += 1
        elif inv0 == PLATE and inv1 == DISH:
            ev["SOUP_PICKUP"] += 1
        elif inv0 == ONION and inv1 == EMPTY:
            ev["PLACEMENT_IN_POT" if obj == POT else "put_onion_on_X"] += 1
        elif inv0 == PLATE and inv1 == EMPTY:
            ev["put_dish_on_X"] += 1
        elif inv0 == DISH and inv1 == EMPTY:
            ev["delivery" if obj == GOAL else "put_soup_on_X"] += 1
    else:
        # no inventory change, movement / stay / no-op interact
        if action == STAY_A:
            ev["STAY"] += 1
        elif action == INTERACT_A:
            if is_table and obj not in (WALL, EMPTY):
                ev["IDLE_INTERACT_X"] += 1
            else:
                ev["IDLE_INTERACT_EMPTY"] += 1
        elif moved:
            ev["MOVEMENT"] += 1
        else:
            ev["IDLE_MOVEMENT"] += 1


def _overcooked_derived_update(
    counts: OvercookedEpisodeCounts,
    a0: int, a1: int,
    inv_pre: np.ndarray, inv_post: np.ndarray,
    pos_pre: np.ndarray, pos_post: np.ndarray,
    adir: np.ndarray, mm: np.ndarray, pad: int, H: int, W: int,
    wm: Optional[np.ndarray] = None,
) -> None:
    """extended-vocabulary bookkeeping for overcooked."""
    TRACKED, PARTNER = 1, 0
    INTERACT_A = 5
    actions = (int(a0), int(a1))

    counts.n_steps += 1
    step_idx = counts.n_steps - 1
    counts.dist_norm = H + W
    counts.grid_h, counts.grid_w = H, W

    if counts.resource_pos is None:
        counts.resource_pos = _oc_scan_resources(mm, pad, H, W)

    faced: List[Tuple[int, int, int]] = []
    for j in (0, 1):
        fx = int(pos_pre[j][0] + adir[j][0])
        fy = int(pos_pre[j][1] + adir[j][1])
        py, px = pad + fy, pad + fx
        if 0 <= py < mm.shape[0] and 0 <= px < mm.shape[1]:
            obj = int(mm[py, px, 0])
        else:
            obj = _OC_EMPTY
        faced.append((fx, fy, obj))

    events = [
        _oc_classify_event(int(inv_pre[j]), int(inv_post[j]), faced[j][2]) for j in (0, 1)
    ]

    for j in (0, 1):
        if events[j] not in _OC_PICKUP_FROM_COUNTER:
            continue
        entry = counts.counter_ledger.pop((faced[j][0], faced[j][1]), None)
        if entry is None:
            continue  # item wasn't placed by either agent this episode
        placer = entry[0]
        if placer == TRACKED:
            # dwell time of a tracked-agent drop, resolved at pickup (by either agent)
            counts.dwell_steps.append(max(0, step_idx - entry[2]))
        if j == TRACKED:
            if placer == PARTNER:
                counts.n_handoff_received += 1
            else:
                counts.n_self_retrieve += 1
        elif placer == TRACKED:
            counts.n_handoff_given += 1

    for j in (0, 1):
        item = _OC_PUT_ON_COUNTER.get(events[j] or "")
        if item is None:
            continue
        pos = (faced[j][0], faced[j][1])
        counts.counter_ledger[pos] = (j, item, step_idx)
        if j == TRACKED:
            counts.placements_by_pos[pos] = counts.placements_by_pos.get(pos, 0) + 1
            key = (pos, int(item))
            counts.placements_by_pos_item[key] = counts.placements_by_pos_item.get(key, 0) + 1

    event_to_resource = {
        "PLACEMENT_IN_POT": "pot",
        "SOUP_PICKUP": "pot",
        "pickup_onion_from_O": "onion_pile",
        "pickup_dish_from_D": "plate_pile",
        "delivery": "goal",
    }
    for j, store in ((TRACKED, counts.resource_usage), (1 - TRACKED, counts.resource_usage_partner)):
        kind = event_to_resource.get(events[j] or "")
        if kind is not None:
            positions = counts.resource_pos.get(kind, [])
            used = (faced[j][0], faced[j][1])
            if used in positions:
                kind_usage = store.setdefault(kind, {})
                kind_usage[used] = kind_usage.get(used, 0) + 1

    sx, sy = int(pos_pre[TRACKED][0]), int(pos_pre[TRACKED][1])
    mid_x, mid_y = W / 2.0, H / 2.0
    if sy < mid_y and sx < mid_x:
        counts.n_region_NW += 1
    elif sy < mid_y:
        counts.n_region_NE += 1
    elif sx < mid_x:
        counts.n_region_SW += 1
    else:
        counts.n_region_SE += 1

    counts.visited_self.add((sx, sy))
    counts.visited_self.add((int(pos_post[TRACKED][0]), int(pos_post[TRACKED][1])))
    counts.visited_partner.add((int(pos_pre[PARTNER][0]), int(pos_pre[PARTNER][1])))
    counts.visited_partner.add((int(pos_post[PARTNER][0]), int(pos_post[PARTNER][1])))

    counts.partner_dist_sum += abs(sx - int(pos_pre[PARTNER][0])) + abs(
        sy - int(pos_pre[PARTNER][1])
    )

    if actions[TRACKED] == INTERACT_A and faced[TRACKED][2] == _OC_AGENT:
        counts.n_interact_into_partner += 1

    partner_action = actions[PARTNER]
    if 0 <= partner_action < len(_OC_DIR_TO_VEC) and np.array_equal(
        pos_pre[PARTNER], pos_post[PARTNER]
    ):
        dx, dy = _OC_DIR_TO_VEC[partner_action]
        target = (int(pos_pre[PARTNER][0]) + dx, int(pos_pre[PARTNER][1]) + dy)
        if (sx, sy) == target:
            counts.n_blocked_partner += 1

    cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
    ex, ey = int(pos_post[TRACKED][0]), int(pos_post[TRACKED][1])
    if (sx, sy) != (ex, ey):
        v_pre = (sx - cx0, sy - cy0)
        v_post = (ex - cx0, ey - cy0)
        if (v_pre != (0.0, 0.0)) and (v_post != (0.0, 0.0)):
            da = float(np.arctan2(v_post[1], v_post[0]) - np.arctan2(v_pre[1], v_pre[0]))
            if da > np.pi:
                da -= 2.0 * np.pi
            elif da < -np.pi:
                da += 2.0 * np.pi
            if da != 0.0:
                counts.ang_signed_sum += da
                counts.ang_abs_sum += abs(da)



USED_CELL_MIN_FRAC = 0.05


def _episode_modal_cell(placements_by_pos_item, items):
    agg: Dict[Tuple[int, int], int] = {}
    for (pos, item), n in placements_by_pos_item.items():
        if items is None or item in items:
            agg[pos] = agg.get(pos, 0) + n
    if not agg:
        return None
    return min(agg.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))[0]


def overcooked_cell_support(ep_counts: List[OvercookedEpisodeCounts]) -> Dict[str, Any]:
    """Per-teammate modal placement cell (all items / onions / plates) and the set of."""
    subsets = {"all": None, "onion": (_OC_ONION,), "dish": (_OC_PLATE, _OC_DISH)}
    row: Dict[str, Any] = {"n_episodes": len(ep_counts)}
    for name, items in subsets.items():
        cells = [c for c in (_episode_modal_cell(ep.placements_by_pos_item, items) for ep in ep_counts)
                 if c is not None]
        if cells:
            cell, n_modal = Counter(cells).most_common(1)[0]
            row[f"modal_{name}_x"], row[f"modal_{name}_y"] = int(cell[0]), int(cell[1])
            row[f"modal_{name}_frac"] = float(n_modal) / len(cells)
            row[f"modal_{name}_n_distinct_ep"] = int(len(set(cells)))
        else:
            row[f"modal_{name}_x"] = row[f"modal_{name}_y"] = -1
            row[f"modal_{name}_frac"] = 0.0
            row[f"modal_{name}_n_distinct_ep"] = 0
    pooled: Dict[Tuple[int, int], int] = {}
    for ep in ep_counts:
        for (pos, _item), n in ep.placements_by_pos_item.items():
            pooled[pos] = pooled.get(pos, 0) + n
    tot = float(sum(pooled.values()))
    used = {pos for pos, n in pooled.items() if tot > 0 and n / tot >= USED_CELL_MIN_FRAC}
    row["used_all_x"] = "|".join(str(v) for v in sorted({p[0] for p in used}))
    row["used_all_y"] = "|".join(str(v) for v in sorted({p[1] for p in used}))
    return row


_MIN_SAMPLES = 20
# Manhattan distance at which the two agents count as "in an encounter".
_CLOSE_DIST = 2
# Reaction window (steps) for attributing a direction switch to an encounter.
_REACT_WINDOW = 3



def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _step_directions(pos_pre: np.ndarray, pos_post: np.ndarray, cx: float, cy: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step signed circulation direction around the layout centroid."""
    v_pre = pos_pre.astype(np.float64) - np.array([cx, cy])
    v_post = pos_post.astype(np.float64) - np.array([cx, cy])
    moved = np.any(pos_pre != pos_post, axis=-1)
    nonzero = (np.abs(v_pre).sum(-1) > 0) & (np.abs(v_post).sum(-1) > 0)
    da = _wrap_pi(np.arctan2(v_post[:, 1], v_post[:, 0]) - np.arctan2(v_pre[:, 1], v_pre[:, 0]))
    valid = moved & nonzero & (da != 0.0)
    return np.sign(da), valid


def _mi_normalized(x: np.ndarray, y: np.ndarray) -> float:
    """Miller-Madow-corrected I(x; y) / H(x) for two binary variables, in [0, 1]."""
    n = x.size
    if n < _MIN_SAMPLES:
        return 0.0
    xs, ys = np.unique(x), np.unique(y)
    if xs.size < 2 or ys.size < 2:
        return 0.0  # one of the variables is constant -> no information to share

    def _H(counts: np.ndarray) -> float:
        p = counts[counts > 0] / n
        h = float(-(p * np.log(p)).sum())
        return h + (p.size - 1) / (2.0 * n)  # Miller-Madow

    joint = np.array([[np.sum((x == a) & (y == b)) for b in ys] for a in xs], dtype=np.float64)
    h_x = _H(joint.sum(axis=1))
    h_y = _H(joint.sum(axis=0))
    h_xy = _H(joint.reshape(-1))
    if h_x <= 0.0:
        return 0.0
    return float(np.clip((h_x + h_y - h_xy) / h_x, 0.0, 1.0))


def _phi(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two +-1 variables (the signed twin of the MI)."""
    if x.size < _MIN_SAMPLES or x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))


def episode_contingency_features(ep) -> np.ndarray:
    """Compute the partner-contingent block for one CachedEpisode."""
    TRACKED, PARTNER = 1, 0
    T = len(ep)
    H, W = ep.H, ep.W
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    s_pre = ep.pos_pre[:, TRACKED, :]
    s_post = ep.pos_post[:, TRACKED, :]
    p_pre = ep.pos_pre[:, PARTNER, :]
    p_post = ep.pos_post[:, PARTNER, :]

    d_self, v_self = _step_directions(s_pre, s_post, cx, cy)
    d_part, v_part = _step_directions(p_pre, p_post, cx, cy)

    a_self = np.arctan2(s_pre[:, 1] - cy, s_pre[:, 0] - cx)
    a_part = np.arctan2(p_pre[:, 1] - cy, p_pre[:, 0] - cx)
    r_self = np.abs(s_pre - np.array([cx, cy])).sum(-1)
    r_part = np.abs(p_pre - np.array([cx, cy])).sum(-1)
    dphi = _wrap_pi(a_part - a_self)
    v_bear = (r_self > 0) & (r_part > 0) & (dphi != 0.0)
    bearing = np.sign(dphi)

    m = v_self & v_bear
    route_partner_mi = _mi_normalized(d_self[m], bearing[m])
    route_partner_phi = _phi(d_self[m], bearing[m])

    dist = np.abs(s_pre - p_pre).sum(-1)
    close = dist <= _CLOSE_DIST

    idx = np.flatnonzero(v_self)
    switch_encounter_lift = 0.0
    if idx.size >= _MIN_SAMPLES:
        later = idx[1:]
        switched = (d_self[idx][1:] != d_self[idx][:-1]).astype(np.float64)
        # was there a close approach in the _REACT_WINDOW steps before `later`?
        cum = np.concatenate([[0], np.cumsum(close.astype(np.int64))])
        lo = np.maximum(0, later - _REACT_WINDOW)
        recent = (cum[later] - cum[lo]) > 0
        if recent.any() and (~recent).any():
            switch_encounter_lift = float(switched[recent].mean() - switched.mean())

    right = s_pre[:, 0] > cx
    p_right = p_pre[:, 0] > cx
    region_conditional_lift = 0.0
    if p_right.sum() >= _MIN_SAMPLES and T > 0:
        region_conditional_lift = float(right[p_right].mean() - right.mean())

    yield_rate = 0.0
    starts = np.flatnonzero(close & ~np.concatenate([[False], close[:-1]]))
    if starts.size:
        n_self = n_part = 0
        for t0 in starts:
            t1 = min(T, t0 + _REACT_WINDOW + 1)
            rs = _reversed_in(d_self, v_self, t0, t1)
            rp = _reversed_in(d_part, v_part, t0, t1)
            if rs and not rp:
                n_self += 1
            elif rp and not rs:
                n_part += 1
        if n_self + n_part > 0:
            yield_rate = n_self / (n_self + n_part)

    return np.array([
        route_partner_mi,
        route_partner_phi,
        switch_encounter_lift,
        region_conditional_lift,
        yield_rate,
    ], dtype=np.float64)


def _reversed_in(d: np.ndarray, valid: np.ndarray, t0: int, t1: int) -> bool:
    """Did the direction sequence flip sign inside [t0, t1)?."""
    seq = d[t0:t1][valid[t0:t1]]
    return bool(seq.size >= 2 and np.any(seq[1:] != seq[:-1]))


_PHASE_WINDOW = 15
# Circular shifts used to build the null for the MI excess statistics.
_NULL_SHIFTS = (53, 101, 157, 211, 277)
# Steps allowed for a contention to resolve / for an anticipatory reroute.
_RESOLVE_WINDOW = 5
_ANTICIPATE_WINDOW = 3
# Ring separation at or below which a step counts as "proximate".
_PROX_SEP = 2
# Overcooked move actions (up/down/right/left); 4 = stay, 5 = interact.
_MOVE_ACTIONS = (0, 1, 2, 3)
_N_ACTIONS = 6




def _portable_steps(ep, cx: float, cy: float):
    """Layout-portable stand-ins for the ring bookkeeping."""
    TRACKED, PARTNER = 1, 0
    d_s, v_s = _step_directions(ep.pos_pre[:, TRACKED], ep.pos_post[:, TRACKED], cx, cy)
    d_p, v_p = _step_directions(ep.pos_pre[:, PARTNER], ep.pos_post[:, PARTNER], cx, cy)
    s = np.where(v_s, d_s, 0).astype(np.int64)
    p = np.where(v_p, d_p, 0).astype(np.int64)
    a = ep.pos_pre[:, TRACKED].astype(np.float64)
    b = ep.pos_pre[:, PARTNER].astype(np.float64)
    absep = np.abs(a - b).sum(axis=-1)
    ang_a = np.arctan2(a[:, 1] - cy, a[:, 0] - cx)
    ang_b = np.arctan2(b[:, 1] - cy, b[:, 0] - cx)
    dang = _wrap_pi(ang_b - ang_a)
    sep_signed = np.sign(dang).astype(np.int64)
    sep_signed[absep == 0] = 0
    return s, p, sep_signed, absep


def _mi_maxnorm(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Miller-Madow-corrected I(x; y) divided by log(k), the max entropy."""
    n = x.size
    if n < _MIN_SAMPLES:
        return 0.0
    xs, ys = np.unique(x), np.unique(y)
    if xs.size < 2 or ys.size < 2:
        return 0.0

    def _H(counts: np.ndarray) -> float:
        p = counts[counts > 0] / n
        return float(-(p * np.log(p)).sum()) + (p.size - 1) / (2.0 * n)

    joint = np.array([[np.sum((x == a) & (y == b)) for b in ys] for a in xs], dtype=np.float64)
    mi = _H(joint.sum(axis=1)) + _H(joint.sum(axis=0)) - _H(joint.reshape(-1))
    return float(mi / np.log(k))


def _mi_excess(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """`_mi_maxnorm` minus its circular-shift null, clipped to [0, 1]."""
    if x.size < _MIN_SAMPLES:
        return 0.0
    obs = _mi_maxnorm(x, y, k)
    null = float(np.mean([_mi_maxnorm(x, np.roll(y, s), k) for s in _NULL_SHIFTS]))
    return float(np.clip(obs - null, 0.0, 1.0))


def episode_v5_features(ep) -> np.ndarray:
    """Contention (Task A) + action-action MI (Task B) block for one CachedEpisode."""
    TRACKED, PARTNER = 1, 0
    T = len(ep)
    cx, cy = (ep.W - 1) / 2.0, (ep.H - 1) / 2.0
    s, p, sep, absep = _portable_steps(ep, cx, cy)   # rotational sense / Manhattan sep
    act_s, act_p = ep.act[:, TRACKED].astype(np.int64), ep.act[:, PARTNER].astype(np.int64)

    opposed = (s != 0) & (p != 0) & (s != p)
    # The agent is moving along the arc towards the partner (shrinking it).
    closing = opposed & (sep != 0) & (np.sign(s) == np.sign(sep)) & (absep <= 3)
    contention_rate = float(closing.mean()) if T else 0.0

    onsets = np.flatnonzero(closing & ~np.concatenate([[False], closing[:-1]]))
    n_self = n_part = 0
    for t0 in onsets:
        d0s, d0p = s[t0], p[t0]
        t_s = t_p = None
        for t in range(t0 + 1, min(T, t0 + _RESOLVE_WINDOW + 1)):
            if t_s is None and s[t] == -d0s:
                t_s = t
            if t_p is None and p[t] == -d0p:
                t_p = t
        if t_s is not None and (t_p is None or t_s < t_p):
            n_self += 1
        elif t_p is not None:
            n_part += 1
    contention_giveway_rate = n_self / (n_self + n_part) if (n_self + n_part) else 0.0

    # Anticipatory: partner closing from >= 3 cells away, agent still opposed.
    approach = ((p != 0) & (np.sign(p) == -np.sign(sep)) & (sep != 0)
                & (absep >= 3) & (s != 0) & (s != p))
    ai = np.flatnonzero(approach)
    n_rev = 0
    for t0 in ai:
        if any(s[t] == -s[t0] for t in range(t0 + 1, min(T, t0 + _ANTICIPATE_WINDOW + 1))):
            n_rev += 1
    anticipatory_avoid_rate = n_rev / ai.size if ai.size else 0.0

    adjacent = absep == 1
    blocked_self = np.isin(act_s, _MOVE_ACTIONS) & (s == 0) & adjacent
    blocked_part = np.isin(act_p, _MOVE_ACTIONS) & (p == 0) & adjacent
    partner_block_rate = float(blocked_self.mean()) if T else 0.0
    block_asymmetry = float(blocked_part.mean() - blocked_self.mean()) if T else 0.0

    both = (s != 0) & (p != 0)
    move_dir_mi = _mi_excess(s[both], p[both], 2)
    prox = both & (absep <= _PROX_SEP)
    move_dir_mi_prox = _mi_excess(s[prox], p[prox], 2)

    lag_s = (s[1:] != 0) & (p[:-1] != 0)
    move_dir_mi_lag_self = _mi_excess(s[1:][lag_s], p[:-1][lag_s], 2)
    lag_p = (p[1:] != 0) & (s[:-1] != 0)
    move_dir_mi_lag_partner = _mi_excess(p[1:][lag_p], s[:-1][lag_p], 2)

    route_phase_mi = 0.0
    if T >= _PHASE_WINDOW:
        kern = np.ones(_PHASE_WINDOW)
        ph_s = np.sign(np.convolve(s, kern, "valid"))
        ph_p = np.sign(np.convolve(p, kern, "valid"))
        q = (ph_s != 0) & (ph_p != 0)
        route_phase_mi = _mi_excess(ph_s[q], ph_p[q], 2)

    action_mi_all = _mi_excess(act_s, act_p, _N_ACTIONS)
    pa = absep <= _PROX_SEP
    action_mi_prox = _mi_excess(act_s[pa], act_p[pa], _N_ACTIONS)

    return np.array([
        contention_rate,
        contention_giveway_rate,
        anticipatory_avoid_rate,
        partner_block_rate,
        block_asymmetry,
        move_dir_mi,
        move_dir_mi_prox,
        move_dir_mi_lag_self,
        move_dir_mi_lag_partner,
        route_phase_mi,
        action_mi_all,
        action_mi_prox,
    ], dtype=np.float64)
