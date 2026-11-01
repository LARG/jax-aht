"""Semantic feature groups and derived routing fractions for the Overcooked convention vocabulary."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

# Tracked agent's own per-episode counts, not the team score.
THROUGHPUT = [
    "delivery",
    "PLACEMENT_IN_POT",
]

COUNTER_USE = [
    # routing fractions from derive()
    "onion_via_counter", "dish_via_counter", "soup_via_counter",
    "onion_from_counter", "dish_from_counter", "soup_from_counter",
    # drop fate (self-retrieve implied) and pickup origin
    "handoff_given_frac", "dead_drop_frac", "handoff_received_frac",
    "counter_item_dwell",
]

COUNTER_IDENTITY = [
    "counter_entropy", "modal_counter_frac", "counter_item_mi",
    "onion_counter_x", "onion_counter_y",
    "dish_counter_x", "dish_counter_y",
]

MOVEMENT = [
    "region_NW_frac", "region_NE_frac", "region_SW_frac",
    "territory_overlap", "circulation_direction",
    "STAY", "MOVEMENT", "IDLE_MOVEMENT", "IDLE_INTERACT_X", "IDLE_INTERACT_EMPTY",
]

ROLE = [
    "onion_role_frac", "pot_role_frac", "plating_role_frac",
    "pot_usage_entropy", "onion_pile_usage_entropy",
    "plate_pile_usage_entropy", "goal_usage_entropy",
    # favoured station instance
    "modal_pot_x", "modal_pot_y", "modal_onion_pile_x", "modal_onion_pile_y",
    "modal_plate_pile_x", "modal_plate_pile_y", "modal_goal_x", "modal_goal_y",
]

PARTNER_COUPLING = [
    "mean_partner_distance",
    # shared resource instances
    "pot_sharing", "onion_pile_sharing", "plate_pile_sharing", "goal_sharing",
    "route_partner_mi", "route_partner_phi", "switch_encounter_lift",
    "region_conditional_lift", "yield_rate",
    "move_dir_mi", "move_dir_mi_prox", "move_dir_mi_lag_self", "move_dir_mi_lag_partner",
    "route_phase_mi", "action_mi_all", "action_mi_prox",
]

CONTENTION = [
    "interact_into_partner", "blocked_partner_steps",
    "contention_rate", "contention_giveway_rate", "anticipatory_avoid_rate",
    "partner_block_rate", "block_asymmetry",
]

# Coupling features defined only when the agents are within a few cells of each other.
PROXIMITY_CONDITIONED = ["yield_rate", "move_dir_mi_prox", "action_mi_prox", "switch_encounter_lift"]

# Raw counts consumed by derive(); never clustered.
DERIVE_INPUTS = [
    "SOUP_PICKUP",
    "put_onion_on_X", "put_dish_on_X", "put_soup_on_X",
    "pickup_onion_from_X", "pickup_dish_from_X", "pickup_soup_from_X",
    "pickup_onion_from_O", "pickup_dish_from_D",
    "handoff_given", "handoff_received", "dead_drop", "self_retrieve",
]

# Numerator raw count of each DERIVED ratio (minimum-support screen).
RATIO_NUMERATOR = {
    "onion_via_counter": "put_onion_on_X", "dish_via_counter": "put_dish_on_X",
    "soup_via_counter": "put_soup_on_X", "onion_from_counter": "pickup_onion_from_X",
    "dish_from_counter": "pickup_dish_from_X", "soup_from_counter": "pickup_soup_from_X",
    "handoff_given_frac": "handoff_given", "dead_drop_frac": "dead_drop",
    "handoff_received_frac": "handoff_received",
}
MIN_EVENT_SUPPORT = 0.01  # population-mean numerator events per episode

DERIVED = [
    "onion_via_counter", "dish_via_counter", "soup_via_counter",
    "onion_from_counter", "dish_from_counter", "soup_from_counter",
    "handoff_given_frac", "dead_drop_frac", "handoff_received_frac",
]


def _ratio(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def derive(df):
    """Return a copy of df with the DERIVED routing-fraction columns added."""
    d = df.copy()
    g = lambda c: d[c].to_numpy(dtype=np.float64)  # noqa: E731
    d["onion_via_counter"] = _ratio(g("put_onion_on_X"), g("put_onion_on_X") + g("PLACEMENT_IN_POT"))
    d["dish_via_counter"] = _ratio(g("put_dish_on_X"), g("put_dish_on_X") + g("SOUP_PICKUP"))
    d["soup_via_counter"] = _ratio(g("put_soup_on_X"), g("put_soup_on_X") + g("delivery"))
    d["onion_from_counter"] = _ratio(g("pickup_onion_from_X"), g("pickup_onion_from_X") + g("pickup_onion_from_O"))
    d["dish_from_counter"] = _ratio(g("pickup_dish_from_X"), g("pickup_dish_from_X") + g("pickup_dish_from_D"))
    d["soup_from_counter"] = _ratio(g("pickup_soup_from_X"), g("pickup_soup_from_X") + g("SOUP_PICKUP"))
    drops = g("handoff_given") + g("dead_drop") + g("self_retrieve")
    d["handoff_given_frac"] = _ratio(g("handoff_given"), drops)
    d["dead_drop_frac"] = _ratio(g("dead_drop"), drops)
    picks = g("pickup_onion_from_X") + g("pickup_dish_from_X") + g("pickup_soup_from_X")
    d["handoff_received_frac"] = _ratio(g("handoff_received"), picks)
    return d


def groups() -> Dict[str, List[str]]:
    return {
        "throughput": list(THROUGHPUT),
        "counter_use": list(COUNTER_USE),
        "counter_identity": list(COUNTER_IDENTITY),
        "movement": list(MOVEMENT),
        "role": list(ROLE),
        "partner_coupling": list(PARTNER_COUPLING),
        "contention": list(CONTENTION),
    }


def group_of(feature_cols: Iterable[str]) -> Dict[str, str]:
    """Map feature -> group name; raises on an ungrouped feature."""
    inv: Dict[str, str] = {}
    for name, cols in groups().items():
        for c in cols:
            inv[c] = name
    missing = [c for c in feature_cols if c not in inv]
    if missing:
        raise KeyError(f"features with no a-priori group (add them to feature_groups.py): {missing}")
    return {c: inv[c] for c in feature_cols}
