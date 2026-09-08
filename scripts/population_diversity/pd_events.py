# PD event vocabularies for hanabi, lbf, overcooked.
# hanabi includes canaan et al. 2020 communicativeness + ipp ratios.
from __future__ import annotations
from typing import List


# hanabi events
HANABI_BASE_FEATURE_NAMES = [
    "play_legal",
    "play_bomb",
    "play_completes_color",
    "discard",
    "discard_at_max_info",
    "hint_color",
    "hint_rank",
    "hint_touches_playable",
    "hint_to_zero_info",
    "info_token_full_idle",
    "life_lost_event",
    "terminal_3_strikes",
    # Canaan et al. 2020 (AAMAS) canonical ratios.
    "communicativeness",
    "ipp",
]


def hanabi_feature_names(num_colors: int, num_ranks: int) -> List[str]:
    """variant-specific hanabi feature names."""
    color_features = [f"play_color_{c}" for c in range(num_colors)]
    rank_features = [f"play_rank_{r}" for r in range(num_ranks)]
    return (
        list(HANABI_BASE_FEATURE_NAMES)
        + color_features
        + rank_features
    )


HANABI_FEATURE_NAMES = hanabi_feature_names(5, 5)

# lbf events
LBF_FEATURE_NAMES = [
    "successful_load_alone",
    "successful_load_cooperative",
    "failed_load",
    "approach_fruit",
    "retreat_from_fruit",
    "collision_with_partner",
    "noop",
    # Per-fruit-level loads.
    "load_lvl_1",
    "load_lvl_2",
    "load_lvl_3",
    # Partner-distance state distributions.
    "state_partner_adjacent",
    "state_partner_mid",
    "state_partner_far",
    # Coordination-behavior events.
    "wait_for_partner",         # adjacent to fruit, partner not adjacent, took noop
    "target_conflict",          # both agents move toward the same nearest fruit
    "solo_attempt_lvl_2",       # tried LOAD alone on a lvl-2 fruit
    "solo_attempt_lvl_3",       # tried LOAD alone on a lvl-3 fruit
    "coop_load_lvl_2",          # successful cooperative load of a lvl-2 fruit
    "coop_load_lvl_3",          # successful cooperative load of a lvl-3 fruit
    # Spatial visitation (steps spent in each quadrant).
    "quadrant_NW_visit",
    "quadrant_NE_visit",
    "quadrant_SW_visit",
    "quadrant_SE_visit",
    # Tempo.
    "early_game_load",          # successful load in first third of episode
    "late_game_load",           # successful load in last third
    # Partner-distance + opportunity.
    "partner_distance_sum",     # cumulative Manhattan distance to partner
    "noop_when_food_visible",   # idled while >= 1 uneaten fruit on map
    # Relative position w.r.t. the partner (fraction of steps).
    "rel_north_frac",           # strictly north of partner (smaller row)
    "rel_south_frac",
    "rel_east_frac",            # strictly east of partner (larger col)
    "rel_west_frac",
    "territory_overlap",        # Jaccard of visited-cell sets with the partner
    "arrival_first_rate",       # frac of coop loads where the tracked agent arrived first
    "mean_wait_at_fruit",       # mean loiter-at-fruit run length, / episode length
    "steps_per_load",           # steps per successful load by the tracked agent
    "path_efficiency",          # Manhattan-optimal / actual move count over inter-load segments
    # Sabotage (force_coop): a "join opportunity" is partner adjacent to a fruit it
    # cannot load alone while the tracked agent is not adjacent.
    "decline_to_join_rate",     # frac of join opportunities where distance did not shrink
    "abandon_partner_rate",     # frac where the agent moved strictly away
    "join_latency",             # mean steps partner waited before the agent joined, / ep_len
]

# overcooked events (SHAPED_INFOS)
OVERCOOKED_SHAPED_INFOS = [
    "put_onion_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "SOUP_PICKUP",
    "PLACEMENT_IN_POT",
    "delivery",
    "STAY",
    "MOVEMENT",
    "IDLE_MOVEMENT",
    "IDLE_INTERACT_X",
    "IDLE_INTERACT_EMPTY",
]

# Extended overcooked vocabulary, appended after the shaped_infos block.
OVERCOOKED_DERIVED_FEATURE_NAMES = [
    # Handoffs via counters.
    "handoff_given",            # tracked agent placed, partner later picked up
    "handoff_received",         # partner placed, tracked agent later picked up
    "dead_drop",                # tracked agent placed, still there at episode end
    "self_retrieve",            # tracked agent placed and picked back up itself
    "counter_entropy",          # normalized entropy of placements over counters
    "modal_counter_frac",       # fraction of placements on the most-used cell; 0.0 with no placements
    # Item-conditioned counter identity (onion vs plate-family); 0.0 when no such placements.
    "counter_item_mi",          # Miller-Madow I(placement cell; item type) / log(min(#cells, #items))
    "onion_counter_x",          # x of the most-used onion placement cell, / (W - 1)
    "onion_counter_y",          # y of the most-used onion placement cell, / (H - 1)
    "dish_counter_x",           # x of the most-used plate/soup placement cell, / (W - 1)
    "dish_counter_y",           # y of the most-used plate/soup placement cell, / (H - 1)
    # Spatial / territorial.
    "region_NW_frac",
    "region_NE_frac",
    "region_SW_frac",
    "territory_overlap",        # Jaccard of visited tiles with the partner
    # Role specialization.
    "onion_role_frac",
    # Partner-relational.
    "mean_partner_distance",
    "interact_into_partner",
    "blocked_partner_steps",
    # Layout-portable convention features.
    "counter_item_dwell",       # median steps a tracked-agent counter drop sits before pickup, / ep_len
    "circulation_direction",    # net signed rotation around the layout centroid, in [-1, 1]
    # Normalized usage entropy over instances of each resource kind; 0.0 when < 2 instances or unused.
    "pot_usage_entropy",
    "onion_pile_usage_entropy",
    "plate_pile_usage_entropy",
    "goal_usage_entropy",
    # Role composition (same denominator as onion_role_frac); 0.0 for an idle agent.
    "pot_role_frac",            # PLACEMENT_IN_POT / n_role
    "plating_role_frac",        # (dish pickups from D or X + SOUP_PICKUP) / n_role
    # Station choice; 0.0 on layouts with < 2 instances of the kind or when unused.
    "modal_pot_x",              # x of the tracked agent's most-used pot, / (W - 1)
    "modal_pot_y",              # y of the tracked agent's most-used pot, / (H - 1)
    "modal_onion_pile_x", "modal_onion_pile_y",
    "modal_plate_pile_x", "modal_plate_pile_y",
    "modal_goal_x", "modal_goal_y",
    "pot_sharing",              # sum_i min(p_i, q_i) over pots: tracked vs partner usage distributions
    "onion_pile_sharing",
    "plate_pile_sharing",
    "goal_sharing",
]

OVERCOOKED_FEATURE_NAMES = list(OVERCOOKED_SHAPED_INFOS) + list(
    OVERCOOKED_DERIVED_FEATURE_NAMES
)

# Partner-contingent features (pd_overcooked_features.episode_contingency_features).
OVERCOOKED_CONTINGENT_FEATURE_NAMES = [
    "route_partner_mi",
    "route_partner_phi",
    "switch_encounter_lift",
    "region_conditional_lift",
    "yield_rate",
]

# Contention and action-MI features (pd_overcooked_features.episode_v5_features).
OVERCOOKED_V5_FEATURE_NAMES = [
    "contention_rate",
    "contention_giveway_rate",
    "anticipatory_avoid_rate",
    "partner_block_rate",
    "block_asymmetry",
    "move_dir_mi",
    "move_dir_mi_prox",
    "move_dir_mi_lag_self",
    "move_dir_mi_lag_partner",
    "route_phase_mi",
    "action_mi_all",
    "action_mi_prox",
]
