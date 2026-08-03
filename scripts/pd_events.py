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
    # Per-fruit-level loads (extension to lift feature dimension above heldout
    # population size, Wang et al. 2024 framework adapted to LBF).
    "load_lvl_1",
    "load_lvl_2",
    "load_lvl_3",
    # Partner-distance state distributions
    "state_partner_adjacent",
    "state_partner_mid",
    "state_partner_far",
    # Coordination-behavior events (extending vocab to lift D above heldout N).
    "wait_for_partner",         # adjacent to fruit, partner not adjacent, took noop
    "target_conflict",          # both agents move toward the same nearest fruit
    "solo_attempt_lvl_2",       # tried LOAD alone on a lvl-2 fruit (succeeded or failed)
    "solo_attempt_lvl_3",       # tried LOAD alone on a lvl-3 fruit
    "coop_load_lvl_2",          # successful cooperative load of a lvl-2 fruit
    "coop_load_lvl_3",          # successful cooperative load of a lvl-3 fruit
    # Spatial visitation (4-quadrant; integer count of steps spent in each).
    "quadrant_NW_visit",
    "quadrant_NE_visit",
    "quadrant_SW_visit",
    "quadrant_SE_visit",
    # Tempo: early- vs late-game load events.
    "early_game_load",          # successful load in first third of episode
    "late_game_load",           # successful load in last third
    # Partner-distance + opportunity.
    "partner_distance_sum",     # cumulative Manhattan distance to partner
    "noop_when_food_visible",   # idled while >= 1 uneaten fruit on map
    # --- extended vocabulary (all normalized to [0, 1]-ish scales) ---
    # Relative spatial configuration w.r.t. the partner (fraction of steps).
    "rel_north_frac",           # tracked agent strictly north of partner (smaller row)
    "rel_south_frac",
    "rel_east_frac",            # strictly east of partner (larger col)
    "rel_west_frac",
    # Territorial overlap: Jaccard of visited-cell sets (tracked agent vs partner).
    "territory_overlap",
    # Among cooperatively-loaded fruits, fraction where the tracked agent arrived first.
    "arrival_first_rate",
    # Mean length of a "loitering next to fruit without loading" run, / episode length.
    "mean_wait_at_fruit",
    # Tempo: steps per successful load by the tracked agent.
    "steps_per_load",
    # Manhattan-optimal / actual move count, averaged over inter-load segments.
    "path_efficiency",
    # --- sabotage vocabulary (force_coop LBF: every fruit needs BOTH agents) ---
    # A "join opportunity" = partner adjacent to a fruit it cannot load alone, tracked
    # agent NOT adjacent to it. Cooperating means closing distance; declining is the
    # LBF form of sabotage. Rates are conditioned on an opportunity existing.
    "decline_to_join_rate",     # fraction of join opportunities where distance did not shrink
    "abandon_partner_rate",     # fraction where the agent moved strictly AWAY (active refusal)
    "join_latency",             # mean steps partner waited before the agent joined, / ep_len
]

# overcooked events (wang+2024 SHAPED_INFOS)
OVERCOOKED_SHAPED_INFOS = [
    "put_onion_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",
    "SOUP_PICKUP",
    "PLACEMENT_IN_POT",
    "delivery",
    "STAY",
    "MOVEMENT",
    "IDLE_MOVEMENT",
    "IDLE_INTERACT_X",
    "IDLE_INTERACT_EMPTY",
]

# Extended overcooked vocabulary. Appended AFTER the shaped_infos block so the
# original 17 columns keep their names and positions.
OVERCOOKED_EXTENDED_FEATURE_NAMES = [
    # Handoff: who put what on which counter, and who took it back off.
    "handoff_given",            # tracked agent placed, partner later picked up
    "handoff_received",         # partner placed, tracked agent later picked up
    "dead_drop",                # tracked agent placed, still there at episode end
    "self_retrieve",            # tracked agent placed and picked back up itself
    "counter_entropy",          # normalized entropy of placements over counters
    # Resource choice: fraction of uses that went to the canonically-first
    # instance of each resource. 0.0 when the layout has < 2 of that resource.
    "pot_first_frac",
    "onion_pile_first_frac",
    "plate_pile_first_frac",
    "goal_first_frac",
    # Spatial / territorial.
    "region_NW_frac",
    "region_NE_frac",
    "region_SW_frac",
    "region_SE_frac",
    "territory_overlap",        # Jaccard of visited tiles with the partner
    # Role specialization.
    "onion_role_frac",
    "role_purity",
    # Delivery tempo.
    "deliveries_per_100_steps",
    "mean_inter_delivery_interval",
    # Partner-relational.
    "mean_partner_distance",
    "interact_into_partner",
    "blocked_partner_steps",
]

OVERCOOKED_FEATURE_NAMES = list(OVERCOOKED_SHAPED_INFOS) + list(
    OVERCOOKED_EXTENDED_FEATURE_NAMES
)
