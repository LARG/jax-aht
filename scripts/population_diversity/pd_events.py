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

OVERCOOKED_FEATURE_NAMES = list(OVERCOOKED_SHAPED_INFOS)
