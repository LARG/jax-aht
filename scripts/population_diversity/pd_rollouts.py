# per-env rollout loops, episode-event counters, and BR loading.
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from scripts.population_diversity.pd_events import (
    HANABI_BASE_FEATURE_NAMES,
    hanabi_feature_names,
    HANABI_FEATURE_NAMES,
    LBF_FEATURE_NAMES,
    OVERCOOKED_SHAPED_INFOS,
    OVERCOOKED_FEATURE_NAMES,
)

log = logging.getLogger("compute_pd")

REPO_ROOT = Path(__file__).resolve().parents[1]

def _unwrap_hanabi_state(env_state):
    s = env_state
    # LogWrapper wraps as env_state.env_state; WrappedEnvState too.
    while hasattr(s, "env_state"):
        s = s.env_state
    return s


def _true_post_state(env, pre_state, env_act, key):
    # env.step auto-resets on done; get the real post-step state from the inner env (no reset)
    wrapper = env._env
    inner = wrapper.env
    inner_pre = _unwrap_hanabi_state(pre_state)
    if hasattr(inner, "step_env"):
        _obs, post, _r, _d, _i = inner.step_env(key, inner_pre, env_act)   # jaxmarl: dict acts + key
    else:
        actions_array = jnp.array([env_act[a] for a in wrapper.agents], dtype=jnp.int32)
        post, _timestep = inner.step(inner_pre, actions_array)             # jumanji: act array, no key
    return post


def state_info_tokens(env_state) -> int:
    s = _unwrap_hanabi_state(env_state)
    return int(jnp.sum(s.info_tokens))


def state_life_tokens(env_state) -> int:
    s = _unwrap_hanabi_state(env_state)
    return int(jnp.sum(s.life_tokens))


def state_fireworks_sum(env_state) -> int:
    s = _unwrap_hanabi_state(env_state)
    return int(jnp.sum(s.fireworks))


def hint_touches_playable(env_state, action: int, layout) -> bool:
    """true if the hint action touches a currently-playable card in partner's hand."""
    s = _unwrap_hanabi_state(env_state)
    # decode action: hint_color uses [2H .. 2H+C-1]; hint_rank uses [2H+C .. 2H+C+R-1]
    cat, target_idx = layout.categorize(action)
    if cat not in ("hint_color", "hint_rank"):
        return False
    # partner's hand; cur_player is the actor, partner is 1 - cur_player
    cur_player = int(s.cur_player_idx.argmax()) if hasattr(s.cur_player_idx, "argmax") else int(s.cur_player_idx)
    partner = 1 - cur_player
    partner_hand = s.player_hands[partner]  # shape: (hand_size, num_colors, num_ranks)
    # next_playable_rank[c] = number of cards already on color-c firework
    fireworks = s.fireworks  # (num_colors, num_ranks)
    next_playable_rank = jnp.sum(fireworks, axis=1).astype(jnp.int32)  # (num_colors,)
    # for each card in partner's hand: if the hint touches it and it's currently playable, count it
    any_touched_and_playable = False
    for slot in range(layout.hand_size):
        card = partner_hand[slot]  # (num_colors, num_ranks)
        if jnp.sum(card) == 0:
            continue  # empty slot
        color = int(jnp.argmax(jnp.sum(card, axis=1)))
        rank = int(jnp.argmax(jnp.sum(card, axis=0)))
        # does the hint touch this card?
        touched = (cat == "hint_color" and color == target_idx) or (cat == "hint_rank" and rank == target_idx)
        if touched and rank == int(next_playable_rank[color]):
            any_touched_and_playable = True
            break
    return bool(any_touched_and_playable)


@dataclass
class HanabiActionLayout:
    hand_size: int
    num_colors: int
    num_ranks: int
    num_actions: int

    def categorize(self, action: int) -> Tuple[str, int]:
        """map an action index to (category, within-category index)."""
        if action < self.hand_size:
            return ("discard", action)
        if action < 2 * self.hand_size:
            return ("play", action - self.hand_size)
        hc_end = 2 * self.hand_size + self.num_colors
        if action < hc_end:
            return ("hint_color", action - 2 * self.hand_size)
        hr_end = hc_end + self.num_ranks
        if action < hr_end:
            return ("hint_rank", action - hc_end)
        # action == 2H+C+R == num_moves-1: the forced NOOP of the non-actor.
        return ("noop", -1)


@dataclass
class HanabiEpisodeCounts:
    """per-episode hanabi event counts."""
    n_play_legal: int = 0          # play with fireworks count incremented
    n_play_bomb: int = 0           # play with life_token decremented
    n_play_completes_color: int = 0  # play of rank-5 (added_info_token=1)
    n_discard: int = 0             # any discard action
    n_discard_at_max_info: int = 0  # discard when info_tokens already at max (wasteful)
    n_hint_color: int = 0
    n_hint_rank: int = 0
    n_hint_touches_playable: int = 0  # hint reveals card(s) currently playable
    n_hint_to_zero_info: int = 0   # hint that brought info_tokens to 0
    n_info_token_full_idle: int = 0  # state with info=max and no hint given
    n_life_lost_event: int = 0     # life_token decremented this step
    terminal_3_strikes: int = 0    # episode ended via life_tokens=0 (0 or 1)
    # per-color and per-rank breakdowns of legal plays, sized at episode start from env num_colors/num_ranks
    n_play_per_color: List[int] = field(default_factory=list)
    n_play_per_rank: List[int] = field(default_factory=list)
    # Canaan et al. 2020 (AIIDE) canonical denominators/numerators
    # communicativeness: hints given by acting player / turns where a hint was available
    # IPP: total bits of info known about played cards / number of plays
    n_hint_available_turns: int = 0  # acting-player turns with info_tokens > 0
    n_acting_hint_given: int = 0     # acting-player hint actions (numerator for communicativeness)
    n_play_actions: int = 0          # acting-player play actions (denominator for IPP)
    sum_play_info_known: float = 0.0  # bits revealed about played card slot, summed over plays
    final_score: float = 0.0
    episode_length: int = 0



def hanabi_episode_to_vector(
    counts: HanabiEpisodeCounts, score_norm: float = 0.0, length_norm: float = 0.0
) -> np.ndarray:
    """per-episode hanabi counts to the theta contribution vector."""
    communicativeness = counts.n_acting_hint_given / max(counts.n_hint_available_turns, 1)
    ipp = counts.sum_play_info_known / max(counts.n_play_actions, 1)
    base = [
        float(counts.n_play_legal),
        float(counts.n_play_bomb),
        float(counts.n_play_completes_color),
        float(counts.n_discard),
        float(counts.n_discard_at_max_info),
        float(counts.n_hint_color),
        float(counts.n_hint_rank),
        float(counts.n_hint_touches_playable),
        float(counts.n_hint_to_zero_info),
        float(counts.n_info_token_full_idle),
        float(counts.n_life_lost_event),
        float(counts.terminal_3_strikes),
        float(communicativeness),
        float(ipp),
    ]
    per_color = [float(x) for x in counts.n_play_per_color]
    per_rank = [float(x) for x in counts.n_play_per_rank]
    return np.array(base + per_color + per_rank, dtype=np.float64)


def build_hanabi_agents(
    hand_size: int, num_colors: int, num_ranks: int, num_actions: int, card_counts: np.ndarray
) -> Dict[str, Any]:
    """build the 8 hanabi heuristics for self-play."""
    from agents.hanabi.agent_policy_wrappers import (
        HanabiCautiousPolicyWrapper,
        HanabiFlawedPolicyWrapper,
        HanabiIGGIPolicyWrapper,
        HanabiInternalPolicyWrapper,
        HanabiOuterPolicyWrapper,
        HanabiPiersPolicyWrapper,
        HanabiSmartBotPolicyWrapper,
        HanabiVanDenBerghPolicyWrapper,
    )

    common = dict(
        hand_size=hand_size,
        num_colors=num_colors,
        num_ranks=num_ranks,
        num_actions=num_actions,
        using_log_wrapper=True,
    )
    return {
        "iggi": HanabiIGGIPolicyWrapper(**common),
        "piers": HanabiPiersPolicyWrapper(**common),
        "van_den_bergh": HanabiVanDenBerghPolicyWrapper(**common),
        "outer": HanabiOuterPolicyWrapper(**common),
        "flawed": HanabiFlawedPolicyWrapper(play_threshold=0.4, **common),
        "cautious": HanabiCautiousPolicyWrapper(**common),
        "internal": HanabiInternalPolicyWrapper(**common),
        "smartbot": HanabiSmartBotPolicyWrapper(card_counts=card_counts, **common),
    }


def rollout_hanabi_two_policy(
    env,
    policy_a, params_a,
    policy_b, params_b,
    layout: HanabiActionLayout,
    num_episodes: int,
    seed: int,
    max_steps: int = 200,
) -> List[HanabiEpisodeCounts]:
    """roll out n episodes pairing policy_a (agent_0) with policy_b (agent_1)."""
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    init_act_onehot = {
        k: jnp.zeros((env.action_space(env.agents[i]).n))
        for i, k in enumerate(env.agents)
    }
    init_reward = {k: jnp.zeros((1)) for k in env.agents}

    rng_master = jax.random.PRNGKey(seed)
    results: List[HanabiEpisodeCounts] = []

    for ep_idx in range(num_episodes):
        rng_master, ep_rng = jax.random.split(rng_master)
        ep_rng, reset_rng = jax.random.split(ep_rng)
        obs, env_state = env.reset(reset_rng)

        hstate_0 = policy_a.init_hstate(1, aux_info={"agent_id": 0})
        hstate_1 = policy_b.init_hstate(1, aux_info={"agent_id": 1})

        act_onehot_prev = init_act_onehot
        reward_prev = init_reward
        done_prev = init_done
        counts = HanabiEpisodeCounts(
            n_play_per_color=[0] * layout.num_colors,
            n_play_per_rank=[0] * layout.num_ranks,
        )
        episode_score = 0.0

        for step_i in range(max_steps):
            avail = env.get_avail_actions(env_state)
            avail = jax.lax.stop_gradient(avail)
            avail_0 = avail["agent_0"].astype(jnp.float32)
            avail_1 = avail["agent_1"].astype(jnp.float32)
            joint_act_oh = jnp.concatenate(
                (
                    act_onehot_prev["agent_0"].reshape(1, 1, -1),
                    act_onehot_prev["agent_1"].reshape(1, 1, -1),
                ),
                axis=-1,
            )

            ep_rng, a0_rng, a1_rng, step_rng = jax.random.split(ep_rng, 4)
            act_0, hstate_0 = policy_a.get_action(
                params=params_a,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done_prev["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=a0_rng,
                aux_obs=(
                    act_onehot_prev["agent_0"].reshape(1, 1, -1),
                    joint_act_oh,
                    reward_prev["agent_0"].reshape(1, 1, -1),
                ),
                env_state=env_state,
                test_mode=False,
            )
            act_0 = int(act_0.squeeze())
            act_1, hstate_1 = policy_b.get_action(
                params=params_b,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=done_prev["agent_1"].reshape(1, 1),
                avail_actions=avail_1,
                hstate=hstate_1,
                rng=a1_rng,
                aux_obs=None,
                env_state=env_state,
                test_mode=False,
            )
            act_1 = int(act_1.squeeze())

            # snapshot pre-step state
            old_info_tokens = int(state_info_tokens(env_state))
            old_life_tokens = int(state_life_tokens(env_state))
            old_fireworks_sum = int(state_fireworks_sum(env_state))
            info_at_max_pre = old_info_tokens == int(env.env.max_info_tokens)

            # who's acting this turn (only the actor moves; the other is forced to noop)
            try:
                s_pre = _unwrap_hanabi_state(env_state)
                cur_player_pre = int(jnp.argmax(s_pre.cur_player_idx))
            except Exception:
                cur_player_pre = 0  # fallback
            # count the BR only (agent_1), on its own turns, per ZSC-Eval
            BR_PLAYER = 1
            br_turn = cur_player_pre == BR_PLAYER
            acting_action = act_0 if cur_player_pre == 0 else act_1

            # hint/discard from the BR's action; plays come from the state delta post-step
            if br_turn:
                cat, slot_idx = layout.categorize(acting_action)
                if cat == "hint_color":
                    counts.n_hint_color += 1
                    # did this hint touch a currently-playable card in partner's hand?
                    if hint_touches_playable(env_state, acting_action, layout):
                        counts.n_hint_touches_playable += 1
                    if old_info_tokens - 1 == 0:
                        counts.n_hint_to_zero_info += 1
                elif cat == "hint_rank":
                    counts.n_hint_rank += 1
                    if hint_touches_playable(env_state, acting_action, layout):
                        counts.n_hint_touches_playable += 1
                    if old_info_tokens - 1 == 0:
                        counts.n_hint_to_zero_info += 1
                elif cat == "discard":
                    counts.n_discard += 1
                    # n_discard_at_max_info stays 0: discard is illegal at max info. kept constant so the vector length matches older runs.
                # play handled post-step; noop ignored

            # BR at max info but didn't hint = wasted hint-token turn
            if br_turn and info_at_max_pre:
                acting_cat, _ = layout.categorize(acting_action)
                if acting_cat not in ("hint_color", "hint_rank"):
                    counts.n_info_token_full_idle += 1

            env_act = {"agent_0": act_0, "agent_1": act_1}
            env_state_pre = env_state  # snapshot for post-step per-color/rank
            obs, env_state, reward, done, _info = env.step(step_rng, env_state, env_act)
            step_reward = float(reward["agent_0"])
            episode_score += step_reward

            # on the last step, read deltas off the true terminal state, not the auto-reset
            post_state = _true_post_state(env, env_state_pre, env_act, step_rng) if bool(done["__all__"]) else env_state
            # post-step deltas: play_legal / play_bomb / play_completes_color / life_lost
            new_info_tokens = int(state_info_tokens(post_state))
            new_life_tokens = int(state_life_tokens(post_state))
            new_fireworks_sum = int(state_fireworks_sum(post_state))
            fireworks_inc = new_fireworks_sum > old_fireworks_sum
            life_dec = new_life_tokens < old_life_tokens

            # Canaan 2020 communicativeness denominator: BR turns where a hint was available
            if br_turn and old_info_tokens > 0:
                counts.n_hint_available_turns += 1

            # attribute the play/IPP events to the BR, only on the BR's turn
            cat, slot_idx = layout.categorize(acting_action)
            if br_turn and cat in ("hint_color", "hint_rank"):
                # communicativeness numerator: BR hints only
                counts.n_acting_hint_given += 1
            if br_turn and cat == "play":
                # Canaan et al. 2020 IPP: bits revealed about the played card's slot, from pre-step colors_revealed + ranks_revealed on the unwrapped state (s_pre, not env_state_pre)
                counts.n_play_actions += 1
                try:
                    bits_color = float(jnp.sum(s_pre.colors_revealed[cur_player_pre, slot_idx]))
                    bits_rank = float(jnp.sum(s_pre.ranks_revealed[cur_player_pre, slot_idx]))
                    counts.sum_play_info_known += bits_color + bits_rank
                except Exception:
                    pass  # skip IPP contribution if state unreadable
                if fireworks_inc:
                    counts.n_play_legal += 1
                    # rank-5 play = added info token
                    if new_info_tokens > old_info_tokens:
                        counts.n_play_completes_color += 1
                    # per-color/per-rank breakdown of legal plays, from the pre-step state
                    try:
                        card = s_pre.player_hands[cur_player_pre][slot_idx]
                        color = int(jnp.argmax(jnp.sum(card, axis=1)))
                        rank = int(jnp.argmax(jnp.sum(card, axis=0)))
                        if 0 <= color < layout.num_colors:
                            counts.n_play_per_color[color] += 1
                        if 0 <= rank < layout.num_ranks:
                            counts.n_play_per_rank[rank] += 1
                    except Exception:
                        pass  # skip per-color/rank if state unreadable
                if life_dec:
                    counts.n_play_bomb += 1
            # life_lost is the actor's bomb; attribute to the BR only when the BR is the actor (keeps event attribution on the BR)
            if br_turn and life_dec:
                counts.n_life_lost_event += 1
            # terminal_3_strikes is an episode-level outcome flag (ran out of lives), not a per-turn event, so it's not gated by br_turn
            if new_life_tokens == 0 and bool(done.get("__all__", jnp.array(False))):
                counts.terminal_3_strikes = 1

            act_onehot_prev = {
                "agent_0": jax.nn.one_hot(act_0, env.action_space("agent_0").n),
                "agent_1": jax.nn.one_hot(act_1, env.action_space("agent_1").n),
            }
            reward_prev = reward
            done_prev = done
            if bool(done["__all__"]):
                break

        counts.final_score = episode_score
        counts.episode_length = step_i + 1
        results.append(counts)

    return results


def rollout_hanabi_self_play(
    env,
    policy,
    layout: HanabiActionLayout,
    num_episodes: int,
    seed: int,
    max_steps: int = 200,
    params=None,
) -> List[HanabiEpisodeCounts]:
    """self-play wrapper: pair policy with itself."""
    return rollout_hanabi_two_policy(
        env, policy, params, policy, params, layout, num_episodes, seed, max_steps=max_steps,
    )




# LBF action layout (Jumanji)
_LBF_NOOP, _LBF_N, _LBF_S, _LBF_W, _LBF_E, _LBF_LOAD = range(6)


@dataclass
class LBFEpisodeCounts:
    """per-episode lbf event counts."""
    n_successful_load_alone: int = 0
    n_successful_load_cooperative: int = 0
    n_failed_load: int = 0
    n_approach_fruit: int = 0
    n_retreat_from_fruit: int = 0
    n_collision_with_partner: int = 0
    n_noop: int = 0
    n_load_lvl_1: int = 0
    n_load_lvl_2: int = 0
    n_load_lvl_3: int = 0
    n_state_partner_adjacent: int = 0  # Manhattan distance == 1
    n_state_partner_mid: int = 0       # Manhattan distance 2-3
    n_state_partner_far: int = 0       # Manhattan distance > 3
    # wider vocab additions (D=13 -> D=28) to push above heldout N
    n_wait_for_partner: int = 0
    n_target_conflict: int = 0
    n_solo_attempt_lvl_2: int = 0
    n_solo_attempt_lvl_3: int = 0
    n_coop_load_lvl_2: int = 0
    n_coop_load_lvl_3: int = 0
    n_quadrant_NW_visit: int = 0
    n_quadrant_NE_visit: int = 0
    n_quadrant_SW_visit: int = 0
    n_quadrant_SE_visit: int = 0
    n_early_game_load: int = 0
    n_late_game_load: int = 0
    partner_distance_sum: int = 0
    n_noop_when_food_visible: int = 0
    # --- derived vocabulary: raw accumulators (see lbf_episode_to_vector) ---
    n_steps: int = 0                     # number of step updates seen this episode
    n_rel_north: int = 0                 # steps strictly north of partner (smaller row)
    n_rel_south: int = 0
    n_rel_east: int = 0                  # steps strictly east of partner (larger col)
    n_rel_west: int = 0
    n_coop_load_events: int = 0          # cooperative loads (denominator of arrival_first_rate)
    n_arrival_first: int = 0             # coop loads where the tracked agent got adjacent first
    wait_run_sum: int = 0                # summed length of finished "wait at fruit" runs
    wait_run_count: int = 0              # number of finished runs
    wait_run_cur: int = 0                # length of the run currently in progress
    path_eff_sum: float = 0.0
    path_eff_count: int = 0
    # --- sabotage vocabulary (LBF force_coop: every fruit needs BOTH agents) ---
    # "Join opportunity": the partner is adjacent to a fruit it CANNOT load alone, and
    # the tracked agent is not adjacent to that fruit. The cooperative act is to close
    # distance to it; declining is the LBF form of sabotage (refusing to approach an
    # apple a teammate is standing at).
    n_join_opportunity: int = 0          # steps presenting such an opportunity
    n_decline_to_join: int = 0           # ... where the agent did NOT reduce its distance
    n_abandon_partner: int = 0           # ... where it strictly INCREASED its distance
    join_latency_sum: int = 0            # summed steps from partner-arrival to self-arrival
    join_latency_count: int = 0          # number of resolved arrival episodes
    # (row, col) -> step at which the partner first became adjacent while self was not
    pending_join: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # --- non-feature bookkeeping (episode-level state carried across steps) ---
    visited_self: set = field(default_factory=set)
    visited_partner: set = field(default_factory=set)
    # (row, col) -> {agent_idx: first step index at which that agent was adjacent}
    first_adjacent: Dict[Tuple[int, int], Dict[int, int]] = field(default_factory=dict)
    seg_start_pos: Optional[Tuple[int, int]] = None  # position at start of current load segment
    seg_moves: int = 0                               # move actions taken in current segment
    final_score: float = 0.0
    episode_length: int = 0


def lbf_episode_to_vector(counts: LBFEpisodeCounts, return_norm: float = 0.0, length_norm: float = 0.0) -> np.ndarray:
    """per-episode lbf counts to the theta contribution vector."""
    ep_len = max(1, int(counts.n_steps) or int(counts.episode_length) or 1)

    # relative-position fractions
    rel_north = counts.n_rel_north / ep_len
    rel_south = counts.n_rel_south / ep_len
    rel_east = counts.n_rel_east / ep_len
    rel_west = counts.n_rel_west / ep_len

    # territory overlap (Jaccard); union is empty only if no steps were taken
    union = counts.visited_self | counts.visited_partner
    territory_overlap = (
        len(counts.visited_self & counts.visited_partner) / len(union) if union else 0.0
    )

    arrival_first_rate = (
        counts.n_arrival_first / counts.n_coop_load_events if counts.n_coop_load_events > 0 else 0.0
    )

    # flush the in-progress wait run without mutating `counts`
    wait_sum = counts.wait_run_sum + counts.wait_run_cur
    wait_n = counts.wait_run_count + (1 if counts.wait_run_cur > 0 else 0)
    mean_wait_at_fruit = (wait_sum / wait_n) / ep_len if wait_n > 0 else 0.0

    n_loads = counts.n_successful_load_alone + counts.n_successful_load_cooperative
    steps_per_load = ep_len / max(1, n_loads)

    # sabotage rates: conditioned on an actual join opportunity existing, so they are
    # not diluted by steps where there was nothing cooperative to do. 0.0 when the
    # opportunity never arose (distinct from "declined every time", which is 1.0).
    decline_to_join_rate = (
        counts.n_decline_to_join / counts.n_join_opportunity
        if counts.n_join_opportunity > 0 else 0.0
    )
    abandon_partner_rate = (
        counts.n_abandon_partner / counts.n_join_opportunity
        if counts.n_join_opportunity > 0 else 0.0
    )
    # mean steps to join a waiting partner, normalized by episode length (1.0 = never)
    join_latency = (
        (counts.join_latency_sum / counts.join_latency_count) / ep_len
        if counts.join_latency_count > 0 else 0.0
    )

    path_efficiency = (
        counts.path_eff_sum / counts.path_eff_count if counts.path_eff_count > 0 else 0.0
    )

    feats = [
            float(counts.n_successful_load_alone),
            float(counts.n_successful_load_cooperative),
            float(counts.n_failed_load),
            float(counts.n_approach_fruit),
            float(counts.n_retreat_from_fruit),
            float(counts.n_collision_with_partner),
            float(counts.n_noop),
            float(counts.n_load_lvl_1),
            float(counts.n_load_lvl_2),
            float(counts.n_load_lvl_3),
            float(counts.n_state_partner_adjacent),
            float(counts.n_state_partner_mid),
            float(counts.n_state_partner_far),
            float(counts.n_wait_for_partner),
            float(counts.n_target_conflict),
            float(counts.n_solo_attempt_lvl_2),
            float(counts.n_solo_attempt_lvl_3),
            float(counts.n_coop_load_lvl_2),
            float(counts.n_coop_load_lvl_3),
            float(counts.n_quadrant_NW_visit),
            float(counts.n_quadrant_NE_visit),
            float(counts.n_quadrant_SW_visit),
            float(counts.n_quadrant_SE_visit),
            float(counts.n_early_game_load),
            float(counts.n_late_game_load),
            float(counts.partner_distance_sum),
            float(counts.n_noop_when_food_visible),
            float(rel_north),
            float(rel_south),
            float(rel_east),
            float(rel_west),
            float(territory_overlap),
            float(arrival_first_rate),
            float(mean_wait_at_fruit),
            float(steps_per_load),
            float(path_efficiency),
            float(decline_to_join_rate),
            float(abandon_partner_rate),
            float(join_latency),
    ]
    assert len(feats) == len(LBF_FEATURE_NAMES), (
        f"lbf feature vector length {len(feats)} != {len(LBF_FEATURE_NAMES)} names"
    )
    return np.array(feats, dtype=np.float64)


def build_lbf_agents(grid_size: int, num_fruits: int) -> Dict[str, Any]:
    """build the lbf heuristic suite for self-play."""
    from agents.lbf.agent_policy_wrappers import (
        LBFEntitledPolicyWrapper,
        LBFGreedyHeuristicPolicyWrapper,
        LBFSequentialFruitPolicyWrapper,
    )

    common = dict(grid_size=grid_size, num_fruits=num_fruits, using_log_wrapper=True)
    return {
        "entitled": LBFEntitledPolicyWrapper(**common),
        "greedy_closest_self": LBFGreedyHeuristicPolicyWrapper(heuristic="closest_self", **common),
        "greedy_closest_teammate": LBFGreedyHeuristicPolicyWrapper(heuristic="closest_teammate", **common),
        "greedy_lowest_level": LBFGreedyHeuristicPolicyWrapper(heuristic="lowest_level", **common),
        "greedy_highest_level": LBFGreedyHeuristicPolicyWrapper(heuristic="highest_level", **common),
        "sequential_lex": LBFSequentialFruitPolicyWrapper(ordering_strategy="lexicographic", **common),
        "sequential_revlex": LBFSequentialFruitPolicyWrapper(ordering_strategy="reverse_lexicographic", **common),
        "sequential_nearest": LBFSequentialFruitPolicyWrapper(ordering_strategy="nearest_agent", **common),
    }


def rollout_two_policy(
    env,
    policy_a, params_a,
    policy_b, params_b,
    num_episodes: int,
    seed: int,
    counts_factory,
    feature_update,
    max_steps: int = 200,
) -> List[Any]:
    """sequential rollout with two distinct policies, one per agent."""
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    rng_master = jax.random.PRNGKey(seed)
    results: List[Any] = []

    for _ep in range(num_episodes):
        rng_master, ep_rng = jax.random.split(rng_master)
        ep_rng, reset_rng = jax.random.split(ep_rng)
        obs, env_state = env.reset(reset_rng)

        hstate_0 = policy_a.init_hstate(1, aux_info={"agent_id": 0})
        hstate_1 = policy_b.init_hstate(1, aux_info={"agent_id": 1})
        done_prev = init_done
        counts = counts_factory()
        episode_return = 0.0
        step_i = 0

        for step_i in range(max_steps):
            avail = env.get_avail_actions(env_state)
            avail = jax.lax.stop_gradient(avail)
            avail_0 = avail["agent_0"].astype(jnp.float32)
            avail_1 = avail["agent_1"].astype(jnp.float32)

            ep_rng, a0_rng, a1_rng, step_rng = jax.random.split(ep_rng, 4)
            act_0, hstate_0 = policy_a.get_action(
                params=params_a,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done_prev["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=a0_rng,
                aux_obs=None,
                env_state=env_state,
                test_mode=False,
            )
            act_0 = int(act_0.squeeze())
            act_1, hstate_1 = policy_b.get_action(
                params=params_b,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=done_prev["agent_1"].reshape(1, 1),
                avail_actions=avail_1,
                hstate=hstate_1,
                rng=a1_rng,
                aux_obs=None,
                env_state=env_state,
                test_mode=False,
            )
            act_1 = int(act_1.squeeze())

            env_act = {"agent_0": act_0, "agent_1": act_1}
            env_state_pre = env_state
            obs, env_state, reward, done, info = env.step(step_rng, env_state, env_act)
            r0 = float(reward["agent_0"])
            r1 = float(reward["agent_1"])
            # on the last step, use the true post-step state, not the auto-reset
            state_post = _true_post_state(env, env_state_pre, env_act, step_rng) if bool(done["__all__"]) else env_state
            feature_update(counts, act_0, act_1, r0, r1, env_state_pre, state_post, info)
            episode_return += r0
            done_prev = done

            if bool(done["__all__"]):
                break

        counts.final_score = episode_return
        counts.episode_length = step_i + 1
        results.append(counts)

    return results


def rollout_simple_self_play(
    env,
    policy,
    num_episodes: int,
    seed: int,
    counts_factory,
    feature_update,
    max_steps: int = 200,
    params=None,
) -> List[Any]:
    """sequential self-play rollout for envs whose policies don't need aux_obs."""
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    rng_master = jax.random.PRNGKey(seed)
    results: List[Any] = []

    for _ep in range(num_episodes):
        rng_master, ep_rng = jax.random.split(rng_master)
        ep_rng, reset_rng = jax.random.split(ep_rng)
        obs, env_state = env.reset(reset_rng)

        hstate_0 = policy.init_hstate(1, aux_info={"agent_id": 0})
        hstate_1 = policy.init_hstate(1, aux_info={"agent_id": 1})
        done_prev = init_done
        counts = counts_factory()
        episode_return = 0.0
        step_i = 0

        for step_i in range(max_steps):
            avail = env.get_avail_actions(env_state)
            avail = jax.lax.stop_gradient(avail)
            avail_0 = avail["agent_0"].astype(jnp.float32)
            avail_1 = avail["agent_1"].astype(jnp.float32)

            ep_rng, a0_rng, a1_rng, step_rng = jax.random.split(ep_rng, 4)
            act_0, hstate_0 = policy.get_action(
                params=params,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done_prev["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=a0_rng,
                aux_obs=None,
                env_state=env_state,
                test_mode=False,
            )
            act_0 = int(act_0.squeeze())
            act_1, hstate_1 = policy.get_action(
                params=params,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=done_prev["agent_1"].reshape(1, 1),
                avail_actions=avail_1,
                hstate=hstate_1,
                rng=a1_rng,
                aux_obs=None,
                env_state=env_state,
                test_mode=False,
            )
            act_1 = int(act_1.squeeze())

            env_act = {"agent_0": act_0, "agent_1": act_1}
            env_state_pre = env_state
            obs, env_state, reward, done, info = env.step(step_rng, env_state, env_act)
            r0 = float(reward["agent_0"])
            r1 = float(reward["agent_1"])
            # on the last step, use the true post-step state, not the auto-reset
            state_post = _true_post_state(env, env_state_pre, env_act, step_rng) if bool(done["__all__"]) else env_state
            feature_update(counts, act_0, act_1, r0, r1, env_state_pre, state_post, info)
            episode_return += r0
            done_prev = done

            if bool(done["__all__"]):
                break

        counts.final_score = episode_return
        counts.episode_length = step_i + 1
        results.append(counts)

    return results


def _lbf_unwrap(env_state):
    s = env_state
    while hasattr(s, "env_state"):
        s = s.env_state
    return s


def _lbf_agent_positions(state) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    s = _lbf_unwrap(state)
    p0 = (int(s.agents.position[0, 0]), int(s.agents.position[0, 1]))
    p1 = (int(s.agents.position[1, 0]), int(s.agents.position[1, 1]))
    return p0, p1


def _lbf_food_positions(state) -> List[Tuple[int, int, int]]:
    """(row, col, level) for foods still uneaten."""
    s = _lbf_unwrap(state)
    # chex dataclasses are always truthy, so guard with `is not None` not `or` in case a fork stores an empty/None placeholder
    food = getattr(s, "food_items", None)
    if food is None:
        food = getattr(s, "food", None)
    if food is None:
        food = getattr(s, "foods", None)
    if food is None:
        return []
    n_food = food.position.shape[0]
    out: List[Tuple[int, int, int]] = []
    for i in range(n_food):
        eaten = bool(food.eaten[i])
        if not eaten:
            r = int(food.position[i, 0])
            c = int(food.position[i, 1])
            lvl = int(food.level[i])
            out.append((r, c, lvl))
    return out


def _lbf_min_dist_to_food(pos: Tuple[int, int], foods: List[Tuple[int, int, int]]) -> int:
    if not foods:
        return 0
    return min(abs(pos[0] - f[0]) + abs(pos[1] - f[1]) for f in foods)


def lbf_step_update(
    counts: LBFEpisodeCounts,
    a0: int,
    a1: int,
    r0: float,
    r1: float,
    state_pre,
    state_post,
    info: Dict,
    horizon: int = 100,
) -> None:
    """update lbf event counts for the teammate being characterized (agent_1).

    agent_1 is the teammate whose behaviour we are describing; agent_0 is its
    paired best response (or, in self-play, a copy of the same policy). This
    intentionally differs from ZSC-Eval, which attributes theta to the BR.
    """
    pre_pos = _lbf_agent_positions(state_pre)
    post_pos = _lbf_agent_positions(state_post)
    foods_pre = _lbf_food_positions(state_pre)
    foods_post = _lbf_food_positions(state_post)

    # Tracked teammate is agent_1; its partner (the paired BR) is agent_0.
    # Local names kept as `br_*` for continuity with the rest of this function.
    br_pre = pre_pos[1]
    br_post = post_pos[1]
    partner_pre = pre_pos[0]
    br_action = a1
    br_reward = r1

    # real env step index of the pre-state, for early/late-game tempo bins
    cur_step = _lbf_step_index(state_pre)
    early_thresh = horizon // 3
    late_thresh = 2 * horizon // 3

    # ------------------------------------------------------------------
    # Derived-vocabulary bookkeeping. Runs BEFORE the action-specific
    # branches below, which `return` early.
    # ------------------------------------------------------------------
    counts.n_steps += 1
    pre_food_set_ext = {(r, c, lvl) for (r, c, lvl) in foods_pre}
    post_food_set_ext = {(r, c, lvl) for (r, c, lvl) in foods_post}
    eaten_ext = pre_food_set_ext - post_food_set_ext

    # relative configuration w.r.t. the partner (row = N/S axis, col = E/W axis)
    if br_pre[0] < partner_pre[0]:
        counts.n_rel_north += 1
    elif br_pre[0] > partner_pre[0]:
        counts.n_rel_south += 1
    if br_pre[1] > partner_pre[1]:
        counts.n_rel_east += 1
    elif br_pre[1] < partner_pre[1]:
        counts.n_rel_west += 1

    # territory: cells occupied at the start of this step (plus the final cell below)
    counts.visited_self.add(br_pre)
    counts.visited_partner.add(partner_pre)
    counts.visited_self.add(br_post)
    counts.visited_partner.add(post_pos[0])

    # first-arrival bookkeeping: earliest step at which each agent stood adjacent
    # to each (still uneaten) fruit. `cur_step` may be -1 if unreadable; fall back
    # to our own step counter so ordering is still well defined.
    order_idx = cur_step if cur_step >= 0 else counts.n_steps - 1
    for (r, c, _lvl) in foods_pre:
        key = (r, c)
        slot = counts.first_adjacent.setdefault(key, {})
        if abs(br_pre[0] - r) + abs(br_pre[1] - c) == 1 and 1 not in slot:
            slot[1] = order_idx
        if abs(partner_pre[0] - r) + abs(partner_pre[1] - c) == 1 and 0 not in slot:
            slot[0] = order_idx

    # cooperative loads: same definition as n_successful_load_cooperative below
    # (tracked agent took LOAD, fruit vanished, both agents adjacent to it).
    if br_action == _LBF_LOAD:
        for (r, c, _lvl) in eaten_ext:
            if abs(br_pre[0] - r) + abs(br_pre[1] - c) != 1:
                continue
            if abs(partner_pre[0] - r) + abs(partner_pre[1] - c) != 1:
                continue
            counts.n_coop_load_events += 1
            slot = counts.first_adjacent.get((r, c), {})
            t_self = slot.get(1)
            t_partner = slot.get(0)
            if t_self is not None and (t_partner is None or t_self < t_partner):
                counts.n_arrival_first += 1

    # "waiting at a fruit": adjacent to some uneaten fruit but not attempting LOAD
    adjacent_to_fruit = any(
        abs(br_pre[0] - r) + abs(br_pre[1] - c) == 1 for (r, c, _l) in foods_pre
    )
    if adjacent_to_fruit and br_action != _LBF_LOAD:
        counts.wait_run_cur += 1
    elif counts.wait_run_cur > 0:
        counts.wait_run_sum += counts.wait_run_cur
        counts.wait_run_count += 1
        counts.wait_run_cur = 0

    # path efficiency: Manhattan-optimal vs actual moves between consecutive loads
    if counts.seg_start_pos is None:
        counts.seg_start_pos = br_pre
    if br_action in (_LBF_N, _LBF_S, _LBF_W, _LBF_E):
        counts.seg_moves += 1
    if br_action == _LBF_LOAD and any(
        abs(br_pre[0] - r) + abs(br_pre[1] - c) == 1 for (r, c, _l) in eaten_ext
    ):
        optimal = abs(counts.seg_start_pos[0] - br_pre[0]) + abs(counts.seg_start_pos[1] - br_pre[1])
        counts.path_eff_sum += min(1.0, optimal / max(1, counts.seg_moves))
        counts.path_eff_count += 1
        counts.seg_start_pos = br_post
        counts.seg_moves = 0

    # partner-distance state: BR-to-partner distance at the start of the step
    partner_dist = abs(br_pre[0] - partner_pre[0]) + abs(br_pre[1] - partner_pre[1])
    counts.partner_distance_sum += partner_dist
    if partner_dist == 1:
        counts.n_state_partner_adjacent += 1
    elif partner_dist <= 3:
        counts.n_state_partner_mid += 1
    else:
        counts.n_state_partner_far += 1

    # per-fruit-level load: a fruit eaten this step is a LOAD for the BR only if the BR was adjacent and took LOAD; compare uneaten foods pre vs post
    pre_food_set = {(r, c, lvl) for (r, c, lvl) in foods_pre}
    post_food_set = {(r, c, lvl) for (r, c, lvl) in foods_post}
    eaten_this_step = pre_food_set - post_food_set
    if br_action == _LBF_LOAD:
        for (r, c, lvl) in eaten_this_step:
            if abs(br_pre[0] - r) + abs(br_pre[1] - c) != 1:
                continue  # BR was not adjacent to this fruit -> partner's load
            if lvl == 1:
                counts.n_load_lvl_1 += 1
            elif lvl == 2:
                counts.n_load_lvl_2 += 1
            elif lvl >= 3:
                counts.n_load_lvl_3 += 1

    # coop-load level: BR + partner both adjacent to the eaten fruit and both took LOAD
    if br_action == _LBF_LOAD and a0 == _LBF_LOAD:
        for (r, c, lvl) in eaten_this_step:
            d_br = abs(br_pre[0] - r) + abs(br_pre[1] - c)
            d_partner = abs(partner_pre[0] - r) + abs(partner_pre[1] - c)
            if d_br == 1 and d_partner == 1:
                if lvl == 2:
                    counts.n_coop_load_lvl_2 += 1
                elif lvl >= 3:
                    counts.n_coop_load_lvl_3 += 1

    # target-conflict: BR and partner both move toward the same nearest fruit
    if br_action in (_LBF_N, _LBF_S, _LBF_E, _LBF_W) and a0 in (_LBF_N, _LBF_S, _LBF_E, _LBF_W):
        d_br_pre = _lbf_min_dist_to_food(br_pre, foods_pre)
        d_partner_pre = _lbf_min_dist_to_food(partner_pre, foods_pre)
        d_br_post = _lbf_min_dist_to_food(br_post, foods_pre)
        d_partner_post = _lbf_min_dist_to_food(post_pos[0], foods_pre)
        if d_br_post < d_br_pre and d_partner_post < d_partner_pre:
            counts.n_target_conflict += 1

    # per-agent (BR == agent_1) action events
    my_pre = br_pre
    my_post = br_post
    action = br_action
    reward = br_reward

    # --- sabotage detection: declining to join a partner waiting at a fruit ---
    # Under force_coop every fruit needs BOTH agents, so a partner standing adjacent to
    # an uneaten fruit is stuck until the tracked agent arrives. Cooperating = reducing
    # distance to THAT fruit; not doing so is the LBF sabotage signature. We evaluate
    # this regardless of action type (a refusal can be a noop, a move away, or a
    # pointless LOAD elsewhere), and we key on the specific fruit the partner is at,
    # not the nearest fruit -- otherwise walking toward a different fruit masks it.
    join_targets = [
        (r, c) for (r, c, _lvl) in foods_pre
        if abs(partner_pre[0] - r) + abs(partner_pre[1] - c) == 1
        and abs(my_pre[0] - r) + abs(my_pre[1] - c) != 1
    ]
    if join_targets:
        # if the partner covers several fruits, credit the one the agent is closest to
        tr, tc = min(join_targets,
                     key=lambda p: abs(my_pre[0] - p[0]) + abs(my_pre[1] - p[1]))
        d_pre_t = abs(my_pre[0] - tr) + abs(my_pre[1] - tc)
        d_post_t = abs(my_post[0] - tr) + abs(my_post[1] - tc)
        counts.n_join_opportunity += 1
        if d_post_t >= d_pre_t:
            counts.n_decline_to_join += 1
        if d_post_t > d_pre_t:
            counts.n_abandon_partner += 1
        # start the latency clock the first time the partner is seen waiting here
        if (tr, tc) not in counts.pending_join and cur_step >= 0:
            counts.pending_join[(tr, tc)] = cur_step
    # resolve latency for any pending fruit the agent has now reached (or that vanished)
    if counts.pending_join and cur_step >= 0:
        my_adj_now = {
            (r, c) for (r, c, _lvl) in foods_pre
            if abs(my_post[0] - r) + abs(my_post[1] - c) == 1
        }
        alive = {(r, c) for (r, c, _lvl) in foods_pre}
        for key in [k for k in counts.pending_join
                    if k in my_adj_now or k not in alive]:
            counts.join_latency_sum += max(0, cur_step - counts.pending_join.pop(key))
            counts.join_latency_count += 1

    # BR's starting quadrant this step; infer grid size from the max coord across both agents (avoids needing env_kwargs)
    gs_proxy = max(my_pre[0], my_pre[1], partner_pre[0], partner_pre[1], 6) + 1
    mid = gs_proxy / 2.0
    if my_pre[0] < mid and my_pre[1] < mid:
        counts.n_quadrant_NW_visit += 1
    elif my_pre[0] < mid and my_pre[1] >= mid:
        counts.n_quadrant_NE_visit += 1
    elif my_pre[0] >= mid and my_pre[1] < mid:
        counts.n_quadrant_SW_visit += 1
    else:
        counts.n_quadrant_SE_visit += 1

    if action == _LBF_NOOP:
        counts.n_noop += 1
        # subcategory: noop while fruit visible
        if len(foods_pre) > 0:
            counts.n_noop_when_food_visible += 1
        # wait_for_partner: BR adjacent to a fruit, partner not adjacent to any fruit, and BR held position
        i_adjacent = any(
            abs(my_pre[0] - r) + abs(my_pre[1] - c) == 1 for (r, c, _) in foods_pre
        )
        partner_adjacent = any(
            abs(partner_pre[0] - r) + abs(partner_pre[1] - c) == 1 for (r, c, _) in foods_pre
        )
        if i_adjacent and not partner_adjacent:
            counts.n_wait_for_partner += 1
        return

    if action == _LBF_LOAD:
        # classify the LOAD by fruit-adjacency, not BR-partner adjacency (LBF force_coop keeps both ~2 apart, never adjacent)
        # coop = BR + partner both adjacent to the eaten fruit; alone = only BR adjacent; failed = ate nothing adjacent
        ate_with_partner = False  # BR ate a fruit and partner was adjacent to it
        ate_alone = False         # BR ate a fruit and partner was not adjacent
        for (r, c, lvl) in eaten_this_step:
            if abs(my_pre[0] - r) + abs(my_pre[1] - c) != 1:
                continue  # BR was not adjacent to this fruit -> partner's load
            d_partner = abs(partner_pre[0] - r) + abs(partner_pre[1] - c)
            if d_partner == 1:
                ate_with_partner = True
            else:
                ate_alone = True

        # solo attempts at high-level fruits: BR adjacent to a lvl-2/3 fruit, took LOAD, partner not adjacent (eaten or not)
        for (r, c, lvl) in foods_pre:
            if abs(my_pre[0] - r) + abs(my_pre[1] - c) != 1:
                continue  # BR not adjacent to this fruit
            d_partner = abs(partner_pre[0] - r) + abs(partner_pre[1] - c)
            if d_partner == 1:
                continue  # partner is also adjacent -> not a solo attempt
            if lvl == 2:
                counts.n_solo_attempt_lvl_2 += 1
            elif lvl >= 3:
                counts.n_solo_attempt_lvl_3 += 1

        if ate_with_partner or ate_alone:
            # early/late game (100-step LBF horizon); cur_step < 0 means step index unreadable, skip tempo binning
            if 0 <= cur_step < early_thresh:
                counts.n_early_game_load += 1
            elif cur_step >= late_thresh:
                counts.n_late_game_load += 1
            if ate_with_partner:
                counts.n_successful_load_cooperative += 1
            else:
                counts.n_successful_load_alone += 1
        else:
            counts.n_failed_load += 1
        return

    # movement actions: N/S/E/W
    if action in (_LBF_N, _LBF_S, _LBF_E, _LBF_W):
        # collision: move attempted but position didn't change (likely partner-blocked)
        if my_pre == my_post:
            counts.n_collision_with_partner += 1
            return
        # approach vs retreat: nearest-food Manhattan distance change
        d_pre = _lbf_min_dist_to_food(my_pre, foods_pre)
        d_post = _lbf_min_dist_to_food(my_post, foods_pre)
        if d_post < d_pre:
            counts.n_approach_fruit += 1
        elif d_post > d_pre:
            counts.n_retreat_from_fruit += 1




# Overcooked action layout (JaxMARL Actions enum: up/down/right/left/stay/interact).
# NOTE: right precedes left, and stay precedes interact -- do not "fix" this to
# alphabetical/intuitive order. See jaxmarl/environments/overcooked/overcooked.py.
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


@dataclass
class OvercookedEpisodeCounts:
    """per-episode overcooked shaped_infos event counts."""
    counts_by_event: Dict[str, int] = field(default_factory=lambda: {ev: 0 for ev in OVERCOOKED_SHAPED_INFOS})
    # --- derived vocabulary: raw accumulators ---
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
    # per-resource-kind: total uses by the tracked agent, and uses of instance #0
    resource_uses: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in _OC_RESOURCE_KINDS})
    resource_first: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in _OC_RESOURCE_KINDS})
    # --- non-feature bookkeeping (episode-level state carried across steps) ---
    # counter position (x, y) -> (placing_agent_idx, item_code, step_idx)
    counter_ledger: Dict[Tuple[int, int], Tuple[int, int, int]] = field(default_factory=dict)
    # counter position -> number of placements by the tracked agent
    placements_by_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)
    visited_self: set = field(default_factory=set)
    visited_partner: set = field(default_factory=set)
    delivery_steps: List[int] = field(default_factory=list)
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

    # resource choice: only meaningful when the layout has >= 2 instances
    resource_fracs = []
    for kind in _OC_RESOURCE_KINDS:
        positions = (counts.resource_pos or {}).get(kind, [])
        uses = counts.resource_uses.get(kind, 0)
        if len(positions) >= 2 and uses > 0:
            resource_fracs.append(counts.resource_first.get(kind, 0) / uses)
        else:
            resource_fracs.append(0.0)

    region_fracs = [
        counts.n_region_NW / ep_len,
        counts.n_region_NE / ep_len,
        counts.n_region_SW / ep_len,
        counts.n_region_SE / ep_len,
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
    # An agent with NO role events has no role at all, so both features must be 0.
    # Without this carve-out n_role=0 gives onion_role_frac=0 -> role_purity=1.0,
    # i.e. a totally idle teammate would score as a maximally pure plate specialist.
    n_role = n_onion + n_plate
    onion_role_frac = (n_onion / n_role) if n_role > 0 else 0.0
    role_purity = abs(2.0 * onion_role_frac - 1.0) if n_role > 0 else 0.0

    deliveries_per_100_steps = 100.0 * ev.get("delivery", 0) / ep_len
    if len(counts.delivery_steps) >= 2:
        gaps = [
            counts.delivery_steps[k] - counts.delivery_steps[k - 1]
            for k in range(1, len(counts.delivery_steps))
        ]
        mean_inter_delivery_interval = (sum(gaps) / len(gaps)) / ep_len
    else:
        mean_inter_delivery_interval = 0.0

    mean_partner_distance = (counts.partner_dist_sum / ep_len) / max(1, counts.dist_norm)

    feats += [
        float(counts.n_handoff_given),
        float(counts.n_handoff_received),
        float(dead_drop),
        float(counts.n_self_retrieve),
        float(counter_entropy),
        *[float(x) for x in resource_fracs],
        *[float(x) for x in region_fracs],
        float(territory_overlap),
        float(onion_role_frac),
        float(role_purity),
        float(deliveries_per_100_steps),
        float(mean_inter_delivery_interval),
        float(mean_partner_distance),
        float(counts.n_interact_into_partner),
        float(counts.n_blocked_partner),
    ]
    assert len(feats) == len(OVERCOOKED_FEATURE_NAMES), (
        f"overcooked feature vector length {len(feats)} != {len(OVERCOOKED_FEATURE_NAMES)} names"
    )
    return np.array(feats, dtype=np.float64)


def _oc_scan_resources(mm: np.ndarray, pad: int, H: int, W: int) -> Dict[str, List[Tuple[int, int]]]:
    """positions (x, y) of each static resource, in canonical (y, x)-sorted order.

    Scanned from maze_map rather than the layout dict so this works regardless of
    how the env was constructed. Pot/pile/goal tiles are walls, so agents can
    never stand on them and their maze_map codes are stable across the episode.
    """
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

    # agent_1 is the teammate being characterized; agent_0 is its paired best
    # response (or a copy of itself in self-play). This intentionally differs
    # from ZSC-Eval, which attributes theta to the BR instead of the teammate.
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
        counts, a0, a1, inv_pre, inv_post, pos_pre, pos_post, adir, mm, pad, H, W,
    )

    if inv0 != inv1:
        # interact succeeded, classify by inventory delta + faced cell
        if inv0 == EMPTY and inv1 == ONION:
            ev["pickup_onion_from_O" if obj == ONION_PILE else "pickup_onion_from_X"] += 1
        elif inv0 == EMPTY and inv1 == PLATE:
            ev["pickup_dish_from_D" if obj == PLATE_PILE else "pickup_dish_from_X"] += 1
            num_notempty_pots = int(((mm[..., 0] == POT) & (mm[..., 2] != POT_EMPTY)).sum())
            num_plates_held = int((inv_pre == PLATE).sum())
            no_plates_on_counters = int((mm == PLATE).sum()) == 0
            if num_plates_held < num_notempty_pots and no_plates_on_counters:
                ev["USEFUL_DISH_PICKUP"] += 1
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
) -> None:
    """derived-vocabulary bookkeeping for overcooked.

    Unlike the shaped_infos reconstruction, this inspects BOTH agents: the
    counter ledger needs to know who put an item down in order to tell a
    handoff from a self-retrieve. Only agent_1's behaviour becomes features.
    """
    TRACKED, PARTNER = 1, 0
    INTERACT_A = 5
    actions = (int(a0), int(a1))

    counts.n_steps += 1
    step_idx = counts.n_steps - 1
    counts.dist_norm = H + W

    if counts.resource_pos is None:
        counts.resource_pos = _oc_scan_resources(mm, pad, H, W)

    # faced cell + object for each agent (interact resolves against the PRE-step
    # position and direction, matching the env's process_interact)
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

    # --- counter ledger -------------------------------------------------
    # Pickups are resolved BEFORE placements so an item can never be picked up
    # on the same step it was put down.
    for j in (0, 1):
        if events[j] not in _OC_PICKUP_FROM_COUNTER:
            continue
        entry = counts.counter_ledger.pop((faced[j][0], faced[j][1]), None)
        if entry is None:
            continue  # item wasn't placed by either agent this episode
        placer = entry[0]
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

    # --- resource choice (tracked agent only) ---------------------------
    event_to_resource = {
        "PLACEMENT_IN_POT": "pot",
        "SOUP_PICKUP": "pot",
        "pickup_onion_from_O": "onion_pile",
        "pickup_dish_from_D": "plate_pile",
        "delivery": "goal",
    }
    kind = event_to_resource.get(events[TRACKED] or "")
    if kind is not None:
        positions = counts.resource_pos.get(kind, [])
        used = (faced[TRACKED][0], faced[TRACKED][1])
        if used in positions:
            counts.resource_uses[kind] += 1
            if used == positions[0]:
                counts.resource_first[kind] += 1

    if events[TRACKED] == "delivery":
        counts.delivery_steps.append(step_idx)

    # --- spatial / territorial ------------------------------------------
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

    # --- partner-relational ---------------------------------------------
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


def build_overcooked_agents(layout: dict) -> Dict[str, Any]:
    """build the overcooked role-heuristic suite for self-play."""
    from agents.overcooked.agent_policy_wrappers import (
        OvercookedIndependentPolicyWrapper,
        OvercookedOnionPolicyWrapper,
        OvercookedPlatePolicyWrapper,
    )

    return {
        "onion": OvercookedOnionPolicyWrapper(layout=layout, using_log_wrapper=True),
        "plate": OvercookedPlatePolicyWrapper(layout=layout, using_log_wrapper=True),
        "independent": OvercookedIndependentPolicyWrapper(layout=layout, using_log_wrapper=True),
    }



def load_full_heldout_for_pd(
    env, env_kwargs: dict, task_name: str, heldout_yaml_key: str, seed: int
) -> Dict[str, tuple]:
    """load every entry from heldout_set.<heldout_yaml_key> via load_heldout_set()."""
    import yaml

    cfg_path = REPO_ROOT / "evaluation" / "configs" / "global_heldout_settings.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"global_heldout_settings.yaml not found at {cfg_path}")
    with cfg_path.open("r") as fh:
        full_yaml = yaml.safe_load(fh)
    if "heldout_set" not in full_yaml or heldout_yaml_key not in full_yaml["heldout_set"]:
        raise KeyError(
            f"heldout_set.{heldout_yaml_key} not found in {cfg_path}. "
            f"Available: {list(full_yaml.get('heldout_set', {}).keys())}"
        )
    heldout_block = full_yaml["heldout_set"][heldout_yaml_key]

    # pre-filter entries whose checkpoint paths or weight files don't exist locally; heuristic agents (no path/weight_file) always pass
    from common.save_load_utils import REPO_PATH

    filtered = {}
    skipped = []
    for name, cfg in heldout_block.items():
        if cfg is None:
            continue
        path_ok = True
        for path_key in ("path", "weight_file"):
            if path_key in cfg:
                p = cfg[path_key]
                resolved = p if os.path.isabs(p) else os.path.join(REPO_PATH, p)
                if not os.path.exists(resolved):
                    path_ok = False
                    skipped.append(f"{name} (missing {path_key}={p})")
                    break
        if path_ok:
            filtered[name] = cfg
    if skipped:
        log.warning("skipping %d heldout entries whose checkpoints aren't local: %s",
                    len(skipped), ", ".join(skipped))

    from evaluation.heldout_core import load_heldout_set

    rng = jax.random.PRNGKey(seed)
    heldout_agents = load_heldout_set(filtered, env, task_name, env_kwargs, rng)
    return heldout_agents


def _resolve_br_leaf(partner_dir: Path) -> Optional[Path]:
    """find the orbax-checkpoint leaf inside a br partner dir."""
    saved = partner_dir / "saved_train_run"
    if saved.exists() and (saved / "_CHECKPOINT_METADATA").exists():
        return saved
    date_dirs = sorted(
        (d for d in partner_dir.iterdir() if d.is_dir()),
        reverse=True,
    )
    for d in date_dirs:
        leaf = d / "ego_train_run"
        if leaf.exists() and (leaf / "_CHECKPOINT_METADATA").exists():
            return leaf
    return None


def _list_available_br_dirs(br_root: Path) -> Dict[str, Path]:
    """map partner-dir-name to resolved orbax leaf path for every valid br."""
    from common.save_load_utils import REPO_PATH
    resolved = br_root if br_root.is_absolute() else Path(REPO_PATH) / br_root
    if not resolved.exists():
        return {}
    out: Dict[str, Path] = {}
    for d in resolved.iterdir():
        if not d.is_dir():
            continue
        leaf = _resolve_br_leaf(d)
        if leaf is not None:
            out[d.name] = leaf
    return out


def _strip_idx_suffix(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    """parse a trailing '(n, m)' or '(n)' index suffix off a name."""
    import re
    m = re.match(r"^(.+?)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$", name)
    if m:
        return m.group(1).strip(), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(.+?)\s*\(\s*(-?\d+)\s*\)\s*$", name)
    if m:
        return m.group(1).strip(), int(m.group(2)), None
    return name, None, None


def _match_partner_to_br(heldout_name: str, available_dirs: List[str], layout_prefix: Optional[str] = None) -> Optional[str]:
    """fuzzy-match a heldout partner to its br dir."""
    base, outer_idx, inner_idx = _strip_idx_suffix(heldout_name)
    base_us = base.replace("-", "_")

    # use the most-specific index for HF naming (inner if both, else outer)
    idx = inner_idx if inner_idx is not None else outer_idx

    candidates: List[str] = []

    # direct + hyphen-to-underscore variants
    for raw in [heldout_name, base, base_us]:
        candidates.append(raw)

    if idx is not None:
        # index-bearing forms common in HF BR dataset
        for prefix in [base_us]:
            candidates.extend([
                f"{prefix}_{idx}",
                f"{prefix}_conf_{idx}",
                f"br_{prefix}_m{idx}",
                f"br_{prefix}_conf_m{idx}",
            ])
            if outer_idx is not None and inner_idx is not None:
                # 'lbrdiv_conf_1_0_serious' style: outer_idx=1, inner_idx=0
                candidates.append(f"{prefix}_{outer_idx}_{inner_idx}")
                candidates.append(f"{prefix}_{outer_idx}_{inner_idx}_serious")

    # try suffix variants on every base candidate
    suffix_variants: List[str] = []
    for c in list(candidates):
        suffix_variants.extend([
            f"{c}_serious",
            f"{c}_long_6e7",
        ])
    candidates.extend(suffix_variants)

    # 'br_for_<name>' prefix (mini-hanabi convention)
    for c in [base, base_us, heldout_name]:
        candidates.append(f"br_for_{c}")

    # layout-prefixed (overcooked)
    if layout_prefix:
        prefixed: List[str] = []
        for c in list(candidates):
            prefixed.append(f"{layout_prefix}_{c}")
        candidates.extend(prefixed)

    # direct match
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c in available_dirs:
            return c

    # pop-methods whose outer idx doesn't line up with the dir name: match on the inner idx (comedi (-1, 0) -> comedi_1_0_serious)
    if inner_idx is not None:
        import re
        pat = re.compile(rf"^{re.escape(base_us)}_-?\d+_{inner_idx}(_.*)?$")
        for d in available_dirs:
            if pat.match(d):
                return d

    # substring match as last resort: any HF dir containing the base name
    for c_base in [base_us, base]:
        if not c_base:
            continue
        for d in available_dirs:
            if c_base in d:
                return d

    return None


def load_brs_for_pd(
    env, env_kwargs: dict, task_name: str, br_root: Path, partner_names: List[str],
    seed: int, layout_prefix: Optional[str] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """load best-response policies for each held-out partner."""
    from common.save_load_utils import REPO_PATH
    from evaluation.heldout_core import load_heldout_set

    rng = jax.random.PRNGKey(seed)
    out: Dict[str, Tuple[Any, Any]] = {}
    skipped: List[str] = []
    br_block: Dict[str, Dict[str, Any]] = {}
    name_map: Dict[str, str] = {}

    available_leaves = _list_available_br_dirs(br_root)
    if not available_leaves:
        log.warning("BR root %s has no valid BR checkpoints; falling back to self-play for all partners.", br_root)
        return out

    available_dirs = list(available_leaves.keys())
    log.info("Found %d candidate BR dirs in %s", len(available_dirs), br_root)

    for name in partner_names:
        matched_dirname = _match_partner_to_br(name, available_dirs, layout_prefix=layout_prefix)
        if matched_dirname is None:
            skipped.append(name)
            continue
        resolved = available_leaves[matched_dirname]
        if not resolved.exists():
            skipped.append(f"{name} (matched {matched_dirname} but leaf missing)")
            continue
        sanitized_key = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
        block_key = f"br_for_{sanitized_key}"
        br_block[block_key] = {
            "path": str(resolved),
            "actor_type": "s5",
            "ckpt_key": "final_params",
            "idx_list": [0],
            "test_mode": True,
        }
        name_map[block_key] = name

    if skipped:
        log.warning("BR checkpoints missing for %d/%d partners: %s",
                    len(skipped), len(partner_names), ", ".join(skipped))
    log.info("Matched BRs for %d/%d partners", len(br_block), len(partner_names))

    if br_block:
        loaded = load_heldout_set(br_block, env, task_name, env_kwargs, rng)
        for full_name, entry in loaded.items():
            base = full_name.split(" (")[0]
            partner_name = name_map.get(base, base[len("br_for_"):] if base.startswith("br_for_") else base)
            out[partner_name] = (entry[0], entry[1])

    return out


def _lbf_step_index(state) -> int:
    """real env step index for a (pre)state, or -1 if unreadable."""
    s = _lbf_unwrap(state)
    sc = getattr(s, "step_count", None)
    if sc is None:
        # WrappedEnvState stores the step under `.step`; walk the wrapper chain.
        w = state
        while w is not None:
            cand = getattr(w, "step", None)
            # guard against bound methods named `step` on env objects
            if cand is not None and not callable(cand):
                sc = cand
                break
            w = getattr(w, "env_state", None)
    if sc is None:
        return -1
    try:
        return int(sc)
    except (TypeError, ValueError):
        return -1
