"""AgentPolicy wrappers for Hanabi held-out partners."""
import jax
import jax.numpy as jnp
from agents.agent_interface import AgentPolicy
from agents.hanabi.random_agent import RandomAgent
from agents.hanabi.rule_based_agent import RuleBasedAgent
from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.piers_agent import PiersAgent
from agents.hanabi.flawed_agent import FlawedAgent
from agents.hanabi.outer_agent import OuterAgent
from agents.hanabi.van_den_bergh_agent import VanDenBerghAgent
from agents.hanabi.smartbot_agent import SmartBotAgent
from agents.hanabi.obl_r2d2_agent import OBLAgentR2D2
from agents.hanabi.bc_lstm_agent import BCLSTMAgent, bc_lstm_forward


class HanabiRandomPolicyWrapper(AgentPolicy):

    def __init__(self, num_actions: int = 20, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = RandomAgent(num_actions=num_actions,
                                  agent_names=agent_names)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiRuleBasedPolicyWrapper(AgentPolicy):
    """Parameterized play/hint/discard priority. See rule_based_agent.py."""

    def __init__(self, strategy: str = "cautious",
                 hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = RuleBasedAgent(
            strategy=strategy, hand_size=hand_size,
            num_colors=num_colors, num_ranks=num_ranks,
            num_actions=num_actions, agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiIGGIPolicyWrapper(AgentPolicy):

    def __init__(self, hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = IGGIAgent(
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions,
            agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiPiersPolicyWrapper(AgentPolicy):

    def __init__(self, play_threshold: float = 0.6, hint_threshold: int = 4,
                 hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = PiersAgent(
            play_threshold=play_threshold, hint_threshold=hint_threshold,
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions,
            agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiFlawedPolicyWrapper(AgentPolicy):

    def __init__(self, mistake_prob: float = 0.3,
                 hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = FlawedAgent(
            mistake_prob=mistake_prob,
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions,
            agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiOuterPolicyWrapper(AgentPolicy):

    def __init__(self, hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = OuterAgent(
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions,
            agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiVanDenBerghPolicyWrapper(AgentPolicy):

    def __init__(self, hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, using_log_wrapper: bool = False,
                 agent_names=None):
        self.policy = VanDenBerghAgent(
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions,
            agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiSmartBotPolicyWrapper(AgentPolicy):

    def __init__(self, hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5,
                 num_actions: int = 21, card_counts=None,
                 using_log_wrapper: bool = False, agent_names=None):
        self.policy = SmartBotAgent(
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions,
            card_counts=card_counts,
            agent_names=agent_names,
        )
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        # reset conventions on episode end
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class HanabiOBLPolicyWrapper(AgentPolicy):
    """OBL R2D2 pretrained agent. hstate is LSTM carry, not AgentState."""

    def __init__(self, weight_file: str, using_log_wrapper: bool = False):
        self.agent = OBLAgentR2D2()
        self.using_log_wrapper = using_log_wrapper
        # resolve relative to repo root (Hydra changes CWD)
        import os
        from common.save_load_utils import REPO_PATH
        if not os.path.isabs(weight_file):
            weight_file = os.path.join(REPO_PATH, weight_file)
        self.weight_file = weight_file
        self._params = None

    def _load_params(self):
        if self._params is None:
            from jaxmarl.wrappers.baselines import load_params
            self._params = load_params(self.weight_file)
        return self._params

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obl_params = self._load_params()
        obs_flat = obs.reshape(-1)

        if avail_actions is not None and avail_actions.ndim >= 1:
            legal_mask = avail_actions.reshape(-1).astype(jnp.float32)
        else:
            legal_mask = jnp.ones(self.agent.out_dim, dtype=jnp.float32)

        carry, action = self.agent.greedy_act(
            obl_params, hstate, (obs_flat, legal_mask)
        )

        carry = jax.lax.cond(
            done.squeeze().astype(bool),
            lambda: self.agent.initialize_carry(rng, batch_dims=()),
            lambda: carry,
        )
        return action, carry

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.agent.initialize_carry(
            jax.random.PRNGKey(0), batch_dims=()
        )


class HanabiBCLSTMPolicyWrapper(AgentPolicy):
    """BC-LSTM human proxy. hstate is LSTM carry; params baked in at init."""

    def __init__(self, weight_file: str, using_log_wrapper: bool = False,
                 greedy: bool = True):
        self.agent = BCLSTMAgent(weight_path=weight_file)
        self.using_log_wrapper = using_log_wrapper
        self.greedy = greedy

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs_flat = obs.reshape(-1)

        if avail_actions is not None and avail_actions.ndim >= 1:
            legal_mask = avail_actions.reshape(-1).astype(jnp.float32)
        else:
            legal_mask = jnp.ones(21, dtype=jnp.float32)

        if self.greedy or test_mode:
            carry, action = self.agent.greedy_act(hstate, obs_flat, legal_mask)
        else:
            carry, action = self.agent.sample_act(hstate, obs_flat, legal_mask, rng)

        carry = jax.lax.cond(
            done.squeeze().astype(bool),
            lambda: self.agent.initialize_carry(),
            lambda: carry,
        )
        return action, carry

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.agent.initialize_carry()
