from jaxmarl.environments.hanabi.hanabi import HanabiEnv

class HanabiWrapperOld(HanabiEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_avail_actions(self, env_state):
        return super().get_legal_moves(env_state)

    def observation_space(self, agent: str):
        self.observation_spaces[agent].shape = (self.observation_spaces[agent].n,)
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        self.action_spaces[agent].shape = (self.action_spaces[agent].n,)
        return self.action_spaces[agent]

    def __getattr__(self, name):
        return getattr(super(), name)