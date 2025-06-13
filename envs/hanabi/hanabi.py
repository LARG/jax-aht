from jaxmarl.environments.hanabi.hanabi import HanabiEnv

class HanabiWrapper(HanabiEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_avail_actions(self, env_state):
        return super().get_legal_moves(env_state)