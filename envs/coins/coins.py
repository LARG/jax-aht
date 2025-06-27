from jaxmarl.environments.coin_game.coin_game import CoinGame

class CoinGameWrapper(CoinGame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_avail_actions(self, env_state):
        # TODO: Implement availability of actions based on the environment state.
        return super().get_avail_actions(env_state)

    def observation_space(self, agent: str):
        """
        Returns the observation space for the given agent.
        If the observation space has an attribute 'n', it will be converted to a shape of (n,).
        """
        space = super().observation_space(agent)
        if hasattr(space, "n"):
            space.shape = (space.n,)
        return space

    def action_space(self, agent: str):
        """
        Returns the action space for the given agent.
        If the action space has an attribute 'n', it will be converted to a shape of (n,).
        """
        space = super().action_space(agent)
        if hasattr(space, "n"):
            space.shape = (space.n,)
        return space

    def __getattr__(self, name):
        return getattr(super(), name)
