from jaxmarl.environments.coin_game.coin_game import CoinGame

class CoinGameWrapper(CoinGame):
    def __init__(self, num_agents=2, grid_size=7, max_steps=1000, **kwargs):
        super().__init__(num_agents=num_agents, grid_size=grid_size, max_steps=max_steps, **kwargs)
        self._name = "CoinGame"
        self._num_agents = num_agents
        self._grid_size = grid_size
        self._max_steps = max_steps

    def reset(self, key=None):
        return super().reset(key=key)

    def step(self, actions):
        return super().step(actions)

    def observation(self, state, agent_id):
        return super().observation(state, agent_id)
