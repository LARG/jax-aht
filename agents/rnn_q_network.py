import flax.linen as nn

from flax.linen.initializers import constant, orthogonal

from agents.rnn_actor_critic import ScannedRNN


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int = 64
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)

        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals
