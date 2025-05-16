# TODO: implement an encoder decoder model for trajectories
# @ Johnny

# model
import jax
import jax.numpy as jnp
from flax import linen as nn

class TrajectoryEncoder(nn.Module):
    hidden_size: int
    embedding_size: int

    @nn.compact
    def __call__(self, trajectory):
        """
        trajectory: [T, D] where T is the sequence length and D is the feature dimension
        Returns: [embedding_size]
        """
        lstm = nn.LSTMCell()
        carry = lstm.initialize_carry(jax.random.PRNGKey(0), (), self.hidden_size)
        for t in range(trajectory.shape[0]):
            carry, _ = lstm(carry, trajectory[t])
        h, _ = carry
        embedding = nn.Dense(self.embedding_size)(h)
        return embedding

class TrajectoryDecoder(nn.Module):
    hidden_size: int
    output_dim: int
    seq_len: int

    @nn.compact
    def __call__(self, embedding):
        """
        embedding: [embedding_size]
        Returns: [T, D] reconstructed trajectory
        """
        lstm = nn.LSTMCell()
        carry = lstm.initialize_carry(jax.random.PRNGKey(0), (), self.hidden_size)
        # Optional: condition hidden state on the embedding
        carry = (nn.Dense(self.hidden_size)(embedding), carry[1])

        outputs = []
        x = jnp.zeros((self.output_dim,))  # Start token

        for _ in range(self.seq_len):
            carry, y = lstm(carry, x)
            x = nn.Dense(self.output_dim)(y)
            outputs.append(x)

        return jnp.stack(outputs)

class TrajectoryAutoencoder(nn.Module):
    encoder_hidden_size: int
    decoder_hidden_size: int
    embedding_size: int
    output_dim: int
    seq_len: int

    def setup(self):
        self.encoder = TrajectoryEncoder(
            hidden_size=self.encoder_hidden_size,
            embedding_size=self.embedding_size
        )
        self.decoder = TrajectoryDecoder(
            hidden_size=self.decoder_hidden_size,
            output_dim=self.output_dim,
            seq_len=self.seq_len
        )

    def __call__(self, trajectory):
        embedding = self.encoder(trajectory)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    

if __name__ == "__main__":
    from flax.training import train_state
    import optax

    trajectory_dim = 10 # Change this to whatever the actual size of the state is
    trajectory_len = 400 # Change this to whatever the actual trajectory length is

    # Create a model
    model = TrajectoryAutoencoder(
        encoder_hidden_size=128,
        decoder_hidden_size=128,
        embedding_size=32,
        output_dim=trajectory_dim,  # dimension of each timestep
        seq_len=trajectory_len
    )

    # Initialize
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((trajectory_len, trajectory_dim))
    params = model.init(key, dummy_input)

    # Create train state
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Define loss
    def loss_fn(params, batch):
        recon, _ = model.apply(params, batch)
        return jnp.mean((recon - batch) ** 2)

    # -----------------------

    @jax.jit
    def train_helper(carry, x):
        def train_step(state, batch):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        new_carry, loss = train_step(carry, x)

        return new_carry, loss

    init_carry = state
    xs = jnp.array([dummy_input] * 10) 
    final_carry, losses = jax.lax.scan(
        train_helper, init_carry, xs, length=10)

    print(final_carry)

    # @jax.jit
    # def train_helper(carry, x):

    #     # new_carry, loss = train_step(carry, x)

    #     return carry, x

    # init_carry = 1
    # xs = jnp.array([i for i in range(10)]) 
    # final_carry, stack = jax.lax.scan(
    #     train_helper, init_carry, xs, length=10)

    # print(final_carry)

    # print(stack)