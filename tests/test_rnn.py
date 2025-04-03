import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import functools
import matplotlib.pyplot as plt

class ScannedRNN(nn.Module):
    gru_hidden_dim: int = 64

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x # resets should have shape (batch_size,), ins should have shape (batch_size, input_dim)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(rnn_state.shape[0]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.gru_hidden_dim)(rnn_state, ins)
        return new_rnn_state, y
    
    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.gru_hidden_dim))


def main():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Model parameters
    seq_len = 20
    batch_size = 2
    input_dim = 5
    hidden_dim = 64
    
    # Initialize model
    model = ScannedRNN(gru_hidden_dim=hidden_dim)
    
    # Create sample data
    key, subkey = jax.random.split(key)
    inputs = jnp.ones((seq_len, batch_size, input_dim))
    # inputs = jax.random.normal(subkey, (seq_len, batch_size, input_dim))
    resets = jnp.zeros((seq_len, batch_size), dtype=bool)  # No resets
    
    # Mark one reset in the middle to verify reset behavior
    resets = resets.at[seq_len//2, :].set(True)
    
    # Initialize parameters
    key, subkey = jax.random.split(key)
    init_carry = model.initialize_carry(batch_size)
    print("Init carry shape: ", init_carry.shape) # (batch_size, hidden_dim)
    params = model.init(subkey, init_carry, (inputs, resets))
    
    # Define a function to apply the model
    def apply_rnn(params, init_carry, inputs, resets):
        return model.apply(params, init_carry, (inputs, resets)) 
    
    ####################################################
    # Test 1: Demonstrate scanning over time axis
    final_state, outputs = apply_rnn(params, init_carry, inputs, resets)
    
    print("=== TEST 1: Automatic Scanning Over Time ===")
    print(f"Input shape: {inputs.shape}") # (seq_len, batch_size, input_dim)
    print(f"Output shape: {outputs.shape}") # (seq_len, batch_size, hidden_dim)
    print(f"Final state shape: {final_state.shape}")
    print(f"Outputs for timestep 0:\n{outputs[0]}")
    print(f"Outputs for timestep -1:\n{outputs[-1]}")
    print(f"Effect of reset at timestep {seq_len//2}:")
    print(f"Before reset (t={seq_len//2-1}):\n{outputs[seq_len//2-1]}")
    print(f"After reset (t={seq_len//2}):\n{outputs[seq_len//2]}")
    
    ####################################################
    # Test 2: Verify that computational graph is maintained
    # Create a target output (just for demonstration)
    target = jnp.ones((seq_len, batch_size, hidden_dim))
    
    # Define a loss function
    def loss_fn(params, init_carry, inputs, resets, target):
        _, outputs = apply_rnn(params, init_carry, inputs, resets)
        return jnp.mean((outputs - target) ** 2)
    
    # Calculate gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, init_carry, inputs, resets, target)
    
    print("\n=== TEST 2: Computational Graph Maintenance ===")
    print("Gradient shapes for GRU parameters:")
    for key, value in jax.tree_util.tree_leaves_with_path(grads):
        print(f"{'.'.join(str(k) for k in key)}: {value.shape}")
    
    ####################################################
    # Test 3: Visualize gradients flowing through time
    # Define a function that computes loss for a single timestep
    def single_step_loss(params, init_carry, inputs, resets, target, step_idx):
        _, outputs = apply_rnn(params, init_carry, inputs[:step_idx+1], resets[:step_idx+1])
        return jnp.mean((outputs[step_idx] - target[step_idx]) ** 2)
    
    # Calculate gradients for different sequence lengths
    gradient_norms = []
    for i in range(1, seq_len + 1):
        step_grad_fn = jax.grad(lambda p: single_step_loss(p, init_carry, inputs, resets, target, i-1))
        step_grads = step_grad_fn(params)
        # Calculate the norm of gradients
        grad_norm = 0
        for g in jax.tree_util.tree_leaves(step_grads):
            grad_norm += jnp.sum(g ** 2)
        gradient_norms.append(jnp.sqrt(grad_norm).item())
    
    print("\n=== TEST 3: Gradient Norm vs Sequence Length ===")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, seq_len + 1), gradient_norms)
    plt.xlabel('Sequence Length')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm vs. Sequence Length')
    plt.grid(True)
    plt.savefig('figures/test_rnn/rnn_gradient_flow.png')
    print("Gradient flow visualization saved to 'rnn_gradient_flow.png'")
    
    ####################################################
    # Test 4: Visualize backpropagation through time
    def single_step_loss_fixed_length(params, init_carry, inputs, resets, target, timestep):
        """Calculate loss for a specific timestep while keeping sequence length constant."""
        _, outputs = apply_rnn(params, init_carry, inputs, resets)
        return jnp.mean((outputs[timestep] - target[timestep]) ** 2)
    
    # Calculate gradients for different timesteps in the full sequence
    bptt_gradient_norms = []
    for i in range(seq_len):
        step_grad_fn = jax.grad(lambda p: single_step_loss_fixed_length(p, init_carry, inputs, resets, target, i))
        step_grads = step_grad_fn(params)
        # Calculate the norm of gradients
        grad_norm = 0
        for g in jax.tree_util.tree_leaves(step_grads):
            grad_norm += jnp.sum(g ** 2)
        bptt_gradient_norms.append(jnp.sqrt(grad_norm).item())
    
    print("\n=== TEST 4: Backpropagation Through Time ===")
    plt.figure(figsize=(10, 6))
    plt.plot(range(seq_len-1, -1, -1), bptt_gradient_norms)
    plt.xlabel('Steps Back in Time from Final Timestep')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Through Backpropagation')
    plt.grid(True)
    plt.savefig('figures/test_rnn/rnn_backprop_through_time.png')
    print("Backpropagation visualization saved to 'rnn_backprop_through_time.png'")





if __name__ == "__main__":
    main()