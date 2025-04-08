'''
Testing the wrap-torch2jax library.
To install, we ran: 
(1) pip install torch
(2) pip install git+https://github.com/rdyro/torch2jax.git
If you get an error about the library failing to compile, verify that 
running "nvcc --version" return a CUDA version >=12
'''

import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
from torch2jax import torch2jax_with_vjp, tree_t2j


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def compare_outputs(torch_output, jax_output, tolerance=1e-5):
    """Compare PyTorch and JAX outputs numerically."""
    # Convert PyTorch tensor to numpy
    torch_np = torch_output.detach().cpu().numpy()
    
    # Convert JAX array to numpy
    jax_np = np.array(jax_output)
    
    # Calculate absolute difference
    abs_diff = np.abs(torch_np - jax_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if outputs are close
    is_close = np.allclose(torch_np, jax_np, rtol=tolerance, atol=tolerance)
    print(f"Outputs are {'close' if is_close else 'different'} (within tolerance {tolerance})")
    
    return is_close


if __name__ == "__main__":
    # Model parameters
    input_size = 10
    hidden_size = 5
    output_size = 2
    
    # Create PyTorch model
    torch_model = SimpleNet(input_size, hidden_size, output_size)
    
    # Create a random input tensor
    torch_input_tensor = torch.randn(1, input_size)
    jax_input_tensor = jnp.array(torch_input_tensor)
    
    # Forward pass with PyTorch model
    torch_output = torch_model(torch_input_tensor)
    print("PyTorch Input Shape:", torch_input_tensor.shape)
    print("PyTorch Output Shape:", torch_output.shape)
    
    # Get PyTorch model parameters
    torch_params, torch_buffers = dict(torch_model.named_parameters()), dict(torch_model.named_buffers())
    torch_params = {k: v.detach() for k, v in torch_params.items()}
    torch_buffers = {k: v.detach() for k, v in torch_buffers.items()}

    # Define a function to call the PyTorch model
    def torch_fwd_fn(params, buffers, input):
        buffers = {k: torch.clone(v) for k, v in buffers.items()}
        return torch.func.functional_call(torch_model, (params, buffers), args=input)
    
    nondiff_argnums = (1, 2)  # buffers, input
    jax_fwd_fn = jax.jit(
        torch2jax_with_vjp(torch_fwd_fn, torch_params, torch_buffers, 
                        torch_input_tensor, nondiff_argnums=nondiff_argnums)
    )
    jax_params, jax_buffers = tree_t2j(torch_params), tree_t2j(torch_buffers)

    # Forward pass with JAX model
    jax_output = jax_fwd_fn(jax_params, jax_buffers, jax_input_tensor)
    print("JAX Output Shape:", jax_output.shape)
    
    # Compare outputs
    compare_outputs(torch_output, jax_output)
