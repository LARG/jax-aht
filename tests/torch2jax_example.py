import torch
import jax
from torch2jax import torch2jax  # this converts a Python function to JAX
from torch2jax import Size, dtype_t2j  # this is torch.Size, a tuple-like shape representation


def torch_fn(a, b):
    return a + b


shape = (10, 2)
a, b = torch.randn(shape), torch.randn(shape)
jax_fn = torch2jax(torch_fn, a, b)  # without output_shapes, torch_fn **will be evaluated once**
jax_fn = torch2jax(torch_fn, a, b, output_shapes=Size(a.shape))  # torch_fn will NOT be evaluated

# you can specify the whole input and output structure without instantiating the tensors
# torch_fn will NOT be evaluated
jax_fn = torch2jax(
    torch_fn,
    jax.ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
    jax.ShapeDtypeStruct(b.shape, dtype_t2j(b.dtype)),
    output_shapes=jax.ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
)

prngkey = jax.random.PRNGKey(0)
device = jax.devices("cuda")[0]  # both CPU and CUDA are supported
a = jax.device_put(jax.random.normal(prngkey, shape), device)
b = jax.device_put(jax.random.normal(prngkey, shape), device)

# call the no-copy torch function
out = jax_fn(a, b)

# call the no-copy torch function **under JIT**
out = jax.jit(jax_fn)(a, b)