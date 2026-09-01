"""Check that JAX can run the operations used during training.

Usage: python scripts/verify_install.py
"""

import sys
import traceback

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal


def devices():
    print(f"jax {jax.__version__}, backend {jax.default_backend()}, {jax.devices()}")


def matmul():
    a = jnp.ones((512, 512), dtype=jnp.float32)
    (a @ a).block_until_ready()


def qr():
    jax.block_until_ready(jnp.linalg.qr(jnp.eye(8, dtype=jnp.float32)))


def orthogonal_init():
    layer = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))
    jax.block_until_ready(layer.init(jax.random.PRNGKey(0), jnp.zeros((1, 32))))


CHECKS = [
    ("devices", devices),
    ("matrix multiplication", matmul),
    ("QR decomposition", qr),
    ("orthogonal init (as used by agents/)", orthogonal_init),
]

GPU_HINT = """
A segfault or a cuSolver/cuBLAS error here usually means a system CUDA toolkit on
LD_LIBRARY_PATH is being mixed with the pip-installed CUDA libraries. See the
troubleshooting section of docs/install_instructions.md.
"""


def main():
    for name, fn in CHECKS:
        try:
            fn()
        except Exception:  # noqa: BLE001 - report any backend check failure cleanly
            traceback.print_exc()
            print(f"[FAIL] {name}")
            if jax.default_backend() == "gpu":
                print(GPU_HINT)
            return 1
        print(f"[ok] {name}")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
