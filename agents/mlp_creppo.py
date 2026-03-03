import jax
import jax.numpy as jnp

import flax.linen as nn

def hl_gauss(inp, num_bins, vmin, vmax, epsilon=0.0):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    x = jnp.clip(inp, vmin, vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(
        vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*inp.shape[:-1], num_bins)

    uniform = jnp.ones_like(target_probs) / num_bins

    return (1 - epsilon) * target_probs + epsilon * uniform

class QNetwork(nn.Module):
    action_dim: int
    num_bins: int
    v_min: float
    v_max: float
    norm_type: str = "layer_norm"
    norm_input: bool = False
    init_alpha: float = 0.01
    hidden_size: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        log_alpha = self.param(
            "log_alpha", nn.initializers.constant(jnp.log(self.init_alpha)), (1,)
        )

        x, avail_actions = x

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            # x = x / 255.0

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim * self.num_bins)(x)

        zero_dist = self.param(
            "zero_dist",
            nn.initializers.constant(
                hl_gauss(jnp.zeros((1,)), self.num_bins, self.v_min, self.v_max)
            ),
            (1, self.num_bins),
        )

        x = x.reshape((-1, self.action_dim, self.num_bins))

        logits = x + zero_dist * 40
        probs = nn.softmax(logits, axis=-1)
        values = jnp.sum(
            probs * jnp.linspace(self.v_min, self.v_max, self.num_bins, endpoint=True),
            axis=-1,
        )  # expectation)

        # Mask unavailable actions if avail_actions is provided
        unavail_actions = 1 - avail_actions
        masked_values = values - (unavail_actions * 1e10)

        return {
            "logits": logits,
            "probs": probs,
            "q_values": masked_values,
            "policy_logits": masked_values / jnp.exp(log_alpha),
        }
