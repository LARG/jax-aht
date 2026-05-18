import jax.numpy as jnp

from agents.bc.lbf_features import augment_lbf_obs


def test_augment_lbf_obs_adds_finite_features():
    obs = jnp.array(
        [[
            1, 1, 2,
            2, 3, 1,
            -1, -1, 0,
            0, 0, 1,
            1, 0, 1,
        ]],
        dtype=jnp.float32,
    )

    augmented = augment_lbf_obs(obs, grid_size=7, num_food=3)

    assert augmented.shape == (1, 73)
    assert jnp.isfinite(augmented).all()
    assert jnp.array_equal(augmented[:, : obs.shape[-1]], obs)
