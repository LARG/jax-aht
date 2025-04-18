import jax

# create initial rng
rng = jax.random.PRNGKey(38410)
print("initial rng: ", rng)

# split rng into two
_, rng1 = jax.random.split(rng)
print("rng1, split from initial rng: ", rng1)

# show that rng is deterministic
_, rng2 = jax.random.split(rng)
print("rng2, split from initial rng: ", rng2)

# show that rng1 and rng2 are the same
print("rng1 == rng2: ", rng1 == rng2)

