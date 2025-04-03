import jax

key = jax.random.PRNGKey(20394)
print("Original key: ", key)

for i in range(5):
    print("Key before splitting: ", key)
    key, new_key = jax.random.split(key)
    print("Key is ", key, "new key is ", new_key)
