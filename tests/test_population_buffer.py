import jax
import jax.numpy as jnp
from agents.population_buffer import BufferedPopulation

class MockPolicy:
    @staticmethod
    def init_hstate(n, aux_info=None):
        return jnp.zeros((n, 1))
    
    @staticmethod
    def get_action(params, obs, done, avail_actions, hstate, rng, aux_obs=None, env_state=None, test_mode=False):
        # Deterministic action based on params for testing
        return params.sum(axis=-1), hstate

def test_softmax_sampling():
    max_pop_size = 5
    temp = 0.5
    pop = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=MockPolicy, sampling_strategy="softmax", temp=temp)
    
    # Initialize buffer
    example_params = jnp.array([1.0, 2.0])
    buffer = pop.reset_buffer(example_params)
    
    # Add agents with specific scores
    buffer = pop.add_agent(buffer, jnp.array([10.0, 10.0]), score=1.0) # idx 0
    buffer = pop.add_agent(buffer, jnp.array([20.0, 20.0]), score=2.0) # idx 1
    
    # Check scores
    assert buffer.scores[0] == 1.0
    assert buffer.scores[1] == 2.0
    assert jnp.all(buffer.filled[:2])
    assert not jnp.any(buffer.filled[2:])
    
    # Check sampling distribution
    dist = pop._get_softmax_sampling_dist(buffer)
    
    # Expected:
    # logits: [1.0/0.5, 2.0/0.5, -inf, -inf, -inf] = [2.0, 4.0, -inf, -inf, -inf]
    # exp(2) / (exp(2)+exp(4)), exp(4) / (exp(2)+exp(4))
    
    expected_probs = jax.nn.softmax(jnp.array([2.0, 4.0]))
    assert jnp.abs(dist[0] - expected_probs[0]) < 1e-5
    assert jnp.abs(dist[1] - expected_probs[1]) < 1e-5
    assert jnp.all(dist[2:] == 0.0)
    
    # Test update_scores
    buffer = pop.update_scores(buffer, jnp.array([0]), jnp.array([5.0]))
    assert buffer.scores[0] == 5.0
    
    # Test sampling many times
    rng = jax.random.PRNGKey(42)
    indices, _ = pop.sample_agent_indices(buffer, 1000, rng)
    
    # With score 5.0 and 2.0, temp 0.5
    # logits [10.0, 4.0]
    counts = jnp.bincount(indices, length=max_pop_size)
    assert counts[0] > counts[1]
    assert jnp.all(counts[2:] == 0)
    
    print("Softmax sampling test passed!")

def test_uniform_sampling():
    max_pop_size = 5
    pop = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=MockPolicy, sampling_strategy="uniform")
    
    # Initialize buffer
    example_params = jnp.array([1.0])
    buffer = pop.reset_buffer(example_params)
    
    # Add agents until full
    for i in range(max_pop_size):
        buffer = pop.add_agent(buffer, jnp.array([float(i)]), score=1.0)
    
    # Check that it's full
    assert buffer.filled_count[0] == max_pop_size
    assert jnp.all(buffer.filled)
    # FIFO pointer should have wrapped to 0
    assert buffer.write_ptr[0] == 0
    
    # Check sampling distribution is flat
    dist = pop._get_uniform_sampling_dist(buffer)
    assert jnp.all(jnp.abs(dist - 1.0/max_pop_size) < 1e-6)
    
    # Now the buffer is full. FIFO should replace idx 0 next.
    buffer = pop.add_agent(buffer, jnp.array([99.0]), score=10.0)
    assert jnp.abs(buffer.params[0, 0] - 99.0) < 1e-5
    assert buffer.write_ptr[0] == 1  # pointer advanced to 1
    
    # Next insertion replaces idx 1 (FIFO), not idx 0 again.
    buffer = pop.add_agent(buffer, jnp.array([100.0]), score=20.0)
    assert jnp.abs(buffer.params[1, 0] - 100.0) < 1e-5  # idx 1 replaced
    assert jnp.abs(buffer.params[0, 0] - 99.0) < 1e-5   # idx 0 unchanged
    assert buffer.write_ptr[0] == 2  # pointer advanced to 2
    
    print("Uniform sampling (FIFO) test passed!")

def test_plr_sampling():
    max_pop_size = 5
    staleness_coef = 0.3
    pop = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=MockPolicy, 
                             sampling_strategy="plr", staleness_coef=staleness_coef)
    
    # Initialize buffer
    buffer = pop.reset_buffer(jnp.array([1.0]))
    
    # Add agents
    buffer = pop.add_agent(buffer, jnp.array([10.0]), score=2.0) # idx 0, age will eventually be 2
    buffer = pop.add_agent(buffer, jnp.array([20.0]), score=1.0) # idx 1, age will eventually be 1
    buffer = pop.add_agent(buffer, jnp.array([30.0]), score=3.0) # idx 2, age will be 0
    
    # Ages should be 2, 1, 0 for the first 3 agents
    assert buffer.ages[0] == 2
    assert buffer.ages[1] == 1
    assert buffer.ages[2] == 0
    
    # Distribution check (temp=1.0 default; scores**(1/1) == scores, so same as before)
    # score_dist = [2/6, 1/6, 3/6], staleness_dist = [2/3, 1/3, 0/3]
    # dist = 0.7 * score_dist + 0.3 * staleness_dist
    dist = pop._get_plr_sampling_dist(buffer)
    expected_dist_0 = 0.7 * (2/6) + 0.3 * (2/3)
    assert jnp.abs(dist[0] - expected_dist_0) < 1e-5
    
    # Test age reset upon sampling
    rng = jax.random.PRNGKey(42)
    indices, new_buffer = pop.sample_agent_indices(buffer, 1, rng)
    
    sampled_idx = indices[0]
    assert new_buffer.ages[sampled_idx] == 0
    # others should be incremented
    for i in range(3):
        if i != sampled_idx:
            assert new_buffer.ages[i] == buffer.ages[i] + 1
            
    print("PLR sampling test passed!")

def test_plr_temperature():
    """Verify that temperature actually affects the PLR score distribution."""
    max_pop_size = 4
    pop_low  = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=MockPolicy,
                                  sampling_strategy="plr", staleness_coef=0.0, temp=0.5)
    pop_high = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=MockPolicy,
                                  sampling_strategy="plr", staleness_coef=0.0, temp=2.0)
    
    buf_low  = pop_low.reset_buffer(jnp.array([1.0]))
    buf_high = pop_high.reset_buffer(jnp.array([1.0]))
    
    scores = [1.0, 2.0, 3.0, 4.0]
    for s in scores:
        buf_low  = pop_low.add_agent(buf_low,  jnp.array([1.0]), score=s)
        buf_high = pop_high.add_agent(buf_high, jnp.array([1.0]), score=s)
    
    dist_low  = pop_low._get_plr_sampling_dist(buf_low)   # staleness_coef=0 -> pure score
    dist_high = pop_high._get_plr_sampling_dist(buf_high)
    
    # Lower temperature should give a more peaked (higher max - lower min) distribution.
    assert (dist_low.max() - dist_low.min()) > (dist_high.max() - dist_high.min()), \
        "Lower temp should produce a more peaked distribution"
    
    # Rank order should be preserved regardless of temperature
    assert jnp.argmax(dist_low) == jnp.argmax(dist_high)
    assert jnp.argmin(dist_low) == jnp.argmin(dist_high)
    
    print("PLR temperature test passed!")


def test_insert_idx_softmax():
    max_pop_size = 3
    pop = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=MockPolicy, sampling_strategy="softmax")
    buffer = pop.reset_buffer(jnp.array([0.0]))
    
    buffer = pop.add_agent(buffer, jnp.array([1.0]), score=10.0)
    buffer = pop.add_agent(buffer, jnp.array([2.0]), score=20.0)
    buffer = pop.add_agent(buffer, jnp.array([3.0]), score=5.0)
    # write_ptr has wrapped: 3 % 3 == 0
    assert buffer.write_ptr[0] == 0
    
    # Buffer is full. FIFO should replace idx 0 (write_ptr == 0).
    buffer = pop.add_agent(buffer, jnp.array([4.0]), score=30.0)
    assert buffer.scores[0] == 30.0
    assert jnp.abs(buffer.params[0] - 4.0) < 1e-5
    assert buffer.write_ptr[0] == 1  # pointer advanced to 1
    
    print("Insert index softmax (FIFO) test passed!")

if __name__ == "__main__":
    test_softmax_sampling()
    test_insert_idx_softmax()
    test_uniform_sampling()
    test_plr_sampling()
    test_plr_temperature()
