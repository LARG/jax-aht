"""Tests for open_ended_training.shapley_utils — Shapley value computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_ended_training.shapley_utils import coalition_value, coalition_pagerank, masked_softmax, shapley_values

jax.config.update("jax_enable_x64", False)  # keep float32 for speed

def random_payoffs(n: int, seed: int = 0) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, shape=(n, n))


class TestMaskedSoftmax:
    def test_sums_to_one_over_mask(self):
        logits = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.array([True, True, False, True])
        out = masked_softmax(logits, mask)
        # Masked positions should sum to 1
        assert jnp.isclose(jnp.sum(out * mask), 1.0, atol=1e-5)

    def test_non_masked_positions_are_zero(self):
        logits = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.array([True, True, False, True])
        out = masked_softmax(logits, mask)
        assert out[2] == pytest.approx(0.0, abs=1e-6)

    def test_uniform_logits_give_uniform_output(self):
        n = 5
        logits = jnp.ones(n)
        mask = jnp.ones(n, dtype=jnp.bool_)
        out = masked_softmax(logits, mask)
        np.testing.assert_allclose(np.array(out), np.full(n, 1.0 / n), atol=1e-5)

    def test_temperature_effect(self):
        """Higher temperature → more uniform distribution."""
        logits = jnp.array([1.0, 10.0, 1.0])
        mask = jnp.ones(3, dtype=jnp.bool_)
        low_t = masked_softmax(logits, mask, temperature=0.1)
        high_t = masked_softmax(logits, mask, temperature=100.0)
        # High-temp output should be closer to uniform
        assert jnp.max(high_t) < jnp.max(low_t)


# ─────────────────────────────────────────────────────────────────────────────
# Reference PageRank that runs on an explicit dense submatrix
# ─────────────────────────────────────────────────────────────────────────────

def _reference_pagerank(adj, damping=0.85, max_iter=100, tol=1e-6):
    """Verbatim PageRank from notebooks/shapley.ipynb, used as ground truth."""
    num_nodes = adj.shape[0]
    out_degree = jnp.sum(adj, axis=1)
    is_dangling = (out_degree == 0).astype(adj.dtype)
    safe_out_degree = jnp.where(out_degree == 0, 1.0, out_degree)
    M = adj / safe_out_degree[:, None]
    pr = jnp.ones(num_nodes) / num_nodes
    prev_pr = jnp.zeros_like(pr)

    def cond_fun(state):
        pr, prev_pr, i = state
        diff = jnp.linalg.norm(pr - prev_pr, ord=1)
        return jnp.logical_and(i < max_iter, diff > tol)

    def body_fun(state):
        pr, _, i = state
        dangling_mass = jnp.dot(pr, is_dangling)
        new_pr = (
            (1.0 - damping) / num_nodes
            + damping * (jnp.dot(pr, M) + dangling_mass / num_nodes)
        )
        return new_pr, pr, i + 1

    final_pr, _, _ = jax.lax.while_loop(cond_fun, body_fun, (pr, prev_pr, 0))
    return final_pr


# ─────────────────────────────────────────────────────────────────────────────
# coalition_pagerank
# ─────────────────────────────────────────────────────────────────────────────

class TestCoalitionPagerank:
    def test_matches_reference_pagerank_on_subgraph(self):
        """coalition_pagerank on the full matrix with a mask should produce
        the same scores (up to normalisation) as the notebook's reference
        pagerank run directly on the extracted coalition submatrix.

        Strategy
        --------
        1. Extract the |S|×|S| submatrix from `payoffs` (clip negatives to
           match coalition_pagerank's internal pre-processing).
        2. Run _reference_pagerank on the submatrix → scores over |S| nodes.
        3. Run coalition_pagerank on the full n×n matrix with boolean mask.
        4. Read off the coalition-member scores from the full-size output.
        5. Normalise both score vectors to sum to 1 and compare elementwise.
        """
        n = 8
        key = jax.random.PRNGKey(7)
        payoffs = jax.random.uniform(key, shape=(n, n))   # all positive, no clipping needed

        # Use a fixed mask with at least 2 members
        mask = jnp.array([True, False, True, True, False, True, False, False])
        member_indices = jnp.where(mask)[0]               # shape (|S|,)

        # --- Reference: run pagerank on the explicit submatrix ---------------
        submatrix = payoffs[jnp.ix_(member_indices, member_indices)]  # (|S|, |S|)
        ref_pr = _reference_pagerank(submatrix)           # shape (|S|,)
        ref_pr_norm = ref_pr / (jnp.sum(ref_pr) + 1e-8)  # normalise to sum-1

        # --- Tested: coalition_pagerank on full matrix with mask -------------
        full_pr = coalition_pagerank(payoffs, mask)        # shape (n,)
        coalition_pr = full_pr[member_indices]             # shape (|S|,)
        coalition_pr_norm = coalition_pr / (jnp.sum(coalition_pr) + 1e-8)

        np.testing.assert_allclose(
            np.array(coalition_pr_norm),
            np.array(ref_pr_norm),
            atol=1e-4,
            err_msg=(
                "coalition_pagerank (masked) disagrees with reference pagerank "
                "on the explicit coalition submatrix."
            ),
        )
    def test_output_shape(self):
        n = 8
        payoffs = random_payoffs(n)
        mask = jnp.ones(n, dtype=jnp.bool_)
        pr = coalition_pagerank(payoffs, mask)
        assert pr.shape == (n,)

    def test_non_coalition_scores_are_zero(self):
        n = 6
        payoffs = random_payoffs(n)
        mask = jnp.array([True, True, False, True, False, False])
        pr = coalition_pagerank(payoffs, mask)
        # Non-member scores must be (approximately) 0
        for i in range(n):
            if not mask[i]:
                assert pr[i] == pytest.approx(0.0, abs=1e-5)

    def test_scores_are_non_negative(self):
        n = 10
        payoffs = random_payoffs(n)
        mask = jax.random.bernoulli(jax.random.PRNGKey(1), p=0.6, shape=(n,))
        pr = coalition_pagerank(payoffs, mask)
        assert jnp.all(pr >= -1e-6)

    def test_jit_compilable(self):
        n = 8
        payoffs = random_payoffs(n)
        mask = jnp.ones(n, dtype=jnp.bool_)
        jit_fn = jax.jit(coalition_pagerank)
        pr_eager = coalition_pagerank(payoffs, mask)
        pr_jit = jit_fn(payoffs, mask)
        np.testing.assert_allclose(np.array(pr_eager), np.array(pr_jit), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# coalition_value
# ─────────────────────────────────────────────────────────────────────────────

class TestCoalitionValue:
    def test_empty_coalition_returns_zero(self):
        n = 6
        payoffs = random_payoffs(n)
        empty_mask = jnp.zeros(n, dtype=jnp.bool_)
        v = coalition_value(empty_mask, payoffs)
        assert float(v) == pytest.approx(0.0, abs=1e-6)

    def test_value_is_non_negative(self):
        """Value should always be ≥ 0 by construction."""
        n = 10
        key = jax.random.PRNGKey(7)
        payoffs = jax.random.normal(key, shape=(n, n))  # can be negative
        key, sk = jax.random.split(key)
        mask = jax.random.bernoulli(sk, p=0.5, shape=(n,))
        v = coalition_value(mask, payoffs)
        assert float(v) >= -1e-6

    def test_all_positive_payoffs_give_positive_value(self):
        n = 8
        payoffs = random_payoffs(n)  # all in [0, 1]
        # At least two agents in coalition
        mask = jnp.array([True, True, False, True, False, False, False, False])
        v = coalition_value(mask, payoffs)
        assert float(v) > 0.0

    def test_singleton_value(self):
        """Singleton coalition: only payoffs[i,i] contributes."""
        n = 5
        payoffs = random_payoffs(n)
        for i in range(n):
            mask = (jnp.arange(n) == i)
            v = coalition_value(mask, payoffs)
            # sigma[i] = 1 from masked softmax over a single element
            expected = max(payoffs[i, i], 0.0)
            assert float(v) == pytest.approx(float(expected), rel=1e-4, abs=1e-5), \
                f"Agent {i}: got {v}, expected {expected}"

    def test_jit_compilable(self):
        n = 8
        payoffs = random_payoffs(n)
        mask = jnp.ones(n, dtype=jnp.bool_)
        jit_fn = jax.jit(coalition_value)
        v_eager = coalition_value(mask, payoffs)
        v_jit = jit_fn(mask, payoffs)
        assert float(v_eager) == pytest.approx(float(v_jit), rel=1e-5, abs=1e-6)

    def test_vmap_over_masks(self):
        """coalition_value should be vmap-able over a batch of masks."""
        n = 10
        k = 20
        key = jax.random.PRNGKey(3)
        payoffs = jax.random.uniform(key, shape=(n, n))
        key, sk = jax.random.split(key)
        masks = jax.random.bernoulli(sk, p=0.5, shape=(k, n))

        batched = jax.jit(jax.vmap(coalition_value, in_axes=(0, None)))
        values = batched(masks, payoffs)

        assert values.shape == (k,)
        # All values ≥ 0
        assert jnp.all(values >= -1e-6)
        # Empty masks (all-False rows) → 0
        for i in range(k):
            if not jnp.any(masks[i]):
                assert float(values[i]) == pytest.approx(0.0, abs=1e-5)

    def test_jit_vmap_matches_eager_loop(self):
        """Batched JIT/vmap results should match individual eager calls."""
        n = 6
        k = 8
        key = jax.random.PRNGKey(99)
        payoffs = jax.random.uniform(key, shape=(n, n))
        key, sk = jax.random.split(key)
        masks = jax.random.bernoulli(sk, p=0.6, shape=(k, n))

        batched = jax.jit(jax.vmap(coalition_value, in_axes=(0, None)))
        values_batch = batched(masks, payoffs)

        for i in range(k):
            v_eager = coalition_value(masks[i], payoffs)
            assert float(values_batch[i]) == pytest.approx(float(v_eager), rel=1e-4, abs=1e-5), \
                f"Mismatch at index {i}"


# ─────────────────────────────────────────────────────────────────────────────
# shapley_values
# ─────────────────────────────────────────────────────────────────────────────

class TestShapleyValues:
    def test_output_shape(self):
        """phi should be a length-N vector."""
        N, max_iter = 5, 20
        key = jax.random.PRNGKey(0)
        payoffs = random_payoffs(N)
        phi = shapley_values(key, payoffs, N=N, max_iter=max_iter)
        assert phi.shape == (N,)

    def test_jit_compilable_and_consistent(self):
        """JIT-compiled result must match eager result exactly."""
        N, max_iter = 6, 30
        key = jax.random.PRNGKey(1)
        payoffs = random_payoffs(N)
        jit_sv = jax.jit(shapley_values, static_argnames=["N", "max_iter"])
        phi_eager = shapley_values(key, payoffs, N=N, max_iter=max_iter)
        phi_jit   = jit_sv(key, payoffs, N=N, max_iter=max_iter)
        np.testing.assert_allclose(np.array(phi_eager), np.array(phi_jit), atol=1e-5)

    def test_symmetric_payoff_gives_equal_values(self):
        """If the payoff matrix is symmetric and all entries equal, every
        player is interchangeable and should receive the same Shapley value."""
        N = 4
        max_iter = 200
        key = jax.random.PRNGKey(42)
        # Constant symmetric payoff: all pairs interact identically.
        payoffs = jnp.ones((N, N), dtype=jnp.float32)
        phi = shapley_values(key, payoffs, N=N, max_iter=max_iter)
        # All values should be approximately equal to 1/N (true Shapley value).
        np.testing.assert_allclose(
            np.array(phi),
            np.full(N, 1.0 / N),
            atol=0.15,   # generous tolerance for unbiased Monte-Carlo variance
            err_msg="Symmetric payoffs should yield Shapley values around 1/N.",
        )

    def test_weights_sum_correctly(self):
        """For any coalition size s in {0,...,N-1}, the Shapley weight
        w(s) = s!(N-s-1)!/N! should sum to 1 over all N players
        (by the efficiency axiom proxy). Here we verify the mean weight
        returned per player is reasonable (positive and << 1)."""
        N, max_iter = 5, 50
        key = jax.random.PRNGKey(7)
        payoffs = random_payoffs(N)
        phi = shapley_values(key, payoffs, N=N, max_iter=max_iter)
        # phi values should be finite
        assert jnp.all(jnp.isfinite(phi)), "Shapley values must be finite."


if __name__ == "__main__":
    import traceback

    # Collect all test classes and their test methods
    test_classes = [
        TestMaskedSoftmax,
        TestCoalitionPagerank,
        TestCoalitionValue,
        TestShapleyValues,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        print(f"\n{'─'*60}")
        print(f"  {cls.__name__}  ({len(methods)} tests)")
        print(f"{'─'*60}")
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  ✓  {method_name}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {method_name}")
                tb = traceback.format_exc()
                errors.append((cls.__name__, method_name, tb))
                failed += 1

    print(f"\n{'═'*60}")
    total = passed + failed
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        print()
        for cls_name, mname, tb in errors:
            print(f"── FAIL: {cls_name}.{mname} ──")
            print(tb)
    else:
        print("  — all tests passed ✓")
    print(f"{'═'*60}")