"""
Utilities for computing coalition values via weighted PageRank for COLE.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, 
                   temperature: float = 1.0) -> jnp.ndarray:
    """Softmax restricted to masked positions; non-masked positions → 0.

    Args:
        logits:      Shape (n,).
        mask:        Boolean shape (n,); True for active positions.
        temperature: Scales logits *before* softmax.  Higher → more uniform.

    Returns:
        Shape (n,).  Values at masked positions sum to 1; elsewhere 0.
    """
    scaled = logits / temperature
    # Flood non-coalition slots with -inf so they don't contribute to exp-sum.
    neg_inf = jnp.finfo(logits.dtype).min
    safe_scaled = jnp.where(mask, scaled, neg_inf)
    # Subtract max for numerical stability (-inf slots stay -inf after shift).
    stable = safe_scaled - jnp.max(safe_scaled)
    exp_vals = jnp.where(mask, jnp.exp(stable), 0.0)
    return exp_vals / (jnp.sum(exp_vals) + 1e-8)


def coalition_pagerank(
    payoffs: jnp.ndarray,
    mask: jnp.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """PageRank restricted to the coalition sub-graph.

    The key challenge is that the coalition membership changes across Shapley samples,
    but JAX requires static array shapes for JIT compilation. We use masking to
    compute PageRank over the coalition sub-graph.

    - Non-coalition rows/columns are zeroed in the adjacency matrix.
    - Dangling mass (from coalition members with no outgoing edges) is
      redistributed uniformly only among coalition members (not the whole
      graph), so random-walk probability mass stays inside the sub-graph.
    - After convergence, non-coalition PageRank scores are exactly 0.

    Args:
        payoffs:  (n, n) float — raw pairwise payoff matrix.  Negative
                  entries are clipped to 0 so all edge weights are ≥ 0.
        mask:     (n,) boolean — True for coalition members.
        damping:  PageRank damping factor (default 0.85).
        max_iter: Maximum number of power iterations.
        tol:      L1 convergence tolerance.

    Returns:
        (n,) float — PageRank scores.  Non-member scores are 0.
    """
    mask_f = mask.astype(payoffs.dtype)              # float version of mask
    coalition_size = jnp.sum(mask_f)                 # scalar |S|

    # ── Build masked adjacency (clip negatives; PageRank needs weights ≥ 0) ──
    adj = jnp.clip(payoffs, a_min=0.0)
    adj = adj * mask_f[:, None] * mask_f[None, :]   # zero non-member rows/cols

    # ── Stochastic transition matrix M ───────────────────────────────────────
    out_degree = jnp.sum(adj, axis=1)               # (n,)
    # Dangling: coalition member with no outgoing edges after masking
    is_dangling_coalition = ((out_degree == 0) & mask).astype(payoffs.dtype)
    safe_out_degree = jnp.where(out_degree == 0, 1.0, out_degree)
    M = adj / safe_out_degree[:, None]               # row-stochastic within mask

    # ── Teleportation target: uniform over coalition members ─────────────────
    teleport = mask_f / (coalition_size + 1e-8)      # (n,)

    # ── Initial PR: uniform over coalition ───────────────────────────────────
    pr = teleport.copy()
    prev_pr = jnp.zeros_like(pr)

    def cond_fun(state):
        pr, prev_pr, i = state
        diff = jnp.linalg.norm(pr - prev_pr, ord=1)
        return jnp.logical_and(i < max_iter, diff > tol)

    def body_fun(state):
        pr, _, i = state
        dangling_mass = jnp.dot(pr, is_dangling_coalition)
        new_pr = (
            (1.0 - damping) * teleport
            + damping * (jnp.dot(pr, M) + dangling_mass * teleport)
        )
        # Numerical safety: mask out non-coalition slots
        new_pr = new_pr * mask_f
        return new_pr, pr, i + 1

    final_pr, _, _ = jax.lax.while_loop(cond_fun, body_fun, (pr, prev_pr, 0))
    return final_pr


def coalition_value(
    mask: jnp.ndarray,
    payoffs: jnp.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    sigma_temperature: float = 10.0,
) -> jnp.ndarray:
    """Compute the value of a coalition using PageRank.

    Algorithm
    ---------
    1. Build the coalition sub-graph S by masking payoffs.
    2. Run PageRank on the sub-graph, PR(S), to rank agents by centrality.
    3. Compute difficulty weights:

           sigma = masked_softmax(-masked_softmax(PR(S), temp=1), temp=sigma_temp)

    High σ is assigned to agents with low PageRank — i.e., those who are hard to
    cooperate with — so that coalitions which succeed against tough partners are
    valued more.

    4. Value = (1/|S|) Σ_{i∈S} Σ_{j∈S}  sigma[j] · payoffs[i,j], clipped ≥ 0.
       Empty coalitions return 0.

    Args:
        mask:              ``(n,)`` boolean — ``True`` for coalition members.
        payoffs:           ``(n, n)`` float — pairwise payoff matrix.
        damping:           PageRank damping factor.
        max_iter:          Maximum PageRank iterations.
        tol:               PageRank convergence tolerance.
        sigma_temperature: Temperature for the outer (inverting) softmax.

    Returns:
        Scalar coalition value ≥ 0.
    """
    mask_f = mask.astype(payoffs.dtype)
    coalition_size = jnp.sum(mask_f)                 # |S|

    # ── Step 1: PageRank over coalition sub-graph ─────────────────────────────
    pr = coalition_pagerank(payoffs, mask,
                            damping=damping, max_iter=max_iter, tol=tol)

    # ── Step 2: Difficulty weights σ ─────────────────────────────────────────
    # Inner softmax normalises PR scores to a probability distribution.
    pr_softmax = masked_softmax(pr, mask, temperature=1.0)

    # Outer softmax over *negative* PR-softmax: low-PR agents get high weight.
    sigma = masked_softmax(-pr_softmax, mask, temperature=sigma_temperature)

    # ── Step 3: Weighted double sum ───────────────────────────────────────────
    # Vectorised: (payoffs @ sigma)[i] = Σ_j sigma[j] * payoffs[i,j]
    # Then dot with mask_f to sum over coalition members i.
    weighted_payoffs = payoffs @ sigma               # (n,)
    value = jnp.dot(mask_f, weighted_payoffs) / (coalition_size + 1e-8)

    # ── Step 4: Clip and handle empty coalition ───────────────────────────────
    value = jnp.maximum(value, 0.0)
    value = jnp.where(coalition_size == 0, 0.0, value)

    return value


def shapley_values(
    key: jnp.ndarray,
    payoffs: jnp.ndarray,
    N: int,
    max_iter: int,
    damping: float = 0.85,
    pagerank_max_iter: int = 100,
    tol: float = 1e-6,
    sigma_temperature: float = 10.0,
) -> jnp.ndarray:
    """Monte-Carlo Shapley value estimator over a cross-play payoff matrix.

    Uses the exact Shapley coalition weights with sampled coalitions, fully
    vectorised over both players and samples via ``jax.vmap``.

    For each player i, we estimate:

        phi[i] = mean_weight * mean_marginal

    where over max_iter sampled coalitions S ⊆ players \\ {i}:

        marginal(S, i) = v(S ∪ {i}) − v(S)
        weight(S)      = |S|! · (N − |S| − 1)! / N!

    v(S) is computed via coalition_value, which applies coalition-aware
    weighted PageRank internally.

    Intuition: inverse-PageRank weighting
    --------------------------------------
    Within coalition_value, PageRank acts as a "ease-of-cooperation" score:
    high-PageRank agents are universal cooperators that most agents do well
    with. Inverting these scores (via a negated softmax) concentrates weight
    on low-PageRank agents — those that are *hard* to cooperate with.

    The coalition value therefore rewards performance against difficult
    partners more than against easy ones. 
    Consequently, phi[i] measures how much player i unlocks cooperation 
    with the otherwise-hard-to-coordinate members of each sampled coalition — 
    rewarding agents that are broadly compatible with the toughest teammates.

    Args:
        key:               JAX PRNG key.
        payoffs:           (N, N) float — cross-play payoff matrix.
        N:                 Population size (static).
        max_iter:          Number of coalition samples per player (static).
        damping:           PageRank damping factor.
        pagerank_max_iter: Maximum PageRank power iterations.
        tol:               PageRank convergence tolerance.
        sigma_temperature: Temperature for the difficulty-weight softmax.

    Returns:
        phi: (N,) float — Shapley value estimate for each player.
    """
    # Split one key per player so each player's samples are independent.
    player_keys = jax.random.split(key, N)    # (N, 2)

    def phi_for_player(i, player_key):
        """Estimate phi[i] using max_iter sampled coalitions."""

        # Split into max_iter sample keys.
        sample_keys = jax.random.split(player_key, max_iter)   # (max_iter, 2)

        def one_sample(sample_key):
            # ── Sample S ⊆ players \ {i} via Bernoulli(0.5) ─────────────────
            base_mask = jax.random.bernoulli(sample_key, p=0.5, shape=(N,))
            s_mask = base_mask.at[i].set(False)   # S excludes player i
            s_with_i = s_mask.at[i].set(True)     # S ∪ {i}

            # ── Evaluate coalition value for S and S ∪ {i} ───────────────────
            v_s = coalition_value(
                s_mask, payoffs,
                damping=damping,
                max_iter=pagerank_max_iter,
                tol=tol,
                sigma_temperature=sigma_temperature,
            )
            v_s_with_i = coalition_value(
                s_with_i, payoffs,
                damping=damping,
                max_iter=pagerank_max_iter,
                tol=tol,
                sigma_temperature=sigma_temperature,
            )

            marginal = v_s_with_i - v_s

            # ── Shapley weight w(|S|) = |S|! · (N−|S|−1)! / N! ─────────────
            # Computed in log-space to handle large N without overflow.
            s_size = jnp.sum(s_mask).astype(jnp.float32)   # |S|
            log_weight = (
                (N - 1) * jnp.log(2.0)                      # log(2^{N-1}) to correct for uniform subset sampling
                + jsp.special.gammaln(s_size + 1)           # log(|S|!)
                + jsp.special.gammaln(N - s_size)           # log((N−|S|−1)!)
                - jsp.special.gammaln(N + 1)                # log(N!)
            )
            weight = jnp.exp(log_weight)

            return marginal * weight

        # Vectorise over the max_iter samples — returns (max_iter,) arrays.
        samples = jax.vmap(one_sample)(sample_keys)

        return jnp.mean(samples)

    # Vectorise over all N players simultaneously.
    players = jnp.arange(N)
    phi = jax.vmap(phi_for_player)(players, player_keys)
    return phi
