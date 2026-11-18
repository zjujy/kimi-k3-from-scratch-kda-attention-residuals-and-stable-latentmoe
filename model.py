"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - short_conv
def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    # TODO: accumulate each kernel tap with a shifted slice add
    T, d = x.shape
    K = w.shape[0]
    y = np.zeros_like(x)
    for t in range(T):
        for j in range(K):
            pos = t - (K - 1) + j
            if pos >= 0:
                y[t] += w[j] * x[pos]
    return y

# Step 2 - kda_qkv
def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    # TODO: project -> short_conv -> swish, then L2-normalize q and k rows
    q = short_conv(x @ params["Wq"], params["cq"])
    k = short_conv(x @ params["Wk"], params["ck"])
    v = short_conv(x @ params["Wv"], params["cv"])

    q = q / (1 + np.exp(-q))
    k = k / (1 + np.exp(-k))
    v = v / (1 + np.exp(-v))

    q = q / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + 1e-6)
    k = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)

    return q, k, v

# Step 3 - kda_gates
def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    # TODO
    beta = x @ params["wb"] + params["bb"]
    beta = 1 / (1 + np.exp(-beta))
    z = x @ params["Wd1"] @ params["Wd2"] + params["ba"]
    return beta, z

# Step 4 - lower_bounded_decay
def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    # TODO
    # scaled = np.exp(A) * z
    # s = np.exp(-np.logaddexp(0.0, -scaled))
    # g = g_min * s
    # alpha = np.exp(g)
    # alpha = np.minimum(alpha, np.nextafter(1.0, 0,0))
    return np.exp(g_min * 1 / (1 + np.exp(-np.exp(A) * z)))

# Step 5 - kda_state_update
def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    # TODO: decay, then delta-rule erase, then write
    S_decay = alpha[:, None] * S
    old_v = k @ S_decay
    delta = v - old_v
    S_new = S_decay + beta * np.outer(k, delta)
    return S_new

# Step 6 - kda_recurrence
def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    # TODO
    T, dk = q.shape
    dv = v.shape[1]
    if S0 is None:
        S = np.zeros((dk, dv), dtype=v.dtype)
    else:
        S = S0.copy()
    
    O = np.zeros((T, dv), dtype=np.result_type(q, v, S))

    for t in range(T):
        S = kda_state_update(S, k[t], v[t], alpha[t], beta[t])
        O[t] = S.T @ q[t]
    return O, S

# Step 7 - cumulative_decay
def cumulative_decay(alpha):
    """Inclusive channel-wise cumulative product of alpha down the time axis.

    alpha: (C, dk) per-step retention factors -> Gamma: (C, dk).
    """
    # TODO
    return np.cumprod(alpha, axis=0)

# Step 8 - chunk_pseudo_values
def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    # TODO
    C = k.shape[0]
    Gamma = cumulative_decay(alpha)
    Khat = k * Gamma
    Kcheck = k / Gamma
    M = np.tril(Khat @ Kcheck.T, k=-1)
    A = np.eye(C) + beta[:, None] * M
    B = beta[:, None] * (v - Khat @ S0)
    U = np.linalg.solve(A, B)
    return U

# Step 9 - kda_chunkwise
def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. Returns (O, S_final).
    """
    # TODO
    T, dk = q.shape
    dv = v.shape[1]
    if S0 is None:
        S = np.zeros((dk, dv))
    else:
        S = S0.copy()
    
    O = np.zeros((T, dv))

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_c = q[start: end]
        k_c = k[start: end]
        v_c = v[start: end]
        alpha_c = alpha[start: end]
        beta_c = beta[start: end]

        Gamma = cumulative_decay(alpha_c)
        Qhat = q_c * Gamma
        Kcheck = k_c / Gamma

        U = chunk_pseudo_values(
            k_c, v_c, alpha_c, beta_c, S
        )

        inter_chunk = Qhat @ S

        intra_chunk = (
            np.tril(Qhat @ Kcheck.T) @ U
        )

        O[start: end] = inter_chunk + intra_chunk
        
        S = Gamma[-1][:, None] * (
            S + Kcheck.T @ U
        )
    
    return O, S

# Step 10 - kda_output_gate
def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    # TODO
    gate = 1.0 / (1.0 + np.exp(-(x @ Wg)))
    rms = np.sqrt(np.mean(o ** 2, axis=-1, keepdims=True) + 1e-6)
    o_norm = o / rms
    gated_o = gate * o_norm
    y = gated_o @ Wo
    return y

# Step 11 - mla_compress_reconstruct
def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    # TODO
    T = x.shape[0]
    c = x @ Wc
    dh = Wk_up.shape[1] // n_heads

    K = (c @ Wk_up).reshape(T, n_heads, dh)
    V = (c @ Wv_up).reshape(T, n_heads, dh)
    
    return c, K, V

# Step 12 - nope_attention
def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    # TODO
    T, d = x.shape
    dh = Wq.shape[1] // n_heads

    Q = (x @ Wq).reshape(T, n_heads, dh)
    c, K, V = mla_compress_reconstruct(
        x, Wc, Wk_up, Wv_up, n_heads
    )
    O = np.zeros((T, n_heads, dh))

    for h in range(n_heads):
        Qh = Q[:, h, :]
        Kh = K[:, h, :]
        Vh = V[:, h, :]

        scores = Qh @ Kh.T / np.sqrt(dh)

        mask = np.triu(
            np.ones((T, T), dtype=bool), k=1
        )
        scores[mask] = -1e9

        scores = scores - np.max(
            scores, axis=-1, keepdims=True
        )
        exp_scores = np.exp(scores)
        attn = exp_scores / np.sum(
            exp_scores, axis=-1, keepdims=True
        )

        O[:, h, :] = attn @ Vh
    
    return O.reshape(T, n_heads * dh)

# Step 13 - mla_output_gate
def mla_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * o) @ Wo - note: no RMSNorm here, unlike KDA's gate.

    o: (T, H*dh) attention output.  x: (T, d) layer input.  Returns (T, d).
    """
    # TODO
    gate = 1.0 / (1.0 + np.exp(-(x @ Wg)))
    gated_o = gate * o
    y = gated_o @ Wo
    return y

# Step 14 - hybrid_schedule
def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    # TODO
    block = ['KDA', 'KDA', 'KDA', 'MLA']
    return block * n_repeats + ['MLA']

# Step 15 - attnres_weights
def attnres_weights(pseudo_q, sources):
    """Softmax over depth: w[i, t] prop. to exp(pseudo_q . RMSNorm(sources[i][t])).

    sources: list of n (T, d) arrays.  Returns (n, T); columns sum to 1.
    """
    # TODO
    K = np.stack(sources, axis=0)

    rms = np.sqrt(
        np.mean(K ** 2, axis=-1, keepdims=True) + 1e-6
    )
    K_norm = K / rms

    logits = K_norm @ pseudo_q
    logits -= np.max(logits, axis=0, keepdims=True)

    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=0, keepdims=True)

# Step 16 - attnres_full
def attnres_full(pseudo_q, sources):
    """h[t] = sum_i attnres_weights(...)[i, t] * sources[i][t] (raw values).

    Returns (T, d).
    """
    # TODO
    weights = attnres_weights(pseudo_q, sources)
    V = np.stack(sources, axis=0)
    weighted_V = weights[:, :, None] * V
    h = np.sum(weighted_V, axis=0)
    return h

# Step 17 - block_partial_sums
def block_partial_sums(layer_outputs):
    """Running sums of a block's layer outputs; entry i sums outputs 0..i.

    Returns a list of independent (T, d) arrays; last entry = block sum b_n.
    """
    # TODO
    partial_sums = []
    running_sum = None
    for output in layer_outputs:
        if running_sum is None:
            running_sum = output.copy()
        else:
            running_sum += output
        partial_sums.append(running_sum.copy())
    return partial_sums

# Step 18 - attnres_block
def attnres_block(pseudo_q, block_reps, partial):
    """Full AttnRes over [b_0..b_{n-1}] plus the current block's partial (if any).

    partial is None for the first layer of a block. Returns (T, d).
    """
    # TODO
    sources = list(block_reps)
    if partial is not None:
        sources.append(partial)
    return attnres_full(pseudo_q, sources)

# Step 19 - situ_glu
def situ_glu(x, Wg, Wu, beta1=4.0, beta2=25.0):
    """(softcap(x@Wg, b1) * sigmoid(x@Wg)) * softcap(x@Wu, b2), softcap = b*tanh(u/b).

    Use sigmoid(g) = 0.5*(1 + tanh(0.5*g)) to stay overflow-safe. |out| <= b1*b2.
    """
    # TODO
    g = x @ Wg
    u = x @ Wu
    g_cap = beta1 * np.tanh(g / beta1)
    u_cap = beta2 * np.tanh(u / beta2)
    gate = 0.5 * (1.0 + np.tanh(0.5 * g))
    return g_cap * gate * u_cap

# Step 20 - route_topk
def route_topk(x, Wr, bias, k):
    """s = sigmoid(x @ Wr); top-k by s + bias (stable, descending);
    p = raw selected scores normalized per token. Returns (s, idx, p).
    """
    # TODO
    logits = x @ Wr
    s = 0.5 * (1.0 + np.tanh(0.5 * logits))
    biased_scores = s + bias
    idx = np.argsort(
        -biased_scores, axis=1, kind='stable'
    )[:, :k]

    selected = np.take_along_axis(s, idx, axis=1)
    p = selected / selected.sum(axis=1, keepdims=True)

    return s, idx, p

# Step 21 - routed_experts
def routed_experts(z, idx, p, experts):
    """u[t] = sum_j p[t,j] * situ_glu(z[t], *experts[idx[t,j]]).

    z: (T, l).  experts: list of (Wg, Wu) latent-width SiTU-GLUs. Returns (T, l).
    """
    # TODO
    T, l = z.shape
    k = idx.shape[1]
    u = np.zeros_like(z)

    for t in range(T):
        for j in range(k):
            expert_id = idx[t, j]
            Wg, Wu = experts[expert_id]
            out = situ_glu(z[t:t + 1], Wg, Wu)
            u[t] += p[t, j] * out[0]
    return u

# Step 22 - stable_latent_moe
def stable_latent_moe(x, params):
    """y = sum_shared SiTU(x) + RMSNorm(routed_aggregate) @ Wup (Eq. 11).

    Route on full-width x; compute in latent width; RMSNorm(u) before Wup.
    """
    # TODO
    z = x @ params['Wdown']
    s, idx, p = route_topk(
        x, params['Wr'], params['bias'], params['k']
    )
    u = routed_experts(z, idx, p, params['experts'])
    u_norm = u / np.sqrt(
        np.mean(u**2, axis=-1, keepdims=True) + 1e-6
    )
    routed_out = u_norm @ params['Wup']
    shared_out = np.zeros_like(x, dtype=float)

    for Wg, Wu in params['shared']:
        shared_out += situ_glu(x, Wg, Wu)
    return shared_out + routed_out

# Step 23 - topk_cutoffs
def topk_cutoffs(s, bias, k):
    """Top-(k+1) on s + bias: first k -> routes (m, k); (k+1)-th biased score
    -> cutoffs (m,). Returns (routes, cutoffs).
    """
    # TODO
    biased_scores = s + bias
    order = np.argsort(
        -biased_scores, axis=1, kind='stable'
    )
    routes = order[:, :k]
    cutoff_idx = order[:, k]
    cutoffs = np.take_along_axis(
        biased_scores, cutoff_idx[:, None], axis=1
    )[:, 0]
    return routes, cutoffs

# Step 24 - quantile_balance_update (not yet solved)
# TODO: implement

# Step 25 - histogram_quantile (not yet solved)
# TODO: implement

# Step 26 - newton_schulz (not yet solved)
# TODO: implement

# Step 27 - per_head_muon (not yet solved)
# TODO: implement

# Step 28 - mini_k3_forward (not yet solved)
# TODO: implement

