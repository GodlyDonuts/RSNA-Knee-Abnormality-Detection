# Full bag gradient review for the CoAtNet challenger

14 October 2026. Mathematical review and primary-source inspection; no executed experiment.

## October18 implementation update

The rejection below describes the October14 declared train-mode-BatchNorm recipe. It does not describe current normalization: `licensed_parent_train.train_with_frozen_bn` explicitly calls `eval()` on every BatchNorm module, and `paired_logit_train.fit` invokes it before training. Current completed epoch receipts additionally assert parent BatchNorm buffers were preserved. Frozen buffers alone would not prove eval-mode; the source establishes it.

Existing `licensed_full_bag.CheckpointedEncode` already implements full-bag activation recomputation with frozen-BN enforcement/RNG preservation and has synthetic gradient/buffer tests. This is existing work, not a newly invented method. The current paired trainer still uses sampled windows and a corrected readout. Removing the earlier BN objection does not remove encoder stochastic replay, compute-cost or variance requirements, nor establish an accuracy benefit. Any new gradient-estimator experiment must compare against the actual current recipe and unchanged held-out population; the older rejection must not be used as a blanket prohibition or a reason to claim a breakthrough.

## Conclusion

Reject this as a drop-in change to the final-week CoAtNet recipe. Full-bag forward with sampled encoder gradients is mathematically sound for a window-separable encoder and can eliminate sampled-bag supervision mismatch. But the declared train-mode BatchNorm invalidates ordinary 12-window replay, the compute still rises roughly threefold versus the existing fit, and the estimator family is established rather than new.

The strongest relevant prior art is ICML 2023’s [unbiased full-set gradient method](https://proceedings.mlr.press/v202/willette23a.html). Its MIL experiment deliberately uses ResNet18 with InstanceNorm, confirmed in Table 12 and the [official implementation](https://github.com/jeffwillette/umbc/blob/master/umbc/models/layers/resnet.py#L94-L102). That result does not validate train-mode CoAtNet BN.

## Exact estimator and assumptions

Let z_i = f_θ(x_i) be the trunk embedding for window i. Put the proposed embedding LayerNorm inside the head H_φ. For the full bag:

L(θ,φ) = loss(H_φ(z_1,…,z_M), y)

Compute all current embeddings without encoder graphs. Differentiate the complete head once, obtaining:
- Exact head gradient ∂L/∂φ
- Cached embedding adjoints q_i = ∂L/∂z_i

Sample windows S with known positive inclusion probabilities π_i. Replay them and differentiate this scalar surrogate only with respect to encoder parameters:

R(θ) = Σ[i∈S] stop_gradient(q_i / π_i) · f_θ(x_i)

Its gradient is Σ[i∈S] J_iᵀq_i/π_i. Conditional on the full forward, its expectation equals Σ[i=1…M] J_iᵀq_i, the full-bag encoder gradient.

Required conditions:
- Replayed features and Jacobians correspond to the original forward: same weights, inputs, normalization function and random realization
- No parameter update between caching and replay
- Each encoder output is computable independently, or sampling/replay includes its entire dependency group
- Every potentially nonzero contribution has positive, correctly calculated inclusion probability
- Cached adjoints and sampling weights are detached; head gradients are computed once and are not multiplied by the encoder correction
- Loss masking and effective-batch denominator remain unchanged

This estimates the gradient of the actual full-bag loss. It does not importance-weight a sampled softmax denominator, so it avoids that ratio-estimation bias.

## The concrete softmax MIL coefficients

For class c, let:
- α_ic = softmax_i(a_c(h_i)), with h_i = LayerNorm(z_i)
- p_c = Σ_i α_ic h_i
- v_c = D_cᵀw_c, where D_c is the realized pooled-feature dropout mask including inverted-dropout scaling
- t_c = v_cᵀp_c + b_c
- δ_c = mask_c × (sigmoid(t_c) − y_c) / active-cell denominator

Then:

∂L/∂h_i = Σ_c δ_c α_ic [v_c + {v_cᵀ(h_i − p_c)}∇h_i a_c(h_i)]

Multiply by the LayerNorm Jacobian to obtain q_i. Both value and attention-score paths matter; using attention weights alone as replay coefficients is incorrect.

## The existing stratified sampler has tractable weights

For series s with m_s windows, define r_s = min(2,m_s), R = Σ_s r_s, K = min(12,M), and F = K−R. When M>R:

π_i = r_s/m_s + (1−r_s/m_s) × F/(M−R)

If every candidate is selected, π_i=1.

For six complete series, π_i=1/7. For [14,14,14,14,14,2], long-series windows still have π_i=1/7, but both short-series windows have π_i=1. Multiplying every replay gradient by seven would therefore be wrong.

## Why BatchNorm breaks the proposed replay

The [current timm source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py) confirms that this CoAtNet configuration uses MBConv BatchNorm; its transformer LayerNorm does not remove the earlier coupling. [PyTorch BatchNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html) computes training moments across windows and spatial positions.

Three distinct problems follow:
- Recomputing BN on twelve selected windows changes both embeddings and Jacobians relative to an 84-window BN forward
- Reusing detached full-bag means/variances matches forward values but omits derivatives through those moments
- A no-grad train-mode forward still updates running buffers; naive replay updates them again

For one BN channel with K normalized positions, upstream adjoints u and normalized activations x̂, its input gradient contains:

γ/σ × [u − mean(u) − x̂ mean(u x̂)]

Here σ = √(variance + ε). Those global correction terms generally give nonzero gradients to windows whose output adjoints were zero. Consequently, sampling output adjoints and performing the exact coupled BN backward does not generally restrict encoder backward work to twelve windows.

There is a valid qualification: partition the full bag into fixed normalization groups, compute train-mode BN separately within each group, and sample/replay entire identical groups. The group-level inverse-probability estimator is unbiased for that partitioned, chunk-BN objective. It is not equivalent to an 84-window BN forward, and introduces partition and buffer-update semantics requiring a changed recipe.

Freezing BN moments or using an already window-local LayerNorm/GroupNorm/InstanceNorm encoder restores the simple theorem. Replacing or freezing the admitted CoAtNet normalization is itself a scientific change with unmeasured transfer consequences.

## Dropout and RNG

The declared trunk has dropout and stochastic depth disabled, simplifying replay. Head dropout is harmless if the head is evaluated once and its realized adjoints are reused.

With stochastic encoder layers, replay must reproduce the exact masks. Merely resetting one seed before a differently shaped subset need not do that. Independent masks generally bias the result because the cached adjoint and replay Jacobian must correspond to the same random forward. The established [GradCache implementation](https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py#L1018-L1141) records RNG context and replays dot-product surrogates.

## Compute and memory

Let F and B denote one window’s encoder forward and backward cost. These cost symbols are local to this section; F above denoted the sampler’s number of fill draws. Ignore the small head and implementation overhead:

- Existing twelve-window fit: 12F + 12B
- Proposed full pass plus replay: 96F + 12B
- Ordinary exact GradCache over 84 windows: 168F + 84B
- Monolithic full-bag training: 84F + 84B

There is a simpler implementation when sampling is chosen before the forward: encode 72 windows without graphs, then the selected twelve with graphs, and compute the head over all 84. Inject a 1/π_i multiplier only into each selected embedding’s encoder derivative. This costs 84F + 12B, without replay.

With illustrative B≈2F:
- Replay: 3.33× the existing fit’s arithmetic
- Preselected one-pass version: 3×
- Respectively about 36% and 32% of exact GradCache arithmetic

Both sampled methods and ordinary GradCache can have twelve-window activation memory. Sampling therefore primarily saves backward compute relative to GradCache, not another sevenfold memory improvement. Cached 84×1,024 FP32 features occupy only 0.328 MiB; cached adjoints add the same amount. Parameter and optimizer memory do not shrink.

These equivalence and memory comparisons assume the valid window-separable encoder or a declared matching dependency-group objective. Ordinary chunked GradCache does not itself repair full-bag train-mode BN coupling.

## The unresolved variance is consequential

If one window supplies the only encoder-gradient contribution g and π=1/7, the estimator is 7g with probability 1/7 and zero otherwise. Its error variance is 6‖g‖². The head sees all evidence every time, but that focal window’s encoder learning remains intermittent.

Unbiasedness applies before gradient clipping. The recipe’s global norm clip at three can systematically suppress rare sevenfold updates; AdamW also does not reproduce the full-gradient optimization trajectory merely because the raw estimator is unbiased.

## Concrete falsifier already established analytically

Use two scalar pre-BN activations x=[θ−1, θ+1], BN affine scale one and bias zero, mean pooling, and positive-label BCE.

True train-mode BN gives:
- μ=θ, variance=1
- Pooled logit=0 for every θ
- True encoder gradient=0

Cache these moments, detach them, and replay either window with inclusion probability 1/2. The corrected replay gradient is:

−1 / (2√(1+ε))

It is nonzero for either choice. Thus detached full-bag moments fail even an exact two-window enumeration; no MRI experiment is needed to reject that repair.

## Decision

Final week: retain the declared candidate and its caveats; do not add this intervention or normalization changes.

Future work: the preselected one-pass estimator is a legitimate, established option if full-bag supervision is the chosen objective and the encoder is already window-local. Ordinary exact GradCache is the correctness reference. Any future claim must first match that reference in expected raw gradients, then establish a useful wall-time/variance tradeoff. Nothing here demonstrates an AUC gain or justifies another current fit.

Scope completed: mathematical derivation, synthetic counterexample, local specification review and primary-source inspection only. No data, weights, training, repository changes or publication.
