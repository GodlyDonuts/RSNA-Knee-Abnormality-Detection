# Conditional density learning without exam gradients through the evidence branch

Local implementation only. No new GPU run, production training, serving candidate
or accuracy result is admitted. Original B/C/C0 and C100 sources/jobs stay unchanged.
Actual C100 pilot acceptance and a separately funded matched pilot are prerequisites.

CPU 982819 measured the original C24 checkpoint on two fixed fitting cases. The
diagnosis-specific density component's gradient on the shift head, after lambda = 0.1,
was only 5.404e-6/4.036e-6 of the exam gradient. Common Gaussian reconstruction
dominated the density gradient. These are FP32 evaluation-mode, nonencoder,
pre-clipping measurements, not Adam updates or held-out accuracy. See the
[actual decomposition](../predictive-density-decomposition-20261020-v1/actual-C24-result.json).

For one study, K observed labels, two masks, b_m=logBF_m,
p_m=sigmoid(a_m+b_m), P=(p_1+p_2)/2 and n_m=64*N_observed,m:

    d L_exam / d b_m = (P-q) p_m(1-p_m) / [2 K P(1-P)]
    d (lambda D) / d b_m = -lambda q / (2 K n_m)

Both multiply the same Bayes-factor Jacobian for evidence-only parameters.
Descriptor scaling changes the geometry but cannot remove this relative 1/n_m
factor. The current formula faithfully implements the supplied proposal. It is
not a discovered coding error. A high common density gradient alone does not
establish strong diagnosis-specific learning.

The single proposed intervention retains C100's exact forward objective values:

    L = BCE(mean_m sigmoid(a_m + stop_gradient(b_m)), q)
        + 100 * original_proper_density_loss

The new explicit `generative_two_mask_loss` creates temporary Evidence views
with only `log_bayes_factor` detached. Original `log_p0` and `log_p1` remain
attached, so the full proper positive/negative density gradient is preserved.
In particular, log_p1 is not rebuilt from a detached Bayes factor. Original
Evidence objects and inference remain unchanged. The context prior receives
the same exam gradient at identical weights; the shared encoder receives exam
gradients through the context prior plus density gradients through the original
predictor. Negative/shift heads receive only the density objective. The wrapper
requires an explicit coefficient and is not wired into any existing trainer.

This candidate requires a **positive density coefficient**. Do not apply it to
C0: with density weight zero, neither objective trains the evidence-only heads.
The current zero-initialized negative/shift heads would remain neutral and yield
zero Bayes factors; other initializations would retain untrained heads. That
would change the intended C0 control rather than test generative learning.

The future comparison is C100 versus this routed C100, with the same inputs,
initialization, masks, supports, optimizer and update budget. Require matched
numerical recovery and stability through peak learning rate before any full fit.
At fixed final, assess actual conditional descriptor fit and predeclared target
permutation/mean interventions with context fixed, alongside B's generic anatomy
control and the allowed held-out evaluation. Permutation sensitivity by itself is
not clinical accuracy or diagnosis-specific evidence.

This removes a competing classification gradient; it does not prove useful
generative evidence. Strong visible context may make the diagnosis condition
redundant, and weak labels or insufficient counterfactual coverage may leave the
two hypotheses poorly identified. A Bayes-factor-only auxiliary BCE would still
be discriminative. Subtracting a stopped-gradient baseline NLL changes no
gradients. Scaling the negative logBF component independently can make the
objective unbounded, so the proper density term remains intact. No contrast,
descriptor normalization, covariance change or raw-input rebuild is bundled here.

## Prepared matched experiment

`worker_builder.py` derives the exact original `39e315…` engineering worker.
Both roles use C's graph, lambda100, the original twelve fitting cases, seed20261020,
effective batch8 with the four-case tail, three head-only updates, and the full
2,916-update learning-rate schedule. The prepared run executes **200 updates per
role**, passing the production peak at update146. This is100 passes over twelve
engineering cases, not100 full-data epochs. The four disjoint engineering cases
remain prediction/recovery checks and cannot select a recipe.

`driver.py` runs C100 full200, G100 full200, G100 stop1 and G100 resume199: **600
actual updates** within a90-minute one-H100/two-CPU bound. It checks identical
fresh initial state, real encoder/head parameter changes, finite complete
histories, and exact G100 full/resume model, AdamW, counters, all RNG and prediction
bytes. G100's only numerical intervention is routing the exam gradient as above.
The original train, density and model modules remain byte-identical.

The driver requires original C100 pilot981620's actual successful terminal
receipt, an explicit subsequent agent engineering decision, separate fresh
capacity for6GiB checkpoints plus4MiB controls, and all older storage obligations.
Worker, driver, CPU admission and coordinator are locally prepared and tested.
Prepared shared-log ownership binds original controller `2a224ae0`/owner32377 and
the actual `7dcd87e3`/33107 continuation, including both original intent hashes.
All primary jobs and paid log proofs belong to evc27; an uncertain original
dispatch can be adopted only through matching sealed-job reconciliation records.
The original981620 pilot, its16 engineering inputs and its runtime remain
unchanged. G100 retains all36GiB of primary fitting scope,128MiB placement
controls, the full C100 future scope and the
existing descriptor-normalizer1MiB liability. Full C100 likewise reserves the
independent G1006GiB+4MiB scope, without speculative paid-byte credit, so optional
execution order cannot spend the other branch’s storage. These are conservative
future obligations, not job authority. The two prepared optional routes passed123
focused local tests and independent review. **No matched GPU job or full G100 fit is staged,
submitted or accepted.** The current C100 pilot and primary B/C/C0 jobs are
unchanged. There is no automatic promotion or claimed clinical improvement.

The prospective G100 admission also retains the separately admitted contrast
pilot's **7GiB +4MiB** on evc27, without claiming paid-byte credit. This addition
affects only future admission. The original C100 and contrast GPU scripts remain
immutable; fresh quota and the actual C100 outcome still govern execution.
