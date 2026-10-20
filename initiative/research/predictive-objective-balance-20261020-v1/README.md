# Measure whether the new objective reaches the predictive heads

The full B/C/C0 input jobs are healthy. Their immutable numerical recipe stays
unchanged. This is a separate, fitting-only measurement on the already completed
24-update C engineering checkpoint; it is not another training run, an accuracy
test, or evidence about the future full model.

## The concrete question

The implemented density objective for one diagnosis and complementary mask is

`density = (-log p0 - q × logBF) / (64 × observed_regions)`.

Its production coefficient is 0.1. With 536 observed regions, its direct
log-Bayes-factor derivative relative to exam BCE near `q=1, p=0.5` is roughly
`0.2 / (64 × 536) = 5.83e-6`. The retained initial C density loss is
0.9195934609, but 0.9189385332 is the fixed Gaussian constant. Its variable
weighted value is therefore about 0.0000655, rather than the apparent 0.092.

These facts identify a scale risk, not a demonstrated failure. The negative
mean head initially receives density gradients that the exam term cannot supply
when the finding shift is zero. Adam can amplify small coordinatewise signals.
Similar combined gradient norms do not determine the auxiliary gradient vector.

Measure the actual two objectives separately, at the same original checkpoint
and on exactly fitting indices 0 and 1 of the original 12-case engineering
training partition. Retain both complete complementary contexts. Report
unweighted exam and **once-weighted** density gradient norms and cosines for the
nonencoder parameter groups, their vector-sum identity, and the observed
posterior contribution of the log Bayes factor relative to context logits.
Remove no observation regions or diagnosis masks.

The encoder runs on CPU in FP32 evaluation mode and its features are detached
for the head-gradient calculation. This is not a BF16 training-gradient replay,
an encoder-gradient measurement, an Adam update, or proof of optimization impact.
No individual labels, predictions, identifiers, images or checkpoint bytes leave
the owning node. Only sanitized aggregate numerical findings are collected.

## Execution scope

Use the already retained original stable-warmup C checkpoint on evc27, original
binding, exact accepted source modules, context cache and fitting targets. Bind
the completed original engineering receipt and inspect original state before
execution. Do not recreate the pilot, transfer its assets, change healthy input
producers, or modify staged full-fitting code.

One CPU allocation requests two cores, 12 GiB RAM, zero GPUs and at most 30
minutes. It streams its small source through stdin, writes no remote files and
has no new node-local or shared-storage liability. The launcher creates one
exclusive ignored local intent and retains any uncertain outcome without retry.
No credentials, network downloads, Kaggle operations, new checkpoints, parameter
updates, protected labels or model selection are involved.

If gradients are weak, preserve that finding and its limitations. Do not retune
the frozen matched comparison from this tiny measurement. It can motivate a
separately declared future intervention; it cannot establish which loss weight
improves competition accuracy.
