# Measure the correction against a model that actually learned

The synthetic-drift screen cannot establish behavior after encoder updates.
`cached_attention_diagnostics.py` computes the linearized estimator's exact IID
logit variance and squared bias relative to the current full-bag attention.
Their sum is its exact diagnostic logit MSE. It also measures ordinary sampled
normalized attention on common128 IID16 draws. The oracle uses FP64 arithmetic;
it is not bitwise deployment equivalence, an encoder-gradient expectation, or
clinical accuracy. No recipe is automatically selected by its output.

Eight focused CPU cases passed, including exhaustive enumeration of every ordered
IID pair on a three-window bag and an untouched-global-RNG check. These prove the
oracle's calculation, not actual learned behavior.

The prepared owning-node GPU worker requires actual original977100 acceptance,
loads each arm's six-update immutable final snapshot and independently reproduces
its decoded model digest from the original recovery process. It uses four of the
17 engineering fitting studies, both views/all94 windows, to compare current
readouts against the original reference. It creates no optimizer, labels, new
weights, cache copies, report reads or accuracy score. Arrays remain ephemeral
on evc48; only aggregate per-target moments may leave the node. Neither tiny
engineering fidelity nor the recovery test can choose a clinical winner.

**Prepared only:** no GPU admission, additional funding or dispatch exists.
Before use, require original977100 terminal acceptance, immutable wrapper/worker/
oracle bindings, fresh quota and all standing reserves, a separately reviewed
single-dispatch coordinator, and a bounded one-H100/two-CPU scope on the original
physical two-GPU owning node. Do not change the original queued recovery to add
this work, repeat it, or reserve a GPU before prerequisites exist.


### Mechanism limit: fidelity is not new supervision

For binary cross-entropy distillation with fixed teacher probability p and
student logit z, the logit derivative is sigmoid(z)-p. At the exact full-bag
parent initialization with the same teacher, this derivative is zero. Therefore
source-zero fitting does not itself supply new diagnostic information: it tests
removal of source-loss conflict and the effects of the sampled training graph.
Sampling/noise/backend differences can still change the result, so this is not a
prediction of its Kaggle outcome or authority to cancel the original fit.
Likewise lower cached-attention readout error would establish a numerical
mechanism, not new clinical knowledge or an accuracy improvement. Any production
claim still needs actual held-out evidence; an improvement beyond the parent
needs a useful changed inductive bias and/or informative legitimate supervision,
not simply more precise self-imitation.
