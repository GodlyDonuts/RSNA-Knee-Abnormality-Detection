# Matched full fitting template for paired logit correction

The proposed treatment is a detached parent correction to each sampled student
logit: full parent logit minus the parent logit on the exact same sampled windows
and view. It changes the training estimator while preserving the ordinary full
inference graph. At parent initialization the correction cancels deterministic
sampling error in the parent logits; nonlinear loss, encoder gradients and
dropout mean that this identity does not establish unbiased gradients or better
classification. The new actual encoder-noise experiment tests that limitation.

`scripts/kaggle/paired_logit_production.py` prepares a three-epoch, 3,887-study,
1,458-update treatment using the original accepted full-population compiler.
All source targets/masks, original teacher outputs, fitting order, IID16 draws,
proposal, effective batch, learning rates, warmup, precision, initialization and
inference remain matched to the original influence control. Only the explicit
paired readout, bound correction loader and minimal optimizer fork change.
The loader checks all 11,661 schedule identities before pixel reads. The existing
private table has already been constructed and preserved; do not reconstruct or
upload it again here. A future Kaggle attachment would be a separate one-time
mutation with its own prior intent and receipt.

The template requires the actual exact original paired recovery139417155 before
compilation. That outcome is not available yet. Ten new synthetic compilation
and failure cases check continuous identical worker/bindings across epochs,
rejection of changed source/table/recipe, pending or partial recovery, ordinary
candidate checkpoints and incomplete optimizer resumption, plus UTF8-safe literal
replacement. These do not execute a real full source or a GPU fit. A new focused
test caught and fixed an output-filename replacement-order error before use.

Compilation does not admit execution. Before any full run, inspect the actual
encoder-noise and forward error with reference uncertainty, empirical mean error
and actual whole-step overhead against the predeclared decision. Forward/backward
timing alone is not whole-step timing. Require an explicit agent decision,
funded immutable checkpoints/publication and fresh node/account budgets. Keep
original live training and version identities unchanged. No corrected full fit,
new Dataset, GPU reservation, Kaggle notebook or competition entry was made by
this preparation. It has no new clinical or competition accuracy result.
