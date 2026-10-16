# Series-join context on the licensed 0.949 reference

The exact attributed reference has a concatenated 96-slice representation with
five fixed series slots [26,22,18,12,18]. Its full 94-center evaluation includes
**eight triples with donors from different series**, including different planes.
This is directly observed in pinned code and synthetic indexing, not a proven
accuracy defect. No original weights, clinical labels or cohorts are changed.

Two predeclared inference candidates:

1. **Edge repeat (primary):** keep all 94 centers; replace a donor across a series
   join with its own center image. This retains potentially valuable peripheral
   evidence and never invents pixels. Index replacement happens on uint8 data
   before the original per-channel ImageNet normalization and anatomical mirror.
2. **Within series (secondary):** remove the mixed-series windows, normally 94→86.
   All retained normalized pixels are unchanged. If every repeated candidate is
   a join, explicitly retain the original bag instead of dropping the study.

The same licensed author-trained model, preprocessing, series selection, head,
sigmoid, anatomical TTA and rank output remain. CPU/GPU constructor parity runs
on deterministic synthetic uint8 volumes on both T4s before the real public MRI
run. Public acceptance also requires the complete actual test.csv roster, all
13 columns, finite probabilities, zero public model failures and original weight
SHA. A public three-study pass is not a hidden score.

These are adaptations of an **external** 0.949 result, not own-training gains or
independent internal validation. The parent used our weak holdout and Gold58 in
its development; do not evaluate/tune these candidates on those internal labels.
Use competition outcomes when slots are available. Preserve losses as well as
wins. No benchmark or clinical claim exists before actual execution/outcomes.

The original external body stays in ignored private artifacts. Its source SHA is
493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde; checkpoint SHA
7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef. Original public
Dataset12431568/v1 and the existing explicit licensed-model admission apply; fresh
permission-description and private parent version checks precede a push. Only our
small indexing helper and surgical source builder are added to Git.

The canonical account ledger gains a separate inference_variants table, keyed by
actual code/policy/weight identity, so the already-scored parent weight record is
never overwritten or presented as new checkpoint bytes. All mutations still use
the single coordinator claim, durable once-only intents and actual source/version
checks. Uncertain mutations retain the claim for read-only reconciliation.

Actual zero-GPU CPU934627 staged both candidates once under a fresh 16 MiB node
control budget; the finite public queue is running and waiting for the original
paired evaluation to finish. Neither variant has been pushed yet. No new Dataset,
weight transfer or competition submission occurred. Leave healthy training and the
paired evaluation untouched; serialize the two public runs after actual capacity
becomes available. See the [actual stage audit](../../../audits/released-series-boundary-stage-20261016-v1/README.md). Five submission slots are already used
today; no public example CSV is ever uploaded as a competition submission.
