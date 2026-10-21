# Four-core preparation experiment

The prospective selected-frame pipeline currently uses two concurrent study
lanes, each with one PyTorch CPU thread. Test whether two threads per lane use
the four available CPU cores more effectively while healthy fitting continues.
This is a distinct private CPU-only experiment, not a model or submission.

The local synthetic check found a maximum image difference of one FP32 step
(1.1920928955078125e-7), with all other context fields exact. It therefore
failed the stronger bitwise replacement criterion. That tiny rounding change
does not establish clinical degradation. This follow-up measures the exact
Kaggle environment rather than silently treating the images as bit-identical.

Six two-lane trials use thread counts1,2,2,1,1,2 on all three public studies.
Discard the first two cold/warming trials from the throughput comparison; the
remaining balanced pairs retain their original ordering. Every complete
context is compared against the first one-thread trial. Image tensors must
remain finite FP32 with maximum absolute difference <=1e-6; all descriptor,
coordinate, mask, support and other values remain byte-identical. No images,
IDs, context hashes or private predictions are emitted. CPU affinity is limited
to four actual cores, memory24GiB, measured experiment780seconds and notebook
900seconds. No learned model, labels, new Dataset, GPU quota or entry.

Do not change a live final consumer from these CPU results alone. Promotion
would need a useful measured gain, explicit numerical decision, and actual
trained prediction/runtime comparison on the original two-T4 qualification.
No alteration to either healthy trainer or its data/weights is authorized by
this experiment. A negative or tiny speed result means keep the current path.

One canonical push, no retry on uncertainty. Fresh node funding preserves
518GiB other scopes plus32MiB existing large-study coordinator, this32MiB scope,
and20GiB reserve. Shared123MiB/370file liabilities and20GiB/5000file floors are
retained; this experiment adds zero shared files.

## Actual result, October 21

Private CPU notebook **140055931 v1** completed once from source
`b770860b2f6d060548aa2aafc354c946a256b647`, candidate
`d20cfeda884eb25a2ac67199eec6870ed65ea824f4e9ab7e322a318d07aac7c6`.
The six trials took223.49seconds total and peaked at4,023,541,760RSSbytes.
The balanced warm means were27.0233seconds for one thread per lane and
21.6329seconds for two: **19.95% less preparation time** in this limited
experiment. Every non-image context field was exact. Two-thread images had
maximum difference1.1920928955078125e-7; they were **not bitwise equal**.

This is a promising contingency, not a clinical result or hidden-runtime proof.
The current frozen final consumers retain their existing exact preparation for
their first trained qualification. A separate two-thread successor is justified
if actual runtime admission fails or measured GPU capacity permits the required
trained prediction comparison. Do not stop healthy training or restart a
notebook to repeat this result. See [original receipt](actual-complete.json)
and [decision and limitations](analysis.json). No GPU quota or entry was used.
