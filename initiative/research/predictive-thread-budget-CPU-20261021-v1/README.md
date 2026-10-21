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
retained; this experiment adds zero shared files. No upload or dispatch yet.
