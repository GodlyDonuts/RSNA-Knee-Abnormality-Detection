# Measure the cost of a complementary MRI teacher on retained H100 inputs

The recovered32-study S6 comparison gives descriptive ordering diversity, not
clinical superiority. Before full teacher generation, measure actual original
S6 inference on one allocated H100 using the eight already retained original
lossless MRI fixtures (5,372windows). Reuse original CPU FP32 output/model/input
proofs; no second data/model acquisition or label/report read.

Run two fresh processes, original GPU main unchanged, one FP32 and one FP16.
Dataset I/O supplies the exact admitted fixture tensors instead of re-decoding
DICOMs. FP32 must match original CPU probabilities within1e-4, FP16 within0.003,
with all eight outputs, original5372window count and zero fallback. Record actual
whole-classifier time, original inference stats and allocated VRAM. A successful
scout is not a full3887 timing bound, full teacher acquisition, training or score.

Physical evc33 has two H100s; request only one H100/two CPUs/24GiB/20minutes.
The node/runtime/source/input/stat and fresh resource admission must pass before
one dispatch. Allocated CUDA context must actually work; no isolation bypass or
retry on failure. Separate process logs and case predictions remain private.
New2MiB controls use the original S6 future7MiB after recovered numerics, leaving
5MiB; original16MiB cap is unchanged. No new model/pixel/checkpoint storage.

Actual Slurm960986 COMPLETED0:0 in57seconds; independent owning-node observer
CPU961190 collected its terminal. Both original eight-study/5372-window passes
completed with no fallbacks. FP32:24.0277seconds/5,628,577,280peak GPUbytes, max
CPU probability error2.9698e-7. FP16:19.3592seconds/4,101,850,624peak GPUbytes, max
error0.0005509172; both meet their original tolerances. Original classifier
reported GPU-only8.8s/4.2s and cached-input waits9.4s for each pass. Model loading
and the original fixture reader are included in measured whole-classifier time;
raw HF download/DICOM rendering are absent. No full-cohort timing bound follows.
Original GPU predictions remain private on evc33; reuse appropriate outputs and
proofs rather than repeating this benchmark. Observer88773 is terminalsuccess.

Next independent step: a bounded HF-raw-source renderer must reproduce original
S6 fixture tensors using exact source hashes, original source/anatomy and fresh
transient-buffer funding. Source coverage for all3887 is now verified, but no
raw-source streaming renderer or full teacher acquisition has been dispatched.
Do not reserve a full teacher GPU until its source path and resource plan work.
