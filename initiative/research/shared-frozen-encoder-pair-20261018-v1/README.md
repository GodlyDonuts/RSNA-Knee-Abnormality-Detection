# Share frozen MRI encoding across matched path heads

Both queued path arms use the same frozen parent and exact study order. The new
execution wrapper shares each detached encoded effective batch in RAM, then
clears it after both original head updates. It does not average gradients,
change targets or train the encoder. Each arm retains its own unchanged trainer,
AdamW, learning-rate schedule, cursor, history and complete RNG checkpoint.

Nine focused synthetic CPU cases pass. Both arms' decoded final and every epoch
model/AdamW/LR/counter/order/all-RNG/history states and prediction CSVs equal
independent original execution. A fresh-process stop/resume matches, and a crash
between arm updates resumes only the lagging arm. The new example fetches32rows
instead of64 across the paired trial. That is half the provider work, not a
measured MRI-encoder/GPU wall-time improvement or an accuracy gain.

Cache admission requires detached tensors, no global provider RNG consumption,
no tensor mutation, at most the original eight-study effective batch and an
explicit4GiB tensor-retention budget. No persistent feature cache or old data
copy is created. A failure does not silently drop images, change batch size or
switch arithmetic. Each arm checkpoints at every update; this IO overhead must
be measured before replacing independent production. Cached NumPy metadata are
not mutation-instrumented; the unchanged reader treats them read-only.

A distinct actual CPU run is prepared on the existing original sixteen fitting
cached-MRI engineering cases, full1024/64head and four updates per arm. It uses
twoCPUs/zeroGPUs/32GiB/onehour, no encoder, no outer label reads and no clinical
metric. New private source/checkpoints have a separate16MiB node-local cap,
retaining the full existing3GiB original fixture allowance, existing large
reserve and all originals. New source hashes, original provider/proof bindings
and current free space gate dispatch. This is a new wrapper execution test,
not a repeat of old independent recovery or a production/competition model.
Original955582/955711/955712 and Kaggle owners remain unchanged. Only actual
GPU online-input acceptance could justify replacing pending independent work.

## Actual new real-MRI execution

CPU956378 stopped during source compilation with system Python3.6, before
creating the output root or submitting any fit. Distinct CPU956387 used the
existing original Python3.12 runtime, confirmed the root was absent and original
32-case copy proof present, then staged the same two files and submitted956388
ONCE. At dispatch753479561216bytes were free, above the explicit retained
612022943803-byte floor plus entire old3GiB fixture and new16MiB output caps.
No old data re-copy or bulk rehash ran. Scientific source881aa020 is immutable.

Actual956388 is RUNNING on evc26 with zero GPUs/two CPUs. CPU956391 observed
one completed optimizer update in EACH arm, eight original provider reads and
eight shared reuses,399377976peak retained tensor bytes and empty stderr. The
frozen encoder is never instantiated by this cached-feature CPU trial. Neither
terminal success nor online/GPU acceleration nor disease accuracy is inferred
from this progress. Inspect its original result/checkpoints before any next use.
No original955582/955711/955712 or active Kaggle owner was replaced.
