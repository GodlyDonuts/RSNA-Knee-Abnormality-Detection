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
