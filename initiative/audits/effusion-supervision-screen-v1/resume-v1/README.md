# Exact checkpoint continuation after scheduler preemption

Original control869770 and candidate869771 were PREEMPTED by Slurm after183/182
finite logged updates, before any epoch or outer prediction completed. No new
model/data error was observed. Both atomic optimizer-boundary checkpoints175
were independently decoded on one zero-GPU CPU869814: exact original arm/run/
full2247/1156/code/base/overlay binding,258 AdamW states and allrankRNG matched.
Original checkpoint bytes and journals stay private on evc32; no fresh-model reset.

The single finite coordinator then submitted distinct explicit continuations
control869815/candidate869816 ONCE in normal partition/QOS, oneH100/twoCPU/96GiB
each on evc32. Original runtime/source/cache/header/allocated-device acceptance
is reused and rechecked. Their trainer commands add only --resume of each
verified checkpoint, same original output and strict deterministic runtime.
Numerical665c0d5, both three-epoch/843-update recipes, image/fold/release/target
boundaries and all checkpoint/publication obligations remain unchanged.

CPU admission required terminal PREEMPTED evidence, no live userGPU job, exact
original launch file sizes/hashes/cache stat inventory, fresh20.005399GiB/5000
shared files after retained liabilities,20GiB node reserve and less64KiB of new
private controls within existing32MiB reservation. No shared bytes/files, new
cache/runtime copy or deletion credit. Actual GPU driver/header admission must
pass before the original numerical trainer loads/resumes. At 04:49 UTC both
continuations actually resumed at update 176 and reached update 187, with
contiguous finite losses and gradients and empty stderr. The original learned
states were restored, rather than reset. See `gpu-progress-001.json` for the
actual observation and exact resumed command bindings. This establishes
optimizer continuation; neither arm has a completed held-out result yet.

Inspect all original intents/results before action; never repeat the coordinator
or submit another continuation while either role is healthy/pending. Current
publication coordinator still uses original outputs, snapshot hashes and shared
atomic lock. Preserve each distinct completed epoch exactly once. If another
interruption occurs, first review actual terminal and source/checkpoint state.
A queued job, decreasing fitting loss or recovered checkpoint is not accuracy.
