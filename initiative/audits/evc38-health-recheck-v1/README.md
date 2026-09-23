# evc38 dual-H100 recheck, September 23

The owner reported evc30 and evc38 as apparently free while fold-1 retry
**843935** was pending. Fresh Slurm node state showed evc30 as
`IDLE+MAINTENANCE+RESERVED`, so it was not schedulable. evc38 was idle, but
an earlier two-H100 pilot on that node had seen only one assigned GPU and
failed CUDA initialization. The pending training job was left unchanged.

Bounded diagnostic **843973** requested evc38's two H100s, two CPUs, 8 GiB
and at most fifteen minutes. It used only an allocated-device/NVML/CUDA
driver check, followed by a pinned-environment two-rank CUDA/NCCL test only
if the driver check passed. The exact script, pre-dispatch node state, one-shot
intent, scheduler response, original logs and terminal accounting are saved
here. It used three seconds of allocation and finished `FAILED 2:0` at the
driver gate: `CUDA_VISIBLE_DEVICES=0,1`, but `nvidia-smi` reported **one** GPU
and `cuInit` returned **999**. No model, MRI data, cache, optimizer or
checkpoint was touched. This reproduces the earlier evc38 device-visibility
failure and confirms it is unsuitable for the current two-GPU run.

The existing fold-1 training retry **843935** and CPU-only watcher **843936**
remain the only active work; evc38 stays excluded. Do not route training to
evc38 based only on idle scheduler status. A future change would require
allocated-device recovery evidence or a cluster repair.

The owner also pointed out that evc32 was using only one GPU. A fresh Slurm
`scontrol show node evc32` confirmed `CfgTRES=gres/gpu=2` and
`AllocTRES=gres/gpu=1`, with `State=MIXED+PLANNED`. Its remaining single GPU
cannot host the unchanged two-GPU fold-1 job on one node. The four allocated
CPUs are not the limiting resource.
