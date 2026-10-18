# Earlier queue placement attempt

Fresh Slurm test-only scheduling predicted evc26 ahead of owning evc33. This
justified one distinct bounded one-minute assigned-context placement request,
not another MRI/model/recovery run. Actual CPU956313 stopped before invoking
sbatch because system Python3.6 does not accept subprocess text=. CPU956314
reconciled the original intent, no dispatch result and no live matching job;
the corrected universal_newlines invocation submitted956315 once. Node-local
controls are bounded at256KiB inside existing control capacity, with753479643136
free bytes observed above the retained603433009211-byte floor. No assets moved.

Actual GPU956315 allocated evc26 and FAILED1:0 after3seconds. Torch emitted
cudaGetDeviceCount Error101 invalid device ordinal; its CUDA availability
assertion failed. No MRI, checkpoint, model or optimizer was read. This proves
this assigned context/runtime could not execute the placement probe, not a
model failure or a physical-GPU diagnosis. No retry, device-isolation override
or data relocation follows. Original955582/955711/955712 remain unchanged.
Owning-node recovery is still queued, and healthy Kaggle findingE3continues.

The frozen source and failed terminal are preserved. Test-only Slurm job numbers
956293/956294 were predictions, not training submissions or allocated GPU jobs.
