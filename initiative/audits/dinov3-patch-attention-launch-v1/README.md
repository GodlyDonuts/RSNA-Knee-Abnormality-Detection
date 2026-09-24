# Patch-attention challenger launch

The owner's September 24 instruction was to continue improving MRI-only
training while Kaggle deployment is diagnosed. Source Git
`203f419ffac058496cdc3fbce0976bcedbb63286` adds a target-specific
DINOv3-L patch-token attention head. The accepted release, masks, folds,
official encoder weights, effective eight-study optimizer batch, and fixed
ten-epoch schedule are unchanged.

Newton's old full release checkout supplied the accepted data and image cache.
The new source was staged as a distinct sparse shared checkout at
`/home/sa305415/rsna-knee/code-patch-attention-203f419`; its HEAD and clean
status were verified without duplicating the 12,000-file release tree. The
incremental Git bundle SHA-256 was
`bda73ec59d55f926ed42c0db44e5e97fae02cca689f1552ca7bc8cd266aa5e42`.
The separate control directory is
`/home/sa305415/rsna-knee/production-launch-203f419-patchattention-v1`.
Remote `pilot.sbatch`, `pilot-config.json`, `production.sbatch`, and
`production-config.json` matched their local SHA-256 hashes, and both scripts
passed `bash -n`. The unique pilot and production output directories were
absent before dispatch. Newton reported no jobs under this account before the
pilot and a Lustre quota of 1,006,788,836/1,048,576,000 kB and
989,727/1,000,000 files after sparse staging.

The durable [dispatch intent](dispatch-intent.json) was saved locally and on
Newton before one `sbatch --parsable` call. Slurm accepted diagnostic pilot
**845214**, requesting one H100, two CPUs, 32 GiB memory, and four hours on
the normal partition. At the first read-only check it was running on evc30;
the scheduler-visible one-GPU NVML/CUDA driver probe, Python header identity,
and actual one-GPU PyTorch CUDA-context check passed. Cache extraction and
real optimizer checks were still in progress. This 100-study, two-epoch pilot
is not a competition model or Kaggle submission.

The distinct ten-epoch [production script](production.sbatch) is only staged.
It must not be submitted until this pilot has finite optimizer updates,
terminal checkpoint/result, exact source and binding checks, and a fresh
job/quota review. Every completed production epoch has an immutable scheduled
snapshot and must be privately published and download/hash verified once.
