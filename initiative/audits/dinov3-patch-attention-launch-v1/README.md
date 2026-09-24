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
the normal partition. It completed `0:0` on evc30 after 9:40. The
scheduler-visible one-GPU NVML/CUDA driver probe, Python header identity,
actual one-GPU PyTorch CUDA context, and 26 finite optimizer updates passed.
The result, run binding, predictions SHA-256, and checkpoint SHA-256 were
verified; see [the outcome](pilot-outcome.json). The diagnostic restricted
both training and outer evaluation to 100 studies. One target had no AUC in
this tiny sample, so its macro AUC is null and the available-target AUC is
not a meaningful architecture comparison or Kaggle score.

After the pilot, the source remained clean, no account jobs were running,
the distinct production output was absent, and Lustre used
1,007,977,604/1,048,576,000 kB and 989,745/1,000,000 files. The separate
[production intent](production-dispatch-intent.json) preceded one submission
of the exact staged [ten-epoch script](production.sbatch). Slurm accepted full
fold-0 job **845218** on one H100 and two CPUs; it was running on evc30 at
the first check, with the allocated CUDA and PyTorch context gates passed.
The pilot checkpoint is not reused for production. Every completed production
epoch has an immutable scheduled snapshot and must be privately published and
download/hash verified once. No Kaggle submission was created.

The production job then bound the full 2,247-study fold-0 training set,
one-GPU world size, `patch_attention`, and a ten-epoch/every-epoch snapshot
schedule. Its first five real optimizer updates had finite losses and
gradients with zero stderr bytes; see the [progress receipt](production-first-progress.json)
and exact [run binding](production-run.json). This confirms training has
started, but supplies no full-fold result yet. The separate
[publisher](../../../scripts/hf/publish_patch_attention_epoch.py) is staged
on Newton with exact SHA-256 and a private path distinct from the global-CLS
baseline. It has not yet uploaded any epoch. A local coordinator must stream
the Hugging Face credential through SSH into a one-CPU Slurm allocation only
after verifying a completed immutable snapshot. The publisher checks 281
updates per epoch, 2,247 training studies, the source code and official license,
and re-downloads every private object for size/SHA-256 verification. Reconcile
an uncertain publication read-only before any retry.

The local [single-epoch coordinator](../../../scripts/hf/dispatch_patch_attention_epoch.py)
is prepared to read the ignored `.env`, check a completed snapshot and the
private destination, write a durable dispatch intent, then stream the token
through SSH to a one-CPU, zero-GPU Slurm upload. Its first read-only call for
epoch 1 correctly refused an incomplete snapshot before writing an intent or
making an upload; see [the acceptance receipt](publisher-dispatch-acceptance.json).
Once an epoch completes, run it once with that epoch number and an active
`--ssh-control` socket. On any uncertain result, inspect its local and Newton
journals and the private HF path read-only; do not repeat the mutation.

GitHub main `500ef60974e117db5d7aa84dae95479070f18045` is also mirrored
to the private [Hugging Face milestone](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/da04bef5f6d886692e2829cb18123c9d586f0f18/experiments/dinov3-v1/milestones/patch-attention-launch-500ef60).
All 16 source, configuration, audit, publisher and manifest files were fully
downloaded and matched by byte size and SHA-256. The pilot result and
production dispatch were separately mirrored to the private
[pilot milestone](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/e180a9d16caf4dcf2f09d0cdd8f1dbfb182ec1dd/experiments/dinov3-v1/milestones/patch-attention-pilot-2236bd7)
at revision `e180a9d16caf4dcf2f09d0cdd8f1dbfb182ec1dd`. All 12 files
passed full download/size/SHA-256 checks. The first production updates are
newer than that mirror and recorded here on GitHub.
