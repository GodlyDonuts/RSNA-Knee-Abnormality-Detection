# Fold-2 single-H100 pilot queued

The owner requested using one available GPU while the two-H100 fold-1 retry
**843935** waits. A second run on fold 1 would conflict with that pending run,
and a single-GPU checkpoint cannot be resumed by its two-rank configuration
because world size is part of the checkpoint binding. This pilot instead uses
the untouched accepted fold **2**, source Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`, pinned official DINOv3-L
weights, accepted masks and MRI cache, and a separate output directory.

One H100 with batch size 1 and eight accumulation steps retains the same
eight-study effective optimizer batch as the two-H100 recipe with four steps
per rank. The frozen encoder, precision, seed and learning rates are unchanged.
A 100-training-study, two-epoch diagnostic config first checks real
single-GPU execution before the prepared ten-epoch production config can be
dispatched. This changes rank ordering and floating-point accumulation, so it
is a matched effective-batch comparison, not a claim of bit-identical training.

The pilot requests one H100, two CPUs, 32 GiB RAM and one hour. Its script
derives from the exact fold-1 recovery script; the differences are the node
request, one-device/CUDA-context checks, single process, fold-2 config and
distinct output. It retains header, environment, cache, pinned-source, stop
and checkpoint guards. `bash -n`, local config validation, effective-batch
sampler comparisons, exact script/config hashes, remote staging hashes,
pinned source HEAD and absent output passed before one dispatch. A clean pinned
checkout was verified separately; allocated preflight checks it again.
The local `acceptance.json` records matching sampler memberships, valid pilot
and production configurations, and a real fold-2/world-1 odd-epoch checkpoint
capture with idempotent replay. This does not establish GPU health.

Slurm accepted the one-shot pilot as **844019** at 19:31 UTC on September 23.
It was initially restricted to evc32, but that node's second GPU was taken
while preparing the run. The same pending job was then updated once to allow
any H100 node except evc27, evc38, evc43, evc44, evc49 and evc50, which have
prior allocated-device failures. `post-dispatch-state.txt` records **PENDING
(Priority)**; no training update or fold-2 score exists yet. Jobs 843935 and
843936 remain unchanged. The exact submitted builder/dispatcher bytes are
preserved as `*.submitted.txt`, matching the durable intent and Newton
copies; their lint-formatted equivalents and all staging/dispatch/routing
receipts are here.

The ten-epoch production config, launcher and odd-epoch preservation source
are only **prepared locally**. Do not submit production until job 844019
completes a finite optimizer pilot, its recovery and source checks pass, quota
is rechecked, and the live job state is reconciled. At most two GPU nodes may
run concurrently. Publish each immutable completed production checkpoint to
the private Hugging Face dataset with full download/hash verification. The
pilot is diagnostic and not eligible for a Kaggle submission.

This launch and the evc38 node-health audit were also committed to the
[private Hugging Face dataset](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/479b880a436f69b4b760bee13640509f13a3c5eb/experiments/dinov3-v1/milestones/fold2-1gpu-507f03e)
at revision `479b880a436f69b4b760bee13640509f13a3c5eb`. All 33 source and
manifest files were downloaded and matched byte counts and SHA-256 hashes.
The one-shot intent and full verification receipts are archived here. This
milestone has no fold-2 checkpoint or Kaggle score.
