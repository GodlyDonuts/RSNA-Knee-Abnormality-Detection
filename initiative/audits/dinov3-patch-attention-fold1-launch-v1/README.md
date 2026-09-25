# Matched patch-attention fold-1 launch

The fold-0 patch-attention run completed ten epochs and 2,810 optimizer updates. Its fixed-final internal macro AUC was 0.7451169387 on 1,156 holdout studies, below the matched global-CLS baseline 0.7783102649. The experiment's predeclared lateral-meniscus target improved from 0.6472 to 0.6816, while MCL and fracture regressed. One fold cannot establish that the head generalizes, so fold 1 is a confirmation experiment rather than a selected competition model. No patch-attention checkpoint has been submitted to Kaggle.

The accepted masks, outer folds, official DINOv3-L weights, cache, optimizer, seed, effective eight-study batch, and ten-epoch stopping schedule are unchanged. The [production config](production-config.json) differs from fold 0 only in `fold: 1`; source remains clean Git `203f419ffac058496cdc3fbce0976bcedbb63286`. Fold 1 has 2,278 training studies, with 285 optimizer steps per epoch expected on one H100 and two CPUs. The distinct output is `/home/sa305415/rsna-knee/runs/dinov3-l16-patchattention-fold1-203f419-production-v1`.

Before dispatch, Newton had no concurrent GPU jobs for this account and the new output directory was absent. The [intent](dispatch-intent.json) preceded one `sbatch --parsable` call. Slurm accepted job **847119** once at 2026-09-24 18:41 UTC; [the receipt](dispatch-result.json) records it. It was `PENDING (Resources)` at 18:43 UTC, with no node or GPU reserved. Do not duplicate the job. The staged [script](production.sbatch) and config matched their local SHA-256 hashes, and `bash -n` passed.

After all ten fold-0 snapshots had been privately published and fully downloaded/hash verified, a bounded retention script checked the completed job, sidecars, final checkpoint, Hugging Face manifests and receipts. Its [preflight intent](local-retention-intent.json) and [terminal result](local-retention-result.json) record deletion of only local epochs 1–9 `.pt` copies, reclaiming **10,954,517,985 bytes**. The final epoch-10 `.pt`, `checkpoint.pt`, all sidecars, run result, and all ten private HF checkpoints remain. This made room for the fold-1 run without changing training data or source. The soft Lustre byte quota at dispatch was 1,015,105,836/1,048,576,000 kB and the file quota 990,077/1,000,000.

The pinned private [epoch publisher](../../../scripts/hf/publish_patch_attention_epoch.py) now recognizes fold 1 under `experiments/dinov3-v1/training-epochs/patchattention-fold1-203f419-seed20260922` and checks its 285-step schedule and 2,278-study binding. The separate [single coordinator](../../../scripts/hf/dispatch_patch_attention_fold1_epoch.py) streams the ignored local HF credential into a one-CPU, zero-GPU Newton allocation after checking a completed immutable snapshot and absent private path. The publisher source SHA-256 is `bd20c1beff52c8f749e9734af07282c84922af584b81d6729d5581ab06ecb30f`, matched on Newton. Its first read-only call correctly refused incomplete epoch 1 before writing an upload intent. Publish each completed epoch exactly once, with full private re-download, byte-size and SHA-256 verification. Reconcile any existing or uncertain intent read-only before another action.

This is an internal training experiment, not a Kaggle score. The fixed-final fold-1 result should be compared with the same global-CLS fold only after the job completes. Rare-target movement and the full three-fold macro need separate scrutiny before any architecture decision.

**First completed epoch:** Job 847119 allocated evc32 and passed the scheduler-visible H100/NVML/CUDA, Python-header, runtime, and cache preflights. It completed 285 finite optimizer updates for epoch 1 and continued into epoch 2 without stderr. The [epoch-1 snapshot and publication receipts](epoch-001/) bind checkpoint SHA-256 `95de8a6739f160af09ef339a2440e376e58c97fff306da8cd38dee125d8ef1e9` to private HF revision `063b77e002cf9b942434e41db8a3bff4a3dbea60`. All four private objects were re-downloaded and checked by size and SHA-256. Do not repeat this publication. This is training progress, not a held-out fold result or Kaggle score.

Epoch 2 was likewise published and fully re-downloaded/hash verified at private
revision `940aec06a5140e593edaa5a8b427ad93c417081b`; its exact checkpoint
and receipts are in [epoch-002](epoch-002/). Epoch 3's remote publisher also
finished its four-object full-byte verification at private revision
`dbc8895a326e43c224634fe207afff267416328f`, but the local SSH dispatch
response later returned 255. The [read-only reconciliation](epoch-003/read-only-reconciliation.json)
checked the durable Newton verified journal, matching private revision and
LFS checkpoint SHA-256, and re-downloaded/hash-checked the three small objects.
The epoch-3 upload was **not retried**. Epoch 4's local dispatcher hung during
SSH prechecks while UCF connectivity was unstable. It was stopped after
confirming that no local epoch-4 dispatch intent or result had been written;
the remote epoch-4 journal is uninspected. Do not dispatch that epoch or
publish later ones until actual remote state, Slurm progress and quota are
inspected.
