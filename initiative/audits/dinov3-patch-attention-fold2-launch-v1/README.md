# Matched patch-attention fold-2 launch

The global-CLS DINOv3-L baseline finished three fixed ten-epoch folds. Patch
attention finished fold 0 at 0.7451169387 internal macro AUC and fold 1 at
0.7214765382. The matched CLS scores were 0.7783102649 and 0.7128332913,
respectively. Fold 1 improved the predeclared lateral-meniscus target, but the
mixed macro and rare-target results do not yet establish an architecture win.
This matched fold-2 run completes the planned three-fold comparison. None of
these values is a Kaggle score.

The accepted masks, outer splits, official pinned DINOv3-L weights, cache,
seed, optimizer, effective eight-study batch, and ten-epoch stopping schedule
remain unchanged. The [production config](production-config.json) differs from
the fold-1 patch-attention config only in `fold: 2`. Training source is clean
Git `203f419ffac058496cdc3fbce0976bcedbb63286`. Fold 2 has 2,281 training
studies and expects 286 optimizer updates per epoch on one H100 and two CPUs.
The distinct output is
`/home/sa305415/rsna-knee/runs/dinov3-l16-patchattention-fold2-203f419-production-v1`.

Before dispatch, the ten baseline fold-2 private checkpoint copies were
verified. The [retention receipt](baseline-fold2-retention-result.json) records
removal of only redundant local epoch 1–9 `.pt` copies, reclaiming
10,936,454,769 bytes. The local final checkpoint, sidecars, result, and all ten
private Hugging Face copies remain. Newton then had no concurrent GPU job for
this account, the output directory was absent, and the staged script and
config matched their local SHA-256 hashes. `bash -n` passed. The
[intent](dispatch-intent.json) preceded exactly one `sbatch --parsable` call;
[Slurm accepted job 848958](dispatch-result.json) at 2026-09-25 09:00 UTC.
The job later allocated `evc47` and passed scheduler-visible H100, NVML and
CUDA-context preflight. Verify optimizer progress and terminal outcome in the
live logs before calling the run successful; never duplicate the job.

Epoch 1 completed all 286 finite optimizer updates and the run advanced into
epoch 2. The immutable 1,217,168,665-byte checkpoint has SHA-256
`3fdf0589c7e8cb3fa3cced46055f6ef46cddf5f24f4c2d6b27c64ff2f7b69521`.
The single CPU-only publisher privately committed it at Hugging Face revision
`30daa7e41aadb3811fb896852b1eebbac74c0c44`. All four objects were fully
re-downloaded and size/SHA-256 verified; independent read-only path metadata
also matches. Exact [epoch-1 receipts](epoch-001/) include the snapshot,
manifest, upload intent, commit and verification result. Do not dispatch epoch
1 again. Continue preserving each newly completed epoch exactly once.

Epoch 2 then completed 572 total optimizer updates and the run advanced into
epoch 3. Its immutable checkpoint SHA-256 is
`0d89e49099f9be32bd9036ee8f1720b1f9a81d413140d80aae271927904a38f5`.
The single CPU-only publisher privately committed it at Hugging Face revision
`927e51907dd1777bfd56a861edbdb926546e099a`; all four objects were fully
re-downloaded and size/SHA-256 verified. The [epoch-2 receipts](epoch-002/)
and an independent read-only path metadata check confirm the exact snapshot,
license and manifest. Do not dispatch epoch 2 again.

Epoch 3 completed 858 cumulative optimizer updates while job 848958 remained
running. Its immutable checkpoint SHA-256 is
`6ae7b8e9266f78343e986d77509c8d91f42d8ed8f165bdccbb31473d67bfdfa1`.
The dedicated one-CPU publisher committed it once to the private dataset at
revision `f606163f049f148461e36b40e69265892230f324`. All four objects were
fully re-downloaded and size/SHA-256 verified, with an empty publisher stderr
and zero remote exit. The [epoch-3 receipts](epoch-003/) preserve the exact
snapshot, manifest, dispatch, commit and verification. Do not repeat it.

Epoch 4 completed 1,144 cumulative optimizer updates, and the same job
continued into epoch 5. Its immutable checkpoint SHA-256 is
`6c7c464069b18d8756d32009ca3dd047351499fd69125aa84a8890d7501fd436`.
The one-CPU publisher privately committed it exactly once at revision
`977d0379ff5d7dbb526dc48ca4e4e06b6009f518`; all four objects were fully
re-downloaded and hash/size verified. Independent private path metadata also
matches, including the 1,217,168,665-byte checkpoint LFS SHA-256. The
[epoch-4 receipts](epoch-004/) bind the scheduled snapshot, manifest, intent,
commit and verification. Do not repeat this upload.

Epoch 5 completed 1,430 cumulative optimizer updates, and the same job
continued into epoch 6 with finite loss and gradients and empty stderr. The
immutable 1,217,168,665-byte checkpoint SHA-256 is
`e30e0b7a21a5e905043f35bdee8fbdd15687f254ca57cd8ed9f36c4b178db1b7`.
The one-CPU publisher privately committed it exactly once at revision
`7a667e9fdaea9e2ffe0b41b502d5d5a8c2aeafc5`; all four objects were fully
re-downloaded and hash/size verified. A separate private path-metadata check
matched the LFS checkpoint SHA and size; the manifest records the pinned
training source, official weights and license. The
[epoch-5 receipts](epoch-005/) bind the scheduled snapshot, manifest,
intent, commit and verification. Do not repeat this upload.

Epoch 6 completed 1,716 cumulative optimizer updates, and job 848958
continued into epoch 7 with finite losses and empty stderr. Its immutable
1,217,168,665-byte checkpoint SHA-256 is
`802041c7ea7ba1d0369ba6e370d12c76fdcf6ff024e478429a46f2b24b99cd17`.
The single CPU-only publisher committed it exactly once to the private dataset
at revision `bcaca20cf58ece9c635b8a36bf0b99035cff2aba`; all four objects
were fully re-downloaded and hash/size verified. Independent read-only private
path metadata confirms all four paths and the checkpoint LFS SHA/size.
The [epoch-6 receipts](epoch-006/) locally match the remote snapshot, upload
intent, manifest, commit and full-verification records byte for byte. Do not
repeat this upload.

Epoch 7 completed 2,002 cumulative optimizer updates, and job 848958
continued into epoch 8 with finite losses and empty stderr. Its immutable
1,217,168,665-byte checkpoint SHA-256 is
`ab87a692c1b65de4c417a661f017a49ab1217b6183505a8fcc745f470689383b`.
The single CPU-only publisher privately committed it exactly once at revision
`9990df21fa6288325cd4a1ba167e737d9db0be80`; all four objects were fully
re-downloaded and hash/size verified. Independent private path metadata
matched all four paths and the checkpoint LFS SHA/size. The
[epoch-7 receipts](epoch-007/) locally match the remote snapshot, manifest,
intent, commit and verification records byte for byte. Do not repeat it.

The [publisher stage receipt](publisher-stage.json) binds the pinned
[private epoch publisher](../../../scripts/hf/publish_patch_attention_epoch.py)
and [fold-2 coordinator](../../../scripts/hf/dispatch_patch_attention_fold2_epoch.py).
The publisher recognizes the fold-2 286-step schedule, 2,281-study binding and
dedicated private path
`experiments/dinov3-v1/training-epochs/patchattention-fold2-203f419-seed20260922`.
Its source SHA-256 `0c8aa81779c8bebc0927fb6c3deac099d99d4096c0020e6142d2713453b8318d`
was matched on Newton. For each completed immutable epoch, use the coordinator
with the live SSH control socket. It streams the ignored local Hugging Face
credential through stdin into a one-CPU, zero-GPU allocation and verifies all
private objects by re-download, byte size and SHA-256. An existing or
uncertain upload intent must be reconciled read-only; do not retry it.

Kaggle code submission 56546143 is a separate global-CLS epoch-8 deployment
candidate. Its hidden rerun had no score or status at the last check. Hold the
remaining account slots until its actual result is known.
