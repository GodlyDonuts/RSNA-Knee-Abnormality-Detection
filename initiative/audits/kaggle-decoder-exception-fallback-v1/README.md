# Bounded pydicom decoder-error fallback

The current Kaggle code submission **56546143** had no terminal hidden status or
score at the 2026-09-25 10:39 UTC authenticated check. This change is a
separate, unsubmitted candidate. Kaggle has not provided a hidden traceback,
so none of the decoder cases below is a proven cause of its older failures.

The pinned pydicom 3.0.2 decoder can raise `NotImplementedError` when a DICOM
declares an unsupported Transfer Syntax UID, or `RuntimeError` when every
available compressed-pixel plugin fails. The opt-in Kaggle series wrapper
previously let both exceptions abort the whole notebook. It now treats only
the exact pydicom message prefixes as an unusable MRI series and seeks a
healthy same-plane alternative. Unrelated runtime, environment, model and
software errors still propagate. The trained `rsna/**/*.py` source,
checkpoint, masks and folds are unchanged.

A real synthetic DICOM with an unsupported Transfer Syntax UID reproduces
pydicom's `NotImplementedError`; the focused regression verifies that the
healthy MRI series is selected. A separate injected plugin-failure message
verifies the `RuntimeError` branch. Negative checks keep a CUDA-memory error
and an unrelated `NotImplementedError` fatal. The complete chunked-inference
test module passed **28 tests** locally with the pinned pydicom version.
This is bounded decoder behavior evidence, not a hidden-test diagnosis.

The earlier [transfer-syntax bundle](../kaggle-transfer-syntax-fallback-v1/)
was prepared from an older exact source projection and remains immutable.
The new [projection](projection.json) combines packaging Git
`fcbac6b31ca36f28127b6be04286e4bd52f15e5c` with the exact trained
runtime from Git `b74060b98f01bff1b924070ccba5ce44dd002f40`.
Its 51 files and 35 trained runtime files were independently SHA-256/size/mode
verified in a distinct private Newton staging directory; see the
[intent](stage-intent.json), [build receipt](projection-build-receipt.json)
and [remote verification](stage-verified.json). The source directory is clean
for the single coordinator, with the manifest and transport archive beside it.
No training checkout was altered.

The exact 51 files plus manifest were preserved once in private
`GodlyDonuts/RSNA-Data` at revision
`00801f3d5e5bf84b07bff8b06ff4391d8a6fe27a`. The
[publisher](publish-projection.py) wrote a durable [intent](hf-intent.json)
before the commit. Its [result](hf-commit-result.json) and
[verification](hf-verified.json) confirm all 52 objects were fully
re-downloaded and SHA-256/size matched; the local and Newton publication
journals are byte-identical. This was private source preservation only.

This change still requires a private Kaggle Dataset and exact notebook version
before a future competition slot could be considered. Wait
for 56546143's actual terminal outcome and use the single Newton account
ledger before any Kaggle publication or submission.

The new, distinct local bundle has now been prepared without a Kaggle
publication. One-CPU, zero-GPU Slurm job **849283** completed `0:0` in 62
seconds. Its pinned [script](prepare.sbatch), durable [dispatch
intent](prepare-dispatch-intent.json), [acceptance](prepare-dispatch-result.json)
and [terminal verification](prepare-terminal-result.json) record the exact
operation. The 42 Dataset payload files, including the 1,215,161,641-byte
fold-2 epoch-8 checkpoint, were independently SHA-256 and size verified on
Newton. The small [operation](prepared-operation.json), [manifest](prepared-bundle-manifest.json),
[snapshot](prepared-snapshot.json), [notebook metadata](prepared-kernel-metadata.json)
and [notebook source](prepared-kernel-run.py.txt) were copied back and checked
against their exact hashes. The bundle is at
`/home/sa305415/rsna-knee/kaggle/bundles/fold2-epoch008-eb280864bd1c-decodererrors-v1`.
No Kaggle Dataset, notebook or competition submission was created; this is a
contingency only while entry 56546143 remains unresolved. The Lustre soft
quota after preparation was 1,018,689,196/1,048,576,000 kbytes and
990,708/1,000,000 files. Preserve room for the ongoing fold-2 checkpoints.
