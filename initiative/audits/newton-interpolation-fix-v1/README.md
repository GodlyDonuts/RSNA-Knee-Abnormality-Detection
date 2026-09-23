# Exact update-23 image-range repair and bounded restart

Two full-data H100 jobs on different nodes made the same 22 optimizer updates
and stalled before update 23; the [repeatability audit](../newton-repeatable-stall-v1/)
preserves their records. CPU job 840800 found 20 pixels in the rank-1 third
microbatch at `1.000000238418579`. The uint16 cache files were byte-verified;
the excess is FP32 bilinear-antialias interpolation roundoff above the strict
normalized-image maximum of one. This identifies a real input-contract defect
at the failure boundary, though GPU causality still needs a bounded replay.

Git `b74060b98f01bff1b924070ccba5ce44dd002f40` clamps only the resized
training input to `[0,1]`; it does not change source images, accepted labels,
masks, folds, or pretrained weights. A regression test covers white uint16
pixels resized from 256 to 224. Local verification passed 967 tests, Ruff,
mypy, archive integrity and the current review-coverage gate; GitHub CI run
35806202039 passed. The private Hugging Face code overlay was byte-verified
at revision `d7c96df979d549554186a0cf1976b4bf34ac2ec9`.

One-CPU Newton job **840938** completed `0:0` using the committed source and
the exact eight studies extracted from the verified full-cache archive. All
eight images are finite and inside `[0,1]`. The other seven image hashes are
unchanged; only the one study with the 20 rounded pixels changed, and its new
maximum is exactly one. `cpu-fixed/` preserves the script, dispatch and full
result with original SHA-256 proof.

The Git bundle for the committed fix was staged and verified, then the Newton
checkout was verified clean at that exact commit; `code-stage/` contains the
receipts. A matching 45-file Kaggle coordinator projection is staged on private
Newton storage at `/home/sa305415/rsna-knee/kaggle/projections/b74060b-b74060b`.
Its manifest SHA-256 is
`94d9d84d8a71addeae015fd0c833cb71c82268929c2616beb21f99837447923b`
and TAR SHA-256 is
`b9c3e6b4675bf3163e9169b07116d6735cdb85bcac736d963482f93ec6b217e5`.
`kaggle-projection/` preserves the small manifest and byte proof. No
checkpoint bundle, notebook, ledger entry or competition submission exists yet.

The single bounded two-H100/four-CPU job **840991** completed `0:0` on evc28
in 9:53 wall time. It crossed the old stall: update 23 completed at 65.53
trainer seconds, and update 26 completed at 78.75 seconds. Its intentional
optimizer-boundary stop wrote a 1,215,174,757-byte checkpoint with SHA-256
`b50ae5591b7b5b02f36423676f1f7fcbedacd69c8d848172bee459e4ae8d044a`.
The wrapper verified all 26 events and the checkpoint. `pilot/` preserves the
original dispatch; `pilot-success/` preserves immutable logs (stderr as a
lossless deterministic gzip), run/invocation records, checkpoint proof, and
byte receipts. The intentional trainer exit 75 appears in stderr, while the
wrapper and Slurm job completed successfully.

The first one-CPU proof job **841089** failed `1:0` before examining the
checkpoint: executing the proof script by absolute path omitted the pinned
source checkout from Python's import path. It created no proof files. The
original staged script remains in `resume-prep/`; the separately archived
`prepare_resume.repaired.sbatch` adds `PYTHONPATH` only to that proof command.
Its staged bytes were verified before the one-CPU retry **841091**, which
completed `0:0`. It decoded checkpoint `(epoch=0, next_update=26, step=26)`,
checked the exact source, release, configuration and binding, and hard-linked
the checkpoint plus the original run/invocation records to `pilot-checkpoints/`.
The proof SHA-256 is
`9b40f9f9f5c4a3b82ca2cb05ae9dc0861066d557d754c64da5f38175e7457aba`.

The first four-hour two-H100/four-CPU continuation **841094** was submitted
after that proof and allocated evc27. Its allocated CUDA visibility probe
timed out before training; Slurm reported `FAILED 2:0` after 1:54, and the
original checkpoint and proof hashes remained unchanged. The single retry
**841107** excludes evc27 and the previously excluded nodes; it allocated
evc28, the successful pilot node. It resumes the same ten-epoch fold-0 run
from update 26; no epoch or Kaggle snapshot is claimed yet. `resume-launch/`
preserves both submit intents/results, original failure output and Slurm state
snapshots. The earlier `files.json` binds the original audit artifacts; each
new subdirectory has its own byte receipt. The accepted masks, folds, image cache, official
weights and pinned `rsna/**/*.py` training source remain unchanged.
