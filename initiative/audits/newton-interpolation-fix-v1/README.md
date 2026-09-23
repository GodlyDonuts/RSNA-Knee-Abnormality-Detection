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

The single bounded two-H100/four-CPU job **840991** has been submitted to run
26 full-data updates, cross the prior update-23 stall, and stop at an exact
checkpoint. Its frozen script, plan, stage and dispatch receipts are in
`pilot/`. Reconcile its actual Slurm state, log and checkpoint before deciding
whether to resume the ten-epoch campaign. `files.json` binds every preserved
small original in this audit.
