# Inference metadata bindings

Held-out inference (`rsna.infer --studies ...`) requires the exact training
series CSV and study manifest, plus caches from the same source revision and
transform. This closes a gap where a checkpoint bound its training metadata but
inference accepted a different effective plane table or series selection.

The checkpoint's `series_sha256` and `manifest_sha256` must be valid SHA256
digests and must match the exact input bytes. Inference parses the same byte
snapshots it hashes. Changing a plane, feature flag, selected series, study
membership, review lineage, manifest label, or even CSV serialization causes a
failure before DICOM preprocessing or prediction. Use the complete original
manifest with `--split gold_evaluation` or `--split weak_validation`; do not
export a new subset merely for evaluation.

After the dataset is opened, its homogeneous source revision must match
`checkpoint.cache_source_revision`, in addition to the existing cache version
and resolution checks. If held-out inference also builds caches with
`--dicom-root`, explicitly supply the matching `--source-revision`; a different
revision fails before preprocessing starts. Cache bytes and full-source evidence
still require the separate complete audit. These checks do not replace it.

The metrics sidecar records the actual `series_sha256`, `manifest_sha256`,
`cache_source_revision`, and
`metadata_binding: exact_checkpoint_series_and_manifest`. Current training
checkpoints already contain all required fields and need no migration.
Older checkpoints missing these bindings fail clearly in held-out mode. There
is no unchecked fallback or argument to override the checkpoint with a new
metadata checksum. Preserve original checkpoints; recover their original bound
inputs or train a properly bound checkpoint rather than inventing provenance.

## Distinct test-template mode

Inference with `--template` uses new public-example or hidden-test metadata;
those study/series rows cannot equal the training CSV. This mode retains its
existing separate test-input behavior and does not require training-table hash
equality. Exactly one of `--studies` or `--template` must be provided.

Its metrics record `series_sha256`, `template_sha256`, the actual cache revision,
and `metadata_binding: separate_test_inputs_not_checkpoint_bound`. This records
the exact bytes used but does **not** authenticate the origin or independent
review of the test export. Supply the separately source-bound effective test
table and preserve its export audit and selection lineage under the same policy
used for training. Existing cache version, resolution, study/series identity,
target order, and exact submission-template checks still apply. Missing training
manifest/series hashes are only tolerated in this distinct test-input mode;
they never establish that a legacy checkpoint is suitable for held-out scoring.

The runtime does not add global orientation heuristics, infer new metadata from
images, inspect gold to choose corrections, or modify any supplied table.

## Tests

`tests/test_inference_bindings.py` runs real model inference on a small cache,
checks recorded input hashes, and exercises modified plane/feature/selection/
lineage metadata, altered manifests, missing or invalid checkpoint bindings,
mismatched cache revisions, and early rejection before raw preprocessing. The
existing continuous-versus-resumed training test also performs successful
inference with its real training checkpoint and exact bound inputs.
