# Required verified-release adapter

Authored by `/root/quality_temporal_review`, staging only. No real memberships, approvals, clinical judgments, QA closures, gold-value selection, or repository edits are produced here.

Public functions:

- `prepare_supervision(reports, adjudications, artifact_root) -> dict`: verifies the fresh actual adjudication inspection, original review provenance, bound code/source/technical policy, concrete semantic registry and all three resolved QA bindings. Return keys: schema_version, semantic_supervision (the unchanged core audit), release_context, gold_uids, technical_eligible_uids, prepared_sha256. Gold IDs authenticate evaluation membership only. Eligibility derives from the bound sampling CSV and exact source/cache memberships.
- `write_verified_release(output, studies, audit, artifact_root) -> dict`: `output` is the labels output directory and must already contain studies.parquet. Freshly repeats preparation, reconstructs the complete table from source and verified correction cells, checks exact labels/quotes/masks/provenance/groups/splits, requires both observed classes in each target in train and weak_validation, and creates verified_release.json exclusively. Existing receipts are not overwritten. Returns the small checkpoint binding.
- `load_verified_release(studies_path, artifact_root) -> (full_DataFrame, checkpoint_binding)`: requires the sibling verified_release.json, replays the entire table and all source/review/technical/semantic/QA checks before returning. A caller may select splits only afterward. Neither caller-supplied UID sets nor a newly computed superficial table hash establish a valid release.
- `validate_checkpoint_binding(binding) -> None`: checks offline identity shape and digest for public-test inference without any training data. It authenticates the checkpoint's recorded identity structurally; it does not claim to replay unavailable training artifacts.

Mask application by labels after preparation:

```python
studies = apply_semantic_supervision(
    studies, audit['semantic_supervision'],
    gold_uids=set(audit['gold_uids']),
    technical_eligible_uids=set(audit['technical_eligible_uids']),
)
```

The checkpoint binding has exactly schema_version=1, manifest_sha256, series_sha256, source_revision, release_receipt_sha256, semantic_registry_sha256, binding_sha256. The final field hashes all other fields with the core canonical no-newline JSON digest. Train must compare actual sampling CSV/cache revision, store this binding, and require exact equality on resume. Labeled inference validates the complete release first and then compares its exact binding. Public inference validates the checkpoint binding and separately hashes its test inputs. Diagnostic subsetting must happen after full release validation and preserve the full checkpoint release identity.

Mandatory coordinator: initiative/reviews-v2/semantic-supervision/release-inputs.json with exactly schema_version, required_inputs (the frozen core input map), code_sha256. Canonical paths are checked independently: initiative/source/train.csv, initiative/reviews-v2/provenance.json, original reviewed/draft directories, protocols/report_weak_v2.md, and the existing fixed correspondence/similarity registry paths. Code map is exactly rsna/{labels,schema,adjudications,correspondence,report_similarity,semantic_supervision,release,audit}.py. Train/infer/model architecture are deliberately outside this clinical data context.

The required split_policy object has exactly schema_version, algorithm='report_pixel_similarity_groups_gold_quarantine_hashmod10_v1', and code_sha256 equal to the coordinator map. The approved semantic context binds that policy, so merely repinning coordinator code hashes cannot silently alter label semantics.

The required technical_policy has exactly schema_version, source_revision, corrected_inventory:FileRef, decoded_inventory_sha256, dicom_audit:FileRef, series_cache_index:FileRef, series_sampling:FileRef, source_series:FileRef, reason. All paths resolve inside artifact_root. source_series is the fixed initiative/source/train_series.csv. The corrected inventory may be .parquet or lossless .parquet.gz; compressed bytes and decoded Parquet SHA are both checked. DICOM audit validation uses the existing complete-audit validator and exact corrected inventory. The cache index must be the bound series_cache_index.parquet beside the audit, with exact inventory series membership. Sampling identities must be authenticated original train-series/cache identities. Policy hashes preserve the already reviewed metadata/preferences; the adapter does not invent acquisition interpretations.

Receipt coverage has 24 target/split rows: positive, negative, masked, positive_groups and negative_groups. Gold values are excluded from these counts and cannot choose holds. A zero observed class fails closed; changing the task/target scope requires a separate decision, never an implicit runtime bypass. Fixed membership and original evidence remain preserved. All code/source/technical bindings are rehashed before writing/returning to catch changes during validation.

The synthetic tests execute the actual review-provenance, adjudication, DICOM inventory, correspondence, similarity, core hold, source/gold and adapter boundaries. They do not mock a successful policy or add a bypass flag. Their six synthetic reports/72 cells and synthetic approval records are isolated temporary fixtures; they make no statement about production readiness.
