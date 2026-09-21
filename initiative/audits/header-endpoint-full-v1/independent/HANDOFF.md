# Independent endpoint survey audit

The completed endpoint artifact passes independent inventory, source-binding, checkpoint and summary validation: **48,772 endpoints, 24,386 series, 4,410 studies, zero failures**. The inventory scan covered all819,635 slices and reproduced the exact frozen lexical-first/last selection. All nine inventory-file hashes match. There are no duplicate inventory paths, cross-study SeriesUID collisions or cross-split StudyUID collisions.

The endpoint artifact SHA256 is `d5569c33e453bb1ccdf4600c39e2093821af7ffc1e7e9691f3a5d0ff97e258f3`. The independent validation summary SHA256 is `defe335340b60785c041a1faa428b247b1b1ee3b637f09f4e04159f1ace84226`.

There are no missing PatientID/FoR values among endpoints, no within-series side or patient-ID mismatches, no within-study patient-ID variation, and no patient-ID hash shared by multiple studies. These are technical identifier results; anonymized IDs do not establish real-world patient uniqueness.

The final endpoints exactly match all58 selected metadata records from three preserved bounded full-native studies(067:37,018:19,017:7).20 prefix hashes also match the separately retained65536-byte digests calculated during full-file validation for017:7. No new native image reads or DICOM downloads were performed for this audit.

## Review candidates

Counts overlap except the prioritized queue tiers.

| Metadata signal | Studies | Meaning |
|---|---:|---|
| Both explicit L and R side values |25| Source/anatomical correspondence review required; not proof of two knees. |
| Blank plus filled side tags |74| Missing values must not be interpreted as right. |
| Noncanonical side words or B |20| Preserve literal values; do not silently turn aliases into authoritative anatomy. |
| At least three repeated sequence descriptions |10| May reflect bilateral sets, repeated acquisitions or normal protocol variation. |
| Description/IOP disagreement |10|15 affected series; inspect technical anatomy/geometry before changing routing. |

There are129 distinct candidate studies. `candidate-review-queue.json` maps each to exact blinded source batch/index without inspecting report targets:35 mixed-side/repeated-set candidates first,10 description/IOP candidates next,84 other tag-review candidates. Prior completed bounded technical assessments remain valid evidence, not automatic dispositions for all other cases.

The strongest systematic description discrepancy is **batch045 index14**, UID `1.2.826.0.1.3680043.8.498.41698150452794914665945455397334605778`: six series have reciprocal sagittal/coronal naming-versus-IOP disagreement. No images were viewed for this case. Other description/IOP candidate locations are000:21,025:21,037:33,042:40,043:41,044:16,069:11,073:38 and077:22. The025:21 description issue was also found during the separate source/header audit.

## Scope and method

- Every final JSONL row was matched to the exact selected source path, three UIDs, expected size/storage/OID and checksummed checkpoint payload. The success/failure partition, summary counts, candidate lists and artifact hashes agree.
- Prefix history, range-budget claims, PixelData boundary and source-size feasibility were independently checked.48,770 records use65536-byte prefixes; two small files were completely received at52,802 bytes and have the matching full-file digest recorded. This validates stored claims and their bindings; raw prefix bytes are generally unavailable for independent rehashing now.
- The plane checker uses the cross product of recorded row/column direction cosines, consistent with [DICOM PS3.3 C.7.6.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html). Its0.90 dominant-normal threshold and0.002 orthonormality tolerance are conservative candidate-detection choices. Missing/strongly oblique vectors and ambiguous multi-plane descriptions are not forced into a plane.
- The final survey is **not** all-slice header coverage or a full-corpus pixel audit. Lexical endpoints are not necessarily geometric endpoints. Nested enhanced per-frame metadata, post-PixelData fields and unsampled middle slices remain outside scope.
- All patient and frame identifiers are hashed. No clinical labels, gold targets or published annotations were opened or changed. The live survey/cache jobs and code were untouched.

## Preserved files

`validation-summary.json` is the completed independent audit; `code-contract-review.json` records the earlier preparation snapshot and therefore intentionally retains its historical pending status. `inventory-validation.json` and `expected-endpoints.jsonl` bind exact source selection. `study-candidates.jsonl`, `series-candidates.jsonl`, `candidate-review-queue.json` and `cross-study-patient-id-groups.json` preserve candidate details. `full-reference-comparison.json` binds the58 full-native reference comparisons. `validate.py`, `test_validate.py`, `compare_full_references.py` and `build_review_queue.py` preserve implementation/reproduction logic.18 focused tests passed, including incorrect UID, false full-file claim, prefix corruption, incomplete headers, impossible payload size, unwhitelisted identifiers, missing-ID handling and plane/side candidate cases.

Run the validator in a fresh output directory with `--prepare --output <fresh-directory>` and then `--output <fresh-directory>` to repeat without overwriting these immutable evidence files. Root owns archival and any subsequent candidate dispositions.
