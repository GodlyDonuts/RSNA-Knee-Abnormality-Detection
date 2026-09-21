# Exact-series selection

`python -m rsna.series_equivalence` exports a separate series table that retains
one deterministic representative of fully equivalent series within each study.
The input table, all DICOM files, all cached volumes, and every original series
UID remain preserved. This is a technical input-selection policy; it does not
change clinical annotations or infer a diagnosis, patient identity or laterality.

The complete DICOM audit and pinned inventory are required before an export.
Their counts, source revision, cache version, exact study/series membership and
audit-index checksum must agree. Train/test pixel overlap blocks export. The
full cross-study graph must also continue into `rsna.labels` for transitive
report/pixel leakage grouping and gold-neighbor quarantine. Selection never
removes a study or a cross-study edge.

## Conservative equivalence

Two series can share a representative only when they belong to the same study
and satisfy all of these conditions:

- Their complete ordered native pixel fingerprints, native volume shape and
  nonconstant flags match. Every slice remains in the comparison, including
  constant slices, and at least one slice must be nonconstant. Matching one
  slice, an unordered set of slices, or only a normalized cache is insufficient.
- Both use unambiguous patient-geometry ordering without cache warnings. All
  stored slice positions agree within 0.001 mm; the stored orientation agrees
  within 0.001 per component and pixel spacing within 0.000001 mm. Relative
  tolerance is zero. Native/cache shapes and photometric interpretation agree.
- The effective plane, fluid sensitivity and fat suppression agree. These are
  all five numerical features consumed by the current model: three plane
  indicators and two binary indicators. Descriptions and source names do not
  determine the representative.
- Their complete decoded `uint16` cache arrays are exactly equal, after each
  candidate cache file passes its audited size and SHA256 checks.

The lexical minimum SeriesInstanceUID is retained. Each proposed group must be
pairwise compatible; small geometric differences cannot accumulate through a
transitive tolerance chain. Different SOP or FrameOfReference identifiers do not
by themselves prevent equivalence. Neither identifier is used to infer anatomy.

The native hashes are the earlier full decoder's numeric fingerprints before
modality rescaling, with matrix shape included. This command verifies their SOP
and source-file hash bindings against the inventory, and freezes the complete
cache metadata files by hash. It does **not** decode the original DICOMs again.
The cache stores all ordered positions, one representative IOP, common pixel
spacing and `original_shape`; it does not retain every slice's IOP or individual
rescale parameters. These limits are recorded in every plan. Exact cache-array
comparison additionally protects the model inputs from divergent preprocessing.

The existing full-audit schema binds the source inventory and cache NPZ hashes,
but does not contain cache-JSON hashes. This selection stage establishes those
additional bindings at planning time, and detects subsequent JSON changes.
Its source-pixel evidence remains dependent on the trusted decoder outputs;
it is not an independent rederivation of native hashes from original bytes.

## Export

First produce the accepted effective plane table with `rsna.series_metadata`.
Then use that table and the completed full audit:

```bash
python -m rsna.series_equivalence \
  --series data/series-reviewed-train/series_effective.csv \
  --inventory initiative/source/inventory \
  --dicom-audit data/dicom-audit-v3/dicom_audit.json \
  --cache data/cache256-v3 \
  --source-revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d \
  --output data/series-selected-train
```

Use the actual cache/audit locations. The output directory must be new. The
chosen raw split must be completely represented in the input table; the default
is `train_series`, with `--raw-split test_series` available for a separately
audited test inventory. An original five-column series CSV is also supported.
An already selected CSV is rejected to prevent losing exclusion lineage.

The command plans the selection, then rechecks the source CSV, inventory,
complete audit, index, every selected-split cache JSON and **every cache NPZ
file's bytes/SHA256**, including retained singletons and excluded duplicates.
This reads the cached corpus once more; it downloads no DICOM files and does not
retain new pixel volumes. Run after the cache/audit jobs finish, against stable
inputs. It is not a substitute for an immutable storage snapshot under concurrent
external file mutation.

The new directory contains:

- `series_selected.csv`: retained rows in their original relative order,
  preserving all input values and plane-review lineage, plus equivalence-group
  and selection-policy digests. Existing `StudyDataset` ignores these extra
  provenance columns and reads each retained series once.
- `series_selection_plan.json`: canonical JSON with every original row, retained
  and excluded memberships, representative UIDs, reasons, candidate comparisons,
  source/cache/audit bindings, fixed policy, and unmodified cross-study training
  edges. Its SHA256 binds the full proposed selection. This is a mechanical
  validation record, not a claim of independent human approval.
- `series_selection_audit.json`: export counts and exact plan/CSV/input digests.

Archive the input effective table, accepted plane decision, full audit, selection
plan and exported table together. The selection command binds the supplied
effective CSV; it does not repeat the plane decision's anatomical adjudication.

## Training and inference integration

No live decoder, preprocessing, audit, dataset or training code is changed by
this module. Pass the **same selected training CSV** to training, resume,
weak validation and heldout inference. Training/resume binds its bytes through
the checkpoint's `series_sha256`; heldout inference enforces the same checksum,
the study manifest checksum, and the training cache source revision. Test tables
have different bytes and require separate provenance under the same policy.
See [inference bindings](inference-bindings.md) for exact requirements and limits.

This removes duplicate model slots before augmentation and attention pooling.
It does not establish that remaining studies are ready for clinical training:
all report reviews, semantic QA, correspondence decisions, complete leakage
isolation and final-release training/inference checks remain required.

## Evidence and checks

The bounded 077:22 audit established a 30-slice within-study pair with distinct
source SOP/file identities but matching complete native pixels, geometry and
cached arrays. The current loader samples both, with identical features and
evaluation tensors. It is the concrete positive equivalence case. The similar
042:40 pair has different native pixels and geometry; both must remain.

Tests cover full selection/export provenance, stable representatives and plan
digests, cross-study retention and train/test blocking, native-vs-normalized
distinctions, model-feature conflicts, changed same-size pixels and metadata,
missing slices/membership, geometric tolerance chaining and preserved plane
review lineage. Bounded real-case checks do not certify the whole corpus graph.
