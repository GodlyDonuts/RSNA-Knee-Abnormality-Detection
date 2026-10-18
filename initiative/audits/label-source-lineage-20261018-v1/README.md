# Owner clarification: image labels versus the bound report release

The owner reports that GPT6Astra analyzed images to compile the dataset and
authorizes recovery of pre-Devin HF versions if subsequent changes damaged it.
Treat that description as an important provenance lead, not proof that the
current training manifest consumes an image-derived annotation export.

Actual read-only audit: four baseline release files (`studies.parquet`,
`study_labels.csv`, `report_annotations.parquet`, `report_annotations_original.parquet`)
are byte-identical to Gitf41879a3 on September22. Earliest Devin-authored commit
in currently fetched all-ref history is October12. HF original release revision
a177562d5f49458e287752b94f085dcb061f6495 and observed current revision
0cf9ab27bb3232eeee0d9a260068ee3480f7a9bf retain the same four LFS SHA256/size
identities, matching current local bytes. This uses authoritative metadata
content identities, not a claimed new full HF byte download.

No rollback is justified by these findings; none was performed. They do not
prove that every file in every HF repository/revision is unchanged. Later
explicit public-label overlays and distinct experiments are separate sources,
not silent replacements of these four immutable baseline files.

The bound September22 release, frozen `protocols/report_weak_v2.md`, source
review provenance and `docs/data-readiness-report.md` explicitly describe
report-derived weak supervision. Current evidence does not locate or bind an
additional whole-cohort Astra image-label export. This does not establish that
no image review occurred; an image-derived export may be a distinct source.
Keep image annotations, report extraction and model pseudo-labels separately
identified, preserve disagreements/missingness, and verify the exact training
recipe's source bindings before changing supervision. Do not relabel or claim
clinical accuracy based on the annotator name alone.

The supplied `Codex_Multi_Source_Supervision_Research_and_Implementation_Brief.md`
was absent at the original check, but is NOW available and read. Its verbatim
copy, working target constructor, synthetic checks and current source-admission
decision are in `initiative/experiments/competitive-v1/multisource-supervision-v1`.
The separate original image-label export remains unlocated.
No MRI/cache/label/mask/fold, healthy training or Kaggle controller changed.

Further original-source inspection found six root legacy label CSVs. The card
at the ORIGINAL source revision52f995f4568e36bfa5fbbaf69a89af41226b4f5d
explicitly says they were report-derived multilingual rule/NLI pseudo-labels,
calibrated on58gold rows, and not direct image inspection. This predates the
corrected report-review release; it is not evidence that older labels would
restore a whole-cohort Astra image review. Root file content identities and
historical card hash/line evidence are preserved separately without reading
clinical/gold label rows. A separate original image-review export is still
unlocated, the owner-provided brief is now available, preserved and implemented as a
standalone target constructor; no new multi-source model or accuracy gain exists.
