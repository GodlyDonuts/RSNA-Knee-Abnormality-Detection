# Successful notebooks versus the latest DINOv3 hidden failure

Authenticated Kaggle `GetSubmission` and `ListSubmissions` agree on September
28: our distinct submission **56550973** completed with an unhandled-notebook
error, zero output bytes and no score. Earlier submissions **55969300** and
**56407271** completed with scores **0.936** and **0.942** and nonzero outputs.
The [account receipt](account-outcome.json) preserves exact fields and five live
slots. No new Dataset, notebook or competition submission was created. Newton
DNS was unavailable from this Mac, so the one canonical cluster ledger still
needs read-only reconciliation; do not invent a local replacement ledger.

The saved private version-1 notebook bytes match the original source hashes in
[the first comparison](../kaggle-successful-submission-comparison-v1/).
Notebook-to-successful-submission association uses ownership, version, title
and timing because the submission API exposes no notebook reference. Their
original third-party source stays in ignored local artifacts and is not
republished. [comparison.json](comparison.json) records the inspected contracts
and differences against our exact submitted wrapper and notebook.

| Area | Successful source behavior | Latest DINOv3 source behavior |
| --- | --- | --- |
| Bad slices | Catch per-slice decode exceptions, retain usable pixels and fill missing sampled slices from a healthy neighbor. | Decode all slices in a selected series; exclude a series only for enumerated error-message prefixes. |
| Slice geometry | Permit tied slice positions; use InstanceNumber/natural ordering when geometry is unavailable. | Cache preparation can accept a geometry warning that the model dataset rejects later. |
| Failed studies | The 0.942 Raptor branch logs failed studies and fills missing branch probabilities with cohort means or 0.5. Optional residual branches can retain healthy base predictions. | Neutral fallback only covers studies deemed unusable during preparation. A later dataset/model/subprocess exception still aborts the notebook. |
| Execution | Mostly uses Kaggle's installed runtime, both T4s, mixed precision, shared model computation and bounded prefetch. | Creates a complete isolated pinned runtime, uses FP32 on the default CUDA device, and reloads a model subprocess every 32 studies. |
| Predictions | Multiple DINOv2/CNN/CoAtNet arms with percentile-rank fusion. | One DINOv3-L scheduled checkpoint with sigmoid probabilities. This affects accuracy and cost, not CSV validity. |

Both write `submission.csv` under `/kaggle/working`, use `test.csv` study IDs,
check the 13 target columns and finite probabilities, and execute private
offline GPU inference. The 0.942 notebook also gates complete model membership
and the final output; it does not simply swallow every global failure. Its
important tolerance is at the data/study boundary. Validity does not require
percentile ranks or rounded decimal values.

## A remaining deployment gap reproduced on identical synthetic DICOMs

[reproduce_geometry_gap.py](reproduce_geometry_gap.py) checks five local source
hashes against the exact submitted projection and the preserved 0.942 notebook
hash. It executes only that notebook's manually reviewed `order_slices` and
`read_slot` functions, never its cells, models, downloads or external writes.
The fixture supplies valid grayscale pixels at slice positions `[0, 0, 2]`, a
healthy same-plane alternative and healthy coronal/axial series.

Our cache preparation accepts the first sagittal series, reports
`duplicate_slice_positions`, and reports zero unusable studies. The unchanged
`CompetitiveStudyDataset` then raises `Patient-geometry cache slices must be
strictly ordered` outside the preparation recovery handler. The model
subprocess would stop, no healthy alternate would be selected, and the final
combined CSV would not be written. On the identical files, the successful
notebook reader returns finite uint8 image data. The [replay result](geometry-gap-result.json)
binds the exact source hashes and observed behavior.

This proves a remaining deployment defect. It **does not prove** that tied
positions exist in Kaggle's hidden inputs or caused any particular hidden
failure. The successful 1,300-training-study Kaggle rehearsal did not reproduce
the hidden test set. An unseen decoder, metadata, model-input or execution
condition remains possible.

From the repository root, with the preserved private source in ignored
`artifacts/successful-submission-review-v1/` and pinned pydicom installed:

```bash
python initiative/audits/kaggle-successful-submission-comparison-v2/reproduce_geometry_gap.py
```

The replay writes only ignored diagnostic artifacts; model weights, accepted
training data, masks and folds are unchanged. The initial harness attempts
misread the upstream ordering function's `(files, success)` return contract;
the final replay uses that contract and independently verifies finite output.
Those harness errors were not Kaggle or model failures.

## Next deployment decision

The evidence supports adopting the successful notebooks' data-boundary recovery
pattern while retaining our trained preprocessing and checkpoint. Validate model
input geometry before accepting a series so alternate selection can still act;
make a study's data error unable to kill unrelated studies; retain strict global
weight/source/runtime checks; log each phase and each recovered UID. Rehearse
valid and fault-injected data through the actual complete inference path before
one distinct guarded hidden submission. Copying the upstream pixel transform
would change our trained input distribution, so only the recovery control flow
should be adapted. No new candidate or submission is authorized by a quota reset
alone, and no guarantee of hidden success follows from this source comparison.

The pinned DICOM replay passed twice (the second from the archived script).
Repository validation passed: 1,023 tests, Ruff lint/format, mypy for all 35
source files, archive verification, and complete v2 review coverage. No
production inference code was modified or deployed by this comparison.
