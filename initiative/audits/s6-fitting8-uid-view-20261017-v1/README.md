# Fixed fitting-only S6 MRI input bridge

Observed October 17, 2026, 05:55 UTC. This prepares the original anatomical and
physical S6 inputs on eight fixed fitting studies for later cross-lineage
comparison. It is CPU-only input engineering, with no disease classifier,
fitting, labels, clinical evaluation or competition entry.

The eight cases are rounded ordinal quantiles of series count within the
existing3887 fitting roster, including its cost extremes. Selection uses no
labels or model predictions. Original source functions, anatomy and geometry
remain unchanged: full field plus100mm knee crop,320 preparation/288 execution,
three slices at3.5mm spacing. [Exact protocol](protocol.json).

## Preserved first failure

Original **139162434v1** errored after471seconds before any accepted item. Its
first item assertion hid the caught reader error, so that exact error string
cannot be recovered from the original wrapper. This was CPU-only; incidental
CUDA-library warnings are not evidence of GPU initialization failure.

Inspection of the actual pinned `probe_infer.OneStudySource` and
`native_cache.prepare` proves that inference on split train opens `train.csv`
for study IDs and writes private `patients.csv` metadata. Our original guard
refused both, a deterministic input-contract violation. We preserve this
explanation as source-derived rather than pretending the lost caught exception
was observed. No original job or slug was retried.
[Original failure](original-input-owner-failure.json).

## Corrected distinct input view, not a geometry change

**139165316v1**, `godlydonuts/rsna-s6-fitting8-uid-cpu-47ae6254eedd`, was pushed
once from sourceSHA47ae6254. It actually **COMPLETED**, and CPU942782 independently collected once and
verified all1,154,209,297compressed bytes, decoded tensors and the fitting roster.
All5372windows pass:248/420/586/628/480/624/780/1606, from studies with3–14series.
Input preparation took516.75s/3.44GBpeakRSS. Original owner20144 is terminal
success, not an unfinished collection. [Actual proof](actual-inputs-private-verified.json).
Its private view contains only eight IDs in a generated `train.csv`, read-only
symlinks to the original `train_series` and complete series metadata, and the
unchanged original train reader. Original labelled train/Gold/report files stay
refused; only source-generated private patient metadata is permitted.

Nine local contract cases establish identical reader path lists/DICOM bytes,
identical metadata, the UID-only roster, access refusals, prewrite caps and
compilation. They do not establish actual MRI or accuracy acceptance.
[Checks](UID-view-local-contract.json), [builder](immutable-input-builder.py.txt).

The new fixtures are capped at1.5GiB within the already admitted2GiB S6 fixture
pool; existing public fixtures consume348312520bytes. Metadata fits the existing
16MiB controls. On actual completion, its sole owner collects once directly to
private node33 storage, independently verifies compressed and decoded bytes and
the selection, and records only sanitized aggregates. No pixels, case IDs,
patient metadata or teacher predictions enter Git/HF/Mac. A later failure retains
the exact caught error privately rather than losing it to a bare assertion.

A distinct CPU-only classifier dispatch now reuses these exact inputs and the
already accepted original model/runtime. Worker changes from the accepted
public-fixture CPU reference are only input/output roots, retaining the same four
device/report AST replacements and original inference. TwoCPU/zeroGPU/16GiB/
45minutes, private2MiB new output cap inside existing16MiB controls, no new pixel/
weight copies or labels. It is **dispatched, not yet terminal or accurate**;
`artifacts/s6-fitting8-classifier-20261017-v1/run_once.py`, exec77108.
[Exact admission](classifier/dispatch-intent.json), [worker identity](classifier/checked.json).
No raw case logits/IDs or pixels leave private node33.
