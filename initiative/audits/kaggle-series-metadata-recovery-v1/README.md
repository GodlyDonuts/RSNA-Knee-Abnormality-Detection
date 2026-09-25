# Bounded test-series metadata recovery

The DINOv3 Kaggle code submissions that reached hidden reruns failed without a
public score. Kaggle's generic notebook error does not expose a hidden-test
traceback, so their cause remains unknown. The current submission 56546143 is
still awaiting a terminal result at the September 25, 11:30 UTC check; this
change is **not** part of that immutable notebook.

The opt-in `scripts/kaggle/chunked_infer.py` series-recovery path previously
passed each `test_series.csv` row directly to the trained selector and dataset.
An unknown `Anatomical_Plane` raises during dataset construction; a missing
`Fluid_Sensitive` or `Fat_Suppression` can also reach the model as a non-finite
feature. Either can abort a hidden study's notebook run even when another
healthy MRI series exists. The wrapper now normalizes plane spelling and numeric
binary flags, drops only rows with invalid MRI metadata before series selection,
and records their count. If a study has no usable series, the already opt-in
study fallback handles it. Missing required columns still fail explicitly.
The accepted training code, data, masks, folds and model weights are unchanged.

The focused 29-test chunked-inference suite passes. A regression writes real
synthetic DICOM files: two valid series remain usable after metadata
normalization, an invalid plane and missing binary flag are excluded, and an
all-invalid second study is designated for neutral fallback. It constructs the
actual `CompetitiveStudyDataset` and verifies finite features. The accepted
24,371 training-series rows and 15 released test-series rows all remain valid
and their plane labels unchanged. These local checks establish the behavior of
this recovery path, **not** that malformed hidden metadata caused an earlier
failure or that a future Kaggle entry will score.

Do not mutate or resubmit 56546143 while its outcome is unresolved. Any later
candidate requires a distinct exact-source package and notebook, a successful
public-example run, a fresh live account allowance, and the single Newton
submission ledger. The three-row public CSV is not a competition submission.
