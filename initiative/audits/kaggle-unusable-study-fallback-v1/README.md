# Bounded fallback for an unusable hidden MRI study

Submission 56528793 completed its public three-study run but Kaggle reported an
unhandled error and zero output bytes for the hidden rerun. Kaggle did not expose
the hidden traceback. The cause is unknown; this change is a candidate defense,
not a diagnosis or a successful competition submission.

The existing resilient preparation already skips a selected series when it has
a recognized DICOM/pixel/geometry error and tries another series. It still
aborted the whole submission if a study had no decodable series, or if the test
roster contained a study absent from the series metadata. The new opt-in
`--fallback-on-unusable-study` path keeps that study's exact roster position and
emits 0.5 for all twelve targets, recording its UID and fallback count in the
chunk receipt. Other studies in that chunk retain their MRI-derived predictions.
An all-unusable test roster still fails explicitly because no model prediction
exists. Cache-integrity, disk, environment, and unexpected model failures still
fail; they are not converted into neutral predictions. The notebook entry opts
in to this new path, so any future notebook must use a distinct, source-bound
bundle and version. No existing Kaggle Dataset, notebook, or submission was
modified or retried.

Focused tests exercise partial and whole invalid chunks, missing series
metadata, a real invalid DICOM beside a decodable one, ordering, and cleanup.
The public three-study examples cannot establish that a hidden failing study
was unreadable. Before another competition submission, package and verify a
distinct private Dataset and offline notebook, inspect the canonical Newton
ledger and live account allowance, and obtain a passing public MRI-only run.
Do not submit the earlier epoch-6 backup with the unchanged inference code.

A follow-up bound the same fallback to malformed DICOM header cases: a
`FileDataset` missing an expected attribute or pixel data is skipped in favor
of a healthy same-plane series. Real synthetic DICOM regressions cover missing
`StudyInstanceUID`, `Rows`, and pixel data. Unrelated `AttributeError`s still
propagate. This is an additional candidate defense; it does not establish the
hidden failure cause.

The source-only milestone is on GitHub main `8282a9f7` and its six small
objects were privately re-downloaded and hash verified at Hugging Face
revision `ea2c78d2fe8b1cc2241e4796b0b2a3a296f51226`; see the
[publication receipt](private-mirror-receipt.json). No checkpoint bytes or
Kaggle competition result are part of this milestone.
