# Truncated DICOM pixel payload: a reproducible Kaggle inference failure path

The DINOv3 code submissions still have no Kaggle score. Kaggle's hidden rerun
reports only an unhandled notebook error and zero output, so this audit does
**not** identify its hidden exception or prove its cause.

The `fix4` series-recovery wrapper catches a small list of known image and
geometry `ValueError` messages. In the pinned pydicom 3.0.2 decoder, a DICOM
whose `PixelData` is half the byte count advertised by its pixel metadata raises
`ValueError: The number of bytes of pixel data is less than expected ...`.
That message was absent from the wrapper's allowlist. A real synthetic DICOM
regression test failed at `ds.pixel_array`, with the exception propagating past
`prepare_usable_series` before an alternate series could be tried. This is a
confirmed deployment-code defect, independent of the model or CSV formatting.

The wrapper now recognizes only that specific truncated-pixel message as an
invalid **series**, tries a healthy same-plane alternate, and retains the
original checkpoint-bound MRI cache and model inference. The regression uses
actual pydicom bytes for one truncated and three valid series; it checks the
selected series, invalid-series count and a loadable model-input cache. Disk,
cache-integrity, model and unrelated decoder errors still propagate. No
clinical labels, training masks, folds, checkpoint weights or `rsna` training
source were changed. A study with no usable MRI series will still fail.

A [competition discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163)
reports two training DICOMs with exactly this half-sized pixel payload. Their
study IDs rank 1,974 and 2,099 in the sorted 4,407-study training roster;
neither was in our first-1,300-study Kaggle rehearsal. This explains why that
diagnostic did not test the known failure mode. It does **not** establish that
the hidden test set has the same defect.

This correction is a source candidate, not a new Kaggle notebook or competition
submission. The owner's current priority is training; any later deployment
requires a distinct immutable bundle, fault-injected and full-scale Kaggle
execution checks, and one guarded hidden submission when authorized again.
