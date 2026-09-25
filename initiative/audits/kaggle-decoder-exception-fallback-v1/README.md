# Bounded pydicom decoder-error fallback

The current Kaggle code submission **56546143** had no terminal hidden status or
score at the 2026-09-25 10:39 UTC authenticated check. This change is a
separate, unsubmitted candidate. Kaggle has not provided a hidden traceback,
so none of the decoder cases below is a proven cause of its older failures.

The pinned pydicom 3.0.2 decoder can raise `NotImplementedError` when a DICOM
declares an unsupported Transfer Syntax UID, or `RuntimeError` when every
available compressed-pixel plugin fails. The opt-in Kaggle series wrapper
previously let both exceptions abort the whole notebook. It now treats only
the exact pydicom message prefixes as an unusable MRI series and seeks a
healthy same-plane alternative. Unrelated runtime, environment, model and
software errors still propagate. The trained `rsna/**/*.py` source,
checkpoint, masks and folds are unchanged.

A real synthetic DICOM with an unsupported Transfer Syntax UID reproduces
pydicom's `NotImplementedError`; the focused regression verifies that the
healthy MRI series is selected. A separate injected plugin-failure message
verifies the `RuntimeError` branch. Negative checks keep a CUDA-memory error
and an unrelated `NotImplementedError` fatal. The complete chunked-inference
test module passed **28 tests** locally with the pinned pydicom version.
This is bounded decoder behavior evidence, not a hidden-test diagnosis.

The earlier [transfer-syntax bundle](../kaggle-transfer-syntax-fallback-v1/)
was prepared from an older exact source projection and remains immutable.
This change requires a **new** source projection, bundle, private Kaggle Dataset
and exact notebook version before it could be considered for a future
competition slot. Wait for 56546143's actual terminal outcome and use the
single Newton account ledger before any Kaggle publication or submission.
