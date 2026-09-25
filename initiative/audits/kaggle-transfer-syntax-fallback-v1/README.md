# Missing DICOM transfer-syntax contingency

The current Kaggle code submission **56546143** is an immutable, already
submitted notebook. Kaggle had not returned a hidden status or score when this
contingency was written. It is not evidence about that submission's hidden
failure or success, and no additional Dataset, notebook or competition entry
was created here.

A synthetic DICOM with a valid pixel payload but no File Meta Transfer Syntax
UID makes the pinned pydicom decoder raise `AttributeError` beginning:

> Unable to decode the pixel data as the dataset's 'file_meta' has no
> (0002,0010) 'Transfer Syntax UID' element

The existing opt-in resilient-series wrapper handled missing pixel data and
some missing dataset attributes, but this message fell through as an unexpected
software failure. In a separate future notebook projection,
[`chunked_infer.py`](../../../scripts/kaggle/chunked_infer.py) now treats only
this exact prefix as an invalid DICOM series. It then tries another eligible
series for the same plane and study. Unexpected `AttributeError` still stops
inference; checkpoint and trained `rsna/**/*.py` source remain unchanged.

[`test_kaggle_chunked_inference.py`](../../../tests/test_kaggle_chunked_inference.py)
constructs the malformed DICOM plus a healthy same-plane alternative, runs
the real decoder/cache path, and verifies that the healthy series is selected.
The focused five-test check passed, including the unexpected-attribute-error
regression. This is a bounded synthetic behavior proof, not a hidden-test
diagnosis. Do not spend another Kaggle slot on this contingency while 56546143
has no actual terminal outcome; any later submission needs a distinct exact
private bundle and saved notebook version with full provenance checks.
