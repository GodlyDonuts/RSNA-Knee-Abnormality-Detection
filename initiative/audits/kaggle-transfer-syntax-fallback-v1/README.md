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

The [code projection stage](projection-stage.json) now preserves this
contingency at Newton without creating a Kaggle Dataset or notebook. It combines
packaging source Git `e4bc97339e4ab613a0e9ef67ca5e72d788e9ea4e` with the
unchanged exact trained `rsna/**/*.py` runtime from Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`. Local and Newton independently
matched the 51-file manifest SHA-256 and every staged file's size, SHA-256 and
mode. This is a ready code input for a distinct candidate if evidence warrants
one; no weights or submission artifact have been packaged.

The exact 51 source files plus `projection.json` were also committed once to
the private `GodlyDonuts/RSNA-Data` dataset at revision
`ac1f01513ce21685dbbb63f07ae87d459fe876b7`. The
[publisher](publish-projection.py) reconstructed the mixed-commit source in
memory, checked the staged Newton manifest and private target path, wrote an
intent before mutation, then fully re-downloaded and SHA-256/size verified all
52 objects. Its [intent](intent.json), [commit result](commit-result.json) and
[verification receipt](verified.json) match the separate Newton journal by
full-file SHA-256. No Kaggle Dataset, notebook version or competition entry was
created by this preservation step. Do not repeat the private commit.

A distinct CPU-only future inference bundle is now staged on Newton at
`/home/sa305415/rsna-knee/kaggle/bundles/fold2-epoch008-eb280864bd1c-transfersyntax-v1`.
The [one-CPU Slurm preparation](prepare.sbatch) was submitted once as job
**849132** and completed `0:0` on `evc21`. It combines the exact projection
above with the already trained fold-2 global-CLS epoch-8 checkpoint SHA-256
`eb280864bd1c2f896210ce360e1ae36d59649c854a9051523c2454a34d08c7b1`.
The [terminal receipt](prepare-terminal-result.json), [operation](prepared-operation.json),
[bundle manifest](prepared-bundle-manifest.json), [snapshot](prepared-snapshot.json),
[kernel metadata](prepared-kernel-metadata.json), and [exact notebook source](prepared-kernel-run.py.txt)
record the unique slugs and byte hashes. The full new 1,215,161,641-byte
checkpoint and all five copied small files matched the remote SHA-256 values.

The previously retained epoch-8 checkpoint had been removed from the run
directory after its private backup. For packaging, a hash-verified hardlink
to the existing older bundle restored those exact bytes without another copy;
see the [link intent](checkpoint-link-intent.json) and [result](checkpoint-link-result.json).
After the new bundle was fully validated, the temporary link was removed under
a separate [cleanup intent](checkpoint-link-cleanup-intent.json) and
[result](checkpoint-link-cleanup-result.json). Both original and new bundles
retain their verified checkpoint copies. No Kaggle Dataset was created, no
notebook was pushed, and no competition submission was made. The prospective
Dataset lookup returned an ambiguous 403, while an authenticated profile
search proved the distinct notebook slug absent. Do not publish or submit this
candidate while submission 56546143 lacks a terminal hidden outcome.
