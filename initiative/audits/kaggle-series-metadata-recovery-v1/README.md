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

The distinct [source projection](projection.json) was built from packaging Git
`90d879a74de55af2bdf6cb2186e61989ad9e6161` while preserving every trained
`rsna/**/*.py` file from Git `b74060b98f01bff1b924070ccba5ce44dd002f40`.
All 51 TAR members were independently re-read against the manifest's hashes,
sizes and modes; the metadata safeguard is present in the wrapper. The
[build receipt](projection-receipt.json) binds the 675,840-byte source TAR SHA
`e8f5ce803ccbf9837df5f311a0bac8865b8bec0d982daf0623b68ca7fe5bdf79`.
The TAR remains in ignored local artifacts and can be rebuilt with
`scripts/kaggle/project_code.py` using those full commits and
`--preserve-training-code`. This is source preparation only: no Kaggle Dataset,
notebook or competition entry was created from it.

The exact projection is now staged on private Newton storage at
`/home/sa305415/rsna-knee/kaggle/code-projections/90d879a-b740-metadata-v1`.
A durable [stage intent](stage-intent.json) preceded the one remote directory
creation and transfer. A separate read-only pass rehashed the archive,
manifest, and all 51 extracted files, checked modes and membership, and
confirmed the trained runtime came from the pinned training commit; see the
[stage verification](stage-verified.json). This created no Kaggle Dataset,
notebook, ledger operation or competition submission. Do not reuse an older
prepared bundle for this newer source; any future bundle needs a distinct
operation and exact manifest.

The same exact 51 source files plus projection manifest were privately
preserved once in `GodlyDonuts/RSNA-Data` at revision
`35bd8c8fa2add6da4d3f6f8a4ad4e3dfd485cb8f`. The pinned
[publisher](publish-projection.py) wrote a durable [intent](hf-intent.json)
before the commit; the [result](hf-commit-result.json) and
[verification](hf-verified.json) confirm all 52 objects were fully downloaded
and hash/size matched. The Newton and local journals are byte-identical, and
a separate read-only private path metadata check matched all 52 sizes. This
is private source preservation only. The current Kaggle submission 56546143
still has no terminal result, and this source has no Kaggle Dataset or notebook.
