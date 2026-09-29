# Bounded test-series metadata recovery

**Terminal result confirmed September 28:** Entry 56550973 failed its hidden
rerun with the same generic unhandled-notebook error, zero bytes and no score.
Both authenticated status methods agree. The successful-source comparison and
a newly reproduced geometry recovery gap are recorded in
[comparison-v2](../kaggle-successful-submission-comparison-v2/). The root cause
of the hidden rerun remains unproven. Do not repeat this notebook or submission.
Newton DNS prevented canonical-ledger reconciliation at this observation.

**Current result, September 25, 12:54 UTC:** The distinct private offline
notebook completed its three-study public MRI-only execution with all three
studies inferred on CUDA and no fallback. Its saved version, source SHA,
attached private Datasets, manifest/checkpoint bindings and two agreeing
notebook output receipts are preserved in the [public verification](public-execution-verification.json),
[inference receipt](public-inference-receipt.json), and
[metrics](public-inference-metrics.json). Kaggle returned HTTP 403 for a direct
download of the public CSV, so the CSV itself was **not** independently
download-hash verified; both saved notebook receipts report the same SHA and
the exact code validates the output schema and probability range before writing.
The single coordinator then submitted this exact code version **once** as
competition entry **56550973**, not as a local CSV upload. The canonical ledger,
`GetSubmission` and `ListSubmissions` agree on its identity. At the first
[submission check](submission-accepted.json), Kaggle had no hidden status,
error or score; three live slots remained. Do not resubmit this version or spend
another slot until its actual hidden result is known.
The [private mirror intent](hf-submission-milestone-intent.json),
[commit](hf-submission-milestone-commit-result.json) and
[download verification](hf-submission-milestone-verified.json) bind six small
public-run/submission audit files (15,851 bytes) to Hugging Face revision
`e53703f0c50c450180bf9801e92ca21f0098ed7c`. This is provenance storage,
not a second competition submission.

At the September 25, 12:14 UTC authenticated check, the previously pending
submission **56546143** became terminal: Kaggle reported `COMPLETE`, its generic
unhandled-notebook-error description, zero output bytes, and no public score.
`GetSubmission` and `ListSubmissions` agreed; the exact account-ledger entry was
reconciled to `remote_outcome=failed` without a new Kaggle mutation. Four live
submission slots remained. The [terminal receipt](hidden-outcome.json) does not
contain a hidden traceback, so it does not prove that malformed metadata caused
this failure. The public `submission.csv` had passed, and this was a code
submission pointing to `submission.csv`, not a CSV file upload.

Preparation of a distinct correction has begun without spending a Kaggle slot.
The first CPU-only transport job 849424 failed at exact-source membership
preflight: a read-only ledger reconciliation import had left two Python bytecode
files in the staged 51-file source tree. Their hashes and the exact cleanup are
recorded in [the retry intent](transport-retry-intent.json). No transport journal
or Kaggle mutation was created. The 51 bound source files remained intact; a
second CPU-only transport job 849425 completed `0:0` after cleanup. Its
[transport receipt](transport-result.json) binds the distinct 1,215,678,819-byte
archive SHA `0852edd9148dab9c3fd4744138d009ad30e943ea4c5fb7d98a3db64ac355cd13`.
A [durable create intent](create-dataset-dispatch-intent.json) preceded the
single private Dataset publication attempt. The server returned once, then a
separate CPU-only read-only reconciliation downloaded and hash verified all 43
logical files. The [verified receipt](dataset-verified.json) binds private Dataset
`godlydonuts/rsna-knee-f2e008-metaguard-eb280864bd1c` version 1, ID
**12188112**, to the exact prepared operation. Its [create intent](dataset-intent.json)
and [server response](dataset-create-response.json) are preserved. A distinct
private offline T4 notebook was pushed once only after that verification. Its
[dispatch intent](push-kernel-dispatch-intent.json) and [exact-source receipt](kernel-push-receipt.json)
bind `godlydonuts/rsna-knee-epoch8-f2-metaguard-eb280864bd1c` version 1,
private, internet off, Nvidia T4, and precisely the two intended private
Datasets plus this competition. The exact source SHA is
`9a44b8c9dc0de88cef1f0559710faef3aceb24829bcff0e807670326d5084903`.
Its public execution was `RUNNING` at the first read-only check; this is not a
competition submission or score. No competition slot was used by these Dataset
and notebook steps.
The CPU-only create and reconciliation allocations were Slurm 849426 and
849427; the notebook push used CPU-only allocation 849428. The next read-only
[Newton quota check](quota-after-publication.json) recorded 1,025,820,112 of
1,048,576,000 kB and 990,875 of 1,000,000 files against the soft limits.
Nine small failure, Dataset, notebook and training epoch-8 audit objects were
also preserved at private Hugging Face revision
`4010a5b05ed7b990d0daa024044829cec612210d`. The
[intent](hf-small-milestone-intent.json), [commit result](hf-small-milestone-commit-result.json)
and [verification](hf-small-milestone-verified.json) show all 23,080 bytes were
fully downloaded back and hash verified. This is provenance preservation, not a
competition submission.

The DINOv3 Kaggle code submissions that reached hidden reruns failed without a
public score. Kaggle's generic notebook error does not expose a hidden-test
traceback, so their cause remains unknown. This change is **not** part of the
immutable notebook submitted as 56546143.

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

Do not mutate or resubmit the failed 56546143. Any later
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
is private source preservation only. Kaggle publication happened later, as
recorded above; the Hugging Face commit itself created no Kaggle object.

The distinct fold-2 epoch-8 bundle was prepared on Newton before Kaggle
publication. Its [one-CPU script](prepare.sbatch) and durable
[dispatch intent](prepare-dispatch-intent.json) bind the staged source,
unchanged 1,215,161,641-byte checkpoint and source snapshot. Slurm accepted
CPU-only job **849389** once; it completed `0:0` in one minute with empty
stderr. A separate one-CPU, zero-GPU read-only verifier job **849399**
completed `0:0`, rehashing all 42 Dataset payload files, exact membership,
the checkpoint, generated notebook and metadata. See the
[terminal result](prepare-terminal-result.json) and
[independent verification](independent-verification.json). The prepared
[operation](prepared-operation.json), [manifest](prepared-bundle-manifest.json),
[snapshot](prepared-snapshot.json), [notebook metadata](prepared-kernel-metadata.json)
and [notebook source](prepared-kernel-run.py.txt) match the remote bytes. The
bundle is privately stored at
`/home/sa305415/rsna-knee/kaggle/bundles/fold2-epoch008-eb280864bd1c-metadata-v1`.
The subsequent private Dataset ID and notebook version are recorded above; a
verified MRI-only public run and fresh live allowance remain prerequisites for
a future Kaggle competition submission. Do not upload the three-row CSV or reuse the older
decoder-error bundle under this operation identity.
