# Earlier scored notebooks change the hidden-failure diagnosis

Authenticated `ListSubmissions` and `GetSubmission` on September 23 found two
earlier completed entries: **55969300 scored 0.936** (September 2) and
**56407271 scored 0.942** (September 20). Both produced nonzero submission
files. The five later DINOv3 entries still have zero output bytes and no score.
The read-only capture in [observations.json](observations.json) records safe API
fields and source hashes. The account's private version-1 notebooks whose run
times immediately precede those submissions are
`godlydonuts/head-and-shoulders-knees-and-toes` and
`godlydonuts/rsna-knee-0942-restructured`. The submission API does not expose a
kernel reference, so the notebook-to-submission association is supported by
account ownership, version, title and timing rather than a direct identity field.
Their original source is retained only in the ignored local review directory;
this audit does not republish third-party code.

Both notebook sources use `test.csv` to determine the test study roster and
write `/kaggle/working/submission.csv`. The 0.942 source additionally reads
`sample_submission.csv` in an executed top-level code cell and raises if its
study IDs differ from the `test.csv` IDs. Its successful hidden score therefore
**strongly undermines** the earlier hypothesis that Kaggle leaves the sample
file at only three rows while replacing `test.csv`. The sample-roster correction
in the prepared DINOv3 epoch-8 notebook remains sensible, but the evidence no
longer supports treating it as the likely solution to all five failures. Kaggle
withholds the hidden traceback by design; no root cause has been proven.

The scored notebooks catch DICOM read/decode errors and tolerate absent or
unusable series. Our unchanged trained inference path currently attempts to
decode **every** series listed for each test study, including series the model
would not select, and raises on any bad series, missing local DICOM, invalid
geometry or pixel data. A known [competition discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163)
identifies two corrupt public training DICOMs; that does not prove corruption
in the hidden test set, but it makes strict all-series decoding a concrete
failure risk. The successful sources also use both possible competition mount
paths, as our current notebook does. Their final output checks cover study IDs,
column order and finite values, as ours does.

**Disposition:** hold the current corrected epoch-8 version from competition
submission despite its three-study public pass. Investigate and test robust
series handling and full-size inference locally, preserve exact trained-code
binding, publish any deployment change under a new immutable notebook identity,
and use at most one newly verified hidden submission when quota permits. Do not
reuse a failed or uncertain submission mutation, and do not claim the earlier
0.936/0.942 scores for our DINOv3 model.

The six small source/audit files from GitHub commit
`55e1073f5269aa72e12d0d7f014418dae2abb04f` and a byte manifest were
committed once to the private Hugging Face dataset at revision
`ee6fd646ad64d72c93abceed33735a92d657cfe8`. The
[verification receipt](hf-verified.json) confirms that all seven objects were
downloaded and hashed. The private source notebooks themselves were not
republished.
