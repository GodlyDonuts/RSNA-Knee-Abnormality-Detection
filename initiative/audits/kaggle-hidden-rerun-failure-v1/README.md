# Five hidden Kaggle reruns failed before producing output

Authenticated `GetSubmission` on September 23 returned `status=COMPLETE`,
`totalBytes=0`, no public score, and the same unhandled-notebook-error message
for submission IDs **56482890** (epoch 1 baseline), **56483451** (epoch 2),
**56483731** (epoch 6), **56483949** (epoch 4), and **56485356** (epoch 10).
[`submissions.json`](submissions.json) preserves the exact safe response fields
and live account allowance. [`capture.py`](capture.py) is the read-only
recollection method. These are failed hidden executions, not pending scores or
successful scored submissions. The canonical Kaggle ledger still records their
one-time submission identities; it must not be cleared to retry them.

The exact private offline notebooks all completed the three-study public
example with checkpoint-bound MRI-only receipts. That execution does not
exercise the roughly 1,300-study hidden roster. The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
says `test.csv`, `test_series.csv` and `test_series/` are replaced for scoring.
It does not explicitly say that the three-row `sample_submission.csv` is
replaced. The version-1 notebook used the sample file to choose test study
IDs. If the sample file stays at three rows, it conflicts with the replaced
series metadata and raises before writing a submission. **This is a leading
hypothesis, not a confirmed hidden traceback.** The API error is generic, and
the browser available to this session was not signed in for hidden logs.

The local correction makes `test.csv` authoritative and treats
`sample_submission.csv` as an optional schema check. Focused tests cover a
larger hidden roster with an unchanged three-row sample and an absent sample.
No corrected Kaggle version or new competition submission is claimed by this
audit. Keep the old epoch-8 version 1 out of the account-wide submission queue
while this failure is diagnosed and a distinct corrected private notebook is
verified. Hidden runtime and score remain unknown.
