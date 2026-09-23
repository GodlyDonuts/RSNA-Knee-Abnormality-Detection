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
series metadata and raises before writing a submission. This was an initial
hypothesis, not a confirmed hidden traceback. A subsequent comparison with
the owner's earlier scored notebooks [strongly disfavors that hypothesis](../kaggle-successful-submission-comparison-v1/):
the 0.942 notebook checks that `sample_submission.csv` and `test.csv` have the
same study IDs. The API error is generic and Kaggle intentionally withholds
specific hidden-run traceback details.

The local correction makes `test.csv` authoritative and treats
`sample_submission.csv` as an optional schema check. Focused tests cover a
larger hidden roster with an unchanged three-row sample and an absent sample.
The distinct corrected epoch-8 private notebook later passed its three-study
public run but has not been submitted. Hold it while the broader failure is
investigated; the sample-roster change alone is no longer a justified basis
for spending a competition slot. Hidden runtime and score remain unknown.
