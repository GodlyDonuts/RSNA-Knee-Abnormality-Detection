# Private hidden-scale inference diagnostic

Kaggle's code submission 56507997 produced no hidden output or score. The
user-provided log belongs to the successful three-study public run, so it does
not identify the hidden failure. This private diagnostic uses the exact
fold-0 epoch-8 `fix4` checkpoint bundle and offline T4 environment. It first
reruns the public example, then runs the same image inference code on 1,300
official **training-image** studies selected deterministically from the
competition mount. It reads study and series identifiers only, never training
labels. Its second CSV is named `pseudo_submission.csv` and is solely a
diagnostic artifact. **This notebook push did not create a competition
submission.** A separate repeat of the old fix4 notebook appeared in the
account at 10:49 UTC, two minutes before this diagnostic push; see the
[read-only account reconciliation](account-reconciliation-20260924.json).
Its origin is not established by this audit. The single Newton ledger still
contains only the original fix4 submission. Kaggle reports three live slots
remaining, so no further competition submission should be made until this
repeat is understood and its actual outcome is known.

The frozen [plan](plan.json), [source](notebook-source.txt), one-call
[push intent](push-intent.json), [result](push-result.json), and
[remote verification](remote-kernel-verified.json) bind private offline notebook
`godlydonuts/rsna-dinov3-e8-train1300-p01` version 1 to source SHA-256
`9dd8e8963d0cc227adce0c5267b14a7733f669618171030f086196a928a8e5aa`.
The remote settings and source bytes were read back and verified; the notebook
was `RUNNING` immediately after the push. Do not repeat `SaveKernel` or create
another competition entry from its public-example CSV. Inspect the exact
version's terminal session status, logs, metrics and output receipt. A passed
1,300-study training-image diagnostic would narrow hidden-scale runtime and
ordinary DICOM-path hypotheses; it would not prove the hidden rerun will pass
or establish a Kaggle score.

The owner's earlier 0.936 and 0.942 OSS notebooks apply percentile-rank
transforms. With just three public studies, these can produce visible fractions
such as 1/3, 2/3, and 1. Our notebook emits raw sigmoid probabilities, giving
more varied decimals. ROC AUC depends on ranking, so decimal formatting cannot
explain the hidden notebook exception.
