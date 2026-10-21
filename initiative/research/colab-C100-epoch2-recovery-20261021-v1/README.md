# Colab epoch-two recovery: separate fitting from publication

At October 21 14:01 UTC the original epoch-one successor was observed terminal
with exit 1. Kaggle returned HTTP429 from CreateDataset during epoch-two
publication. Full checkpoint at update972 is intact,880,115,123bytes,
SHA `0fab361de8da414968391b8f7a596190d68cb1e2966e72e96429bcc8b5dae516`.
Original logs contain487–975; original process may have performed update976
before its publication exception, without logging/saving it. No model or input
failure is shown. The authenticated complete six-page owned Dataset listing
still excludes the exact intended epoch-two slug; metadata returned403. No
creation or upload was retried during recovery.

The distinct recovery restores the same complete972 model/AdamW/RNG/history and
unchanged numerical fit, binding, recipe and data order. It replays973–975 and
any unlogged976. New output is `run-C100-seed20261021-recovery-v2`. Epochs1/2
and both failed predecessors are retained. Original ten-hour deadline remains.

Publication is decoupled: new checkpoints/exports are immutably saved and hashed
locally, with explicit **publication pending** receipts. This worker performs
zero network/account operations and never labels a local checkpoint as remotely
verified. Available Colab disk70,540,644,352bytes funds an incremental10GiB with
the20GiB floor. No input download, GPU allocation or Dataset creation is needed.
All new checkpoint bytes must be privately exported and independently verified
before runtime release or deployment. The final-serving contract now binds both actual recovery sources and checks
epoch2 against its exact complete checkpoint SHA. It requires separate private
byte proofs for all six epochs and the terminal; local storage alone cannot
produce an admitted handoff.

Three focused tests check unchanged model/data-loader AST, exact source pin and
resume target, local immutable storage without false remote acceptance, and the
post-fit publication plan. Scoped serving/handoff tests additionally reject
missing private proofs and altered recovery checkpoint bindings.

Actual recovery cell `cell-jn0mRnxvyCa0` is running from source3bdd97c2; finite
encoder updates were observed through1080/2916 in epoch3. The earlier retained
[progress receipt](actual-resumed-progress.json) includes exact admission and
step1060. No accuracy improvement is established.

Paced publication cell `cell-xtKDvqCk0sT2` is queued, not executed, from
source0b83e278. Its pinned script SHA is
`89dec4c3e19b8fcf3595f735da445a163fb7132bc2005067f15fdf0c51ec4839`.
It waits for the complete fit, reuses epoch1, rechecks the known absent epoch2
slug before a distinct explicit attempt, and spaces new Dataset creates by
at least900seconds. Every object must be privately downloaded and size/SHA
verified. An uncertain mutation stops without retry or runtime release. The
A100 is released only after all seven proofs and final metadata are printed.
This publication step uses the original isolated private namespaces and
storage reservation; no competition submission or new GPU is requested.
The deadline and accepted scientific recipe are unchanged.
