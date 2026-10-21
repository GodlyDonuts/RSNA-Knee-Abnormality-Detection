# Colab epoch-two recovery: separate fitting from publication

At October 21 14:01 UTC the original epoch-one successor was observed terminal
with exit 1. Kaggle returned HTTP429 from CreateDataset during epoch-two
publication. Full checkpoint at update972 is intact,880,115,123bytes,
SHA `0fab361de8da414968391b8f7a596190d68cb1e2966e72e96429bcc8b5dae516`.
Original logs contain487–975; original process may have performed update976
before its publication exception, without logging/saving it. No model or input
failure is shown. The authenticated complete six-page owned Dataset listing
still excludes the exact intended epoch-two slug; metadata returned403. No
creation or upload is retried.

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
before runtime release or deployment. The former completion/consumer contract
must be updated; no stale completion handoff should be issued.

Two focused tests check unchanged model/data-loader AST, exact source pin and
resume target, plus local immutable storage without false remote acceptance.
Actual GPU recovery/progress must still be recorded after launch.
