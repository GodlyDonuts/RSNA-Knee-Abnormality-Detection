# Original C100 pilot terminal observer

**Started once at18:11:18 UTC**, PID **15130**, frozen observer source
`2488a4086fc854af733c7accf84377ff69f9f6fc`. Actual18:11:25 readback
confirms its exclusive intent, empty stderr, zero GPU allocations and original
pilot **981620 PENDING**. [Actual start](actual-started.json); [live proof](actual-live.json).
This finite read-only owner observes only that original job, source `068a14a7`,
binding `e11c7c7f…`, on **evc27**. It neither relocates the pilot nor
admits/promotes a full C100 run. No terminal result exists yet.

The original launch used the following interface; **do not run it again**:

```bash
python3 initiative/research/predictive-density100-observer-20261020-v1/observe.py \
  --source-git <exact-committed-observer-source> \
  --socket /tmp/rsna-research-20261001-stable.sock
```

The exclusive ignored journal is
`artifacts/predictive-density100-observer-20261020-v1`. Any existing journal
requires read-only reconciliation; rerunning is rejected. Source bytes are frozen
at the supplied commit and retained in memory; later Git commits cannot rebind
the running observer.

Every 120 seconds it reads `sacct` and the account-wide live queue, filtering the
exact original job. It stops initiating observations at **October 21 17:15 UTC**.
A terminal result observed before that boundary permits one bounded terminal
collection already in flight to finish, at most five minutes later. This uses
**one CPU, zero GPUs**, at most three allocated minutes, on evc27; it writes no
remote files and does not read model tensors or MRI. No new shared files are made.

Completed `0:0` requires the original binding, preparation, run identity,
24-update finite history, full/stop/resume acceptance and matching original
terminal. `actual-accepted.json` retains the exact unchanged sanitized
`GPU-accepted.json` bytes. `actual-terminal.json` records its byte count/SHA and
the original source/job proofs. Strict allowed keys reject private identifiers,
prediction arrays or unexpected fields before export.

A failed scheduler outcome produces only `actual-terminal-failure.json`,
containing compact state and log hashes/sizes. It cannot export a success or
resubmit the pilot. Repeated scheduler transport loss is bounded to 30 minutes.
An explicitly unallocated immediate CPU request may be retried within the same
bound. A timeout, SSH loss or other uncertain outcome after requesting CPU
collection stops for reconciliation; it is not replayed.

No tokens, Kaggle operations, source downloads, model updates, automatic
promotion, or accuracy claims are part of this observer.
