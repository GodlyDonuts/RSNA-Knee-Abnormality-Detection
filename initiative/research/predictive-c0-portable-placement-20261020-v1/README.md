# Accepted C0 placement on evc27

At **October 20, 13:43 UTC**, the exclusive idle-owner handoff completed.
Fitting **68792** and deployment **77985** stopped before any phase; no remote
job was cancelled. New fitting owner **80995** and deployment successor
**80996** are live with original intent files and empty stderr, from committed
and pushed control source `926b657770564c79b23509592e2d6a95761abd70`.
They wait for the complete cache and actual finals respectively. **No context
transfer, full predictive GPU fitting or predictive deployment phase has run.**
[Actual handoff/start](actual-owner-started.json); [fitting handoff](actual-fitting-owner68792-stopped.json);
[deployment handoff](actual-deployment-owner77985-stopped.json).
The [13:45 UTC readback](actual-owners-live.json) confirms both new owners alive,
empty stderr and zero phases; the original deployment phase directory is absent.

The accepted future placement is **B/C on evc33 and C0 on evc27**, one ordinary
H100 per arm, preserving frozen `44cc559e` numerical science and initialization.
Healthy cache owner **68411** and jobs **979664/979665** remain unchanged.
The [13:32 UTC observation](../parallel-serving-account-20261020-v1/actual-input-progress-1332.json)
recorded **842 raw/840 contexts of 3,887**. Raw throughput of approximately
3.38 studies/minute leaves about 3.4 hours against the 20-hour limit if sustained.
Earlier node-27 scheduler forecasts are not reservations or guaranteed starts;
no full-cohort training or relocation speedup has been measured.

## Narrow execution path

`owner.py` freezes itself, `placement.py` and `remote.py` at a new commit. It
requires actual stopped-before-phase receipts for fitting owner68792 and
deployment owner77985, plus an accepted funding decision. It does not stop
either process. The handoff must demonstrate no child processes, no original
admission/dispatch/deployment intents, no predictive full GPU job and no cancelled
remote jobs. Existing cache producers continue unchanged.

Once both original3887 producers are actually complete, this sole successor:

1. Executes frozen44cc admission and owner controls on33 with arms explicitly
   filtered to B/C. It immediately submits those two original jobs once.
2. Uses one CPU on33 and one on27 for a private framed contexts-only transfer.
   No pixels, private UIDs, labels, source bodies or weights traverse Mac/Git.
   The source reuses original whole-byte proofs plus current file stats;
   the receiver hashes all streamed bytes. Per-context completion receipts
   preserve exact checksums and every original renderer-provenance field.
   Only destination path/stat fields change. The small private capsule carries
   the unchanged full target table, exact own numerical source, existing parent
   source/license and source/admission receipts. No raw DICOM or runtime copy.
3. On27, reuses the original accepted55,685-file runtime/asset replica from
   CPU978186 and the current original parent stat proof. The unchanged44cc
   initialization function executes on CPU with27's admitted overlay path;
   its complete model/schema/version results must equal CPU979667, including
   shared tensor digest`dbf69ff8…`. The ordinary two-H100 node and fresh
   storage are checked again before C0 dispatch. The original worker, model,
   optimizer, schedule, seed, targets, ordering and publisher remain byte exact.
4. Submits only C0 on27, one H100/two CPUs/eight hours. Polls all three exact
   jobs and preserves every sealed epoch once. The unchanged publisher runs
   on each actual owning node under the existing canonical shared lock.
5. After actual C0 completion and all six private five-object verification
   receipts, returns only final epoch6 objects plus the unchanged original
   terminal/publication receipt to
   `/tmp/rsna-predictive-C0-return-20261020-v1` on33, under a512MiB cap.
   Original27 states and epochs remain retained. The return does not republish.

Each mutating phase has a no-overwrite intent. Unknown mutation outcomes stop the
finite owner for read-only reconciliation; no job, transfer or upload is
automatically retried. Read-only progress observation can recover from SSH loss
for at most30minutes. Receiver connections allow at most30minutes for the source
CPU allocation; the existing2.5-hour context and45-minute return allocations
remain finite. Per-dispatch receipts record the actual submitted arm count,
physical two-GPU node limit and aggregate three-GPU placement limit. `missing_contexts` can identify already verified destination members,
but this first owner deliberately refuses an existing receiver root: any future
interrupted-transfer continuation needs a separately reviewed receipt and
control, never a blind owner restart.

## Scientific versus placement provenance

Scientific files remain source`44cc559e947560a635c34fee7b1f2a3b7be3a241`;
the worker and publisher hashes are checked against actual CPU979667. New
placement-control commit/hashes and exact handoff digest are separately recorded
in C0 runtime, input admission, funding and binding. The original runtime and
admission receipt hashes therefore differ legitimately between nodes. The
publisher's binding digest binds the complete new placement-bearing binding;
it does not claim the new CPU wrapper was part of44cc.

Original mixed-renderer provenance is retained for all3887 contexts. Original
context-index SHA remains the source identity; the relocated index has its own
SHA and fresh owning-node stats. No old producer hash is relabeled.

## Funding and deployment

Destination27 retains191GiB+564MiB existing liabilities and adds96GiB contexts,
12GiB C0 fitting/publication and64MiB controls, plus20GiB free reserve. Before
any paid bytes this requires343,182,147,584 free bytes; the recorded observation
leaves492,158,087,168 bytes above that requirement. Fresh admission is mandatory.
The whole96GiB remains reserved minus actual paid context bytes, not an assumed
smaller projection.

The original unused33 C0 12GiB scope remains reserved and covers its≤512MiB
final mirror. Original six GPU log files/24MiB are reassigned only after the
exclusive handoff; there is no additional shared log scope. CPU operations use
stdio. The existing376-file future scope retires only exact original dispatch
stdout/stderr files once they physically exist as owned private regular files.
At most six files are subtracted; missing files remain reserved. The deployment
observer applies the same proof to its other368-file scope, retaining its own
eight serialized account files. Shared byte liabilities stay conservative.
The accepted decision retains **123 MiB/376 future shared files**, minus only
those exact physically paid GPU logs. Pending decision files remain disabled
templates; they are not the decisions used by the live owners. Retain all
current obligations plus the 20 GiB/5,000-file reserves.

Root completed the committed-control review, idle-owner handoff and funding
acceptance before launching these two owners once. `deployment.py`
is the distinct finite successor over frozen account-owner source8fd43b98. It
reads this directory's actual finals/publications, requires actual B/C node33
and C0 node27, verifies the explicit final-return receipt, and uses the returned
C0 run root/license. The generated owning33 observer verifies the return receipt
SHA and counts only actual on33 mirror bytes, retaining the unspent remainder
of the original12GiB scope. The existing CPU serving compiler consumes that exact
seven-file mirror and checks the original C0 Slurm job. No node33 C0 training
receipt is fabricated. Numerical serving and canonical account controls remain
frozen8fd; the new placement/deployment adapter has separately bound source.

`deployment-decision.pending.json` remains disabled; the live owner uses
`deployment-decision.accepted.json`. Its initial Kaggle requirement is
**3,360 GPU-VM seconds**: 1,200 for combined qualification, three 600-second
public watchdogs and three 120-second public VM startup allowances. Actual
trained precision/memory, runtime decision, authoritative capacity and submission
allowance still govern each mutation. Hidden runtime remains unproven. The new
control is not permission to overdraw that envelope or use paid credits.

Launch interfaces retained for provenance; **the owners have already started,
so do not rerun these commands**:

```sh
python3 initiative/research/predictive-c0-portable-placement-20261020-v1/owner.py \
  --source-git <new-frozen-commit> \
  --decision <accepted-placement-decision.json> \
  --handoff <actual-fitting-owner68792-stopped.json> \
  --deployment-handoff <actual-deployment-owner77985-stopped.json>
```

The separate deployment launch interface is:

```sh
python3 initiative/research/predictive-c0-portable-placement-20261020-v1/deployment.py \
  --source-git <same-new-frozen-commit> \
  --decision <accepted-portable-deployment-decision.json> \
  --fitting-handoff <actual-fitting-owner68792-stopped.json> \
  --deployment-handoff <actual-deployment-owner77985-stopped.json>
```

Local tests cover original one-arm Slurm dispatch with a fake scheduler,
missing-only exact transfer and corrupted/interrupted refusal, actual original
launch-journal handoff, read-only retries versus single-attempt mutations,
node27 final/return provenance, real generated observer filesystem accounting,
and disabled unaccepted decisions. These are operational tests, not actual
transfer, GPU acceptance, training progress or clinical accuracy.

Operational ownership and placement are accepted. No measured speedup, model
improvement, completed transfer or full predictive training is claimed.
