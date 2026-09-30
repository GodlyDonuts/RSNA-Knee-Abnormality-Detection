# A32 fold-0 epoch-10 Kaggle candidate

September 30, 2026. The owner explicitly requested this completed checkpoint.
SHA `3ed9a08f144fb0b3e30dee7bfc65df13f5c4647b2a0532a8d04d3241da0d4db0`;
training Git `6f1ee6d332de29a8d6e5807aeac4777620865933`, 2,810 optimizer updates.
Internal fixed-final macro AUC 0.8620940794 on 1,156 held-out development studies
is not a Kaggle score. All ten original checkpoints are privately HF verified.

The deployment projection uses packaging Git `57aaca6320d745538d169b76cea34ecaea40e076`
and byte-exact historical training rsna modules. The only eligibility change
accepts completed even checkpoints from cadence-one preservation as well as the
older cadence-two runs; it does not rewrite a sidecar or weight binding. All
1,082 checks passed before this packaging commit. Full all-data training is a
separate source and model; its weights cannot be swapped into this candidate.

CPU preparation **865252 COMPLETED 0:0** under a one-CPU, zero-GPU allocation.
One private Dataset create RPC returned successfully. Exact Dataset
`godlydonuts/rsna-knee-a32f0e010-3ed9a08f144f`, ID **12282965**, version **1**,
was fully downloaded and every payload's size/SHA verified. The private offline
T4 notebook `godlydonuts/rsna-knee-a32f0-epoch10-3ed9a08f144f` version **1**
was pushed once through the canonical Newton coordinator/ledger. Its exact
source, privacy, offline GPU configuration and Dataset sources were verified.
The public MRI-only execution completed. Exact source, checkpoint/bundle receipt,
13-column output, public test UID order, finite probabilities and CSV SHA all
passed. One version-bound code submission returned Kaggle entry **56697342** at
2026-09-30 04:39 UTC. The submission names the exact notebook version and its
`submission.csv` output; the three-row public CSV was not uploaded. The first
read-only authenticated outcome remains null status/score, with four live slots
remaining. Do not resubmit this candidate or infer hidden success from its public
run. Inspect the actual hidden outcome before selecting another checkpoint.

The bundle retains original sigmoid probabilities, licenses, physical cache
geometry, corrected series/pixel handling and chunked hidden-scale inference.
T4 deployment explicitly uses FP32 although training used H100 BF16. Private
shared runtime Dataset ID 12139697 is unchanged and remains byte-bound. This
public execution is a deployment check, not hidden-run success or a score.

All later submission mutations use only
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3`. Existing dispatch intents
or uncertain API results require read-only reconciliation, never another push,
Dataset create or submission retry. Credentials stay local and are streamed
through stdin to bounded CPU-only phases. Large weights/bundles remain private
on Newton and private external repositories.
