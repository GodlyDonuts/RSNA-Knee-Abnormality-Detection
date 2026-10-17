# Actual fitting-only teacher-view screen

CPU941335 completed once in **1.288 seconds**, using the original admitted
3,887-study paired teacher logits. The source/UID and teacher-array identities
match the original input proof. No source labels/masks, reports, images, weights,
protected clinical values, encoder inference or optimizer were accessed. Only
aggregates were collected; raw records remain private.

The predeclared triage is **intermediate**, not weak and not a promoted nontrivial
screen. Mean absolute plain/mirror probability gaps range **0.01287–0.02707**.
Strict-pair ordering discordance ranges **2.75–7.41%**; Baker's is highest (7.41%),
then MCL (6.69%). The largest fraction of studies with >0.10 probability gap is
Contusion, **4.37%**, below the predeclared 10% threshold. Individual tail gaps can
be large, but that does not establish clinically wrong predictions.

These are all-study teacher ordering pairs, **not positive-negative AUC errors**:
clinical labels were deliberately not read. The probability-consensus MSE is
imitation fidelity, not disease accuracy. The screen does not admit another GPU
arm or alter current source labels, targets or healthy training. A later matched
consensus-target experiment remains a hypothesis, subject to actual current
results and fresh funding.

[All twelve target aggregates](actual-aggregate.json),
[predeclared protocol](../../experiments/competitive-v1/teacher-view-consistency-screen-v1/README.md).
The exact executed source is sourceGit6c5d4454, SHA
0e9b5164851efde48629415a6897c8774d71f724f2bc26711c91b27ff624b4f1,
preserved in `actual-executed-source.py.txt`. Later formatting does not replace
that receipt. Four numerical tests cover ties, non-rank-changing shifts, complete
reversal and nonfinite refusal; they are not clinical validation.
