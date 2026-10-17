# Current project state

**Verified October 17, 2026, 10:55 UTC.** The owner targets **0.97**. Best actual
project public AUC remains **0.950**; the target has not been reached. Read the
[research brief](research-brief.md), [execution handoff](codex-continuation.md)
and [research-agent handoff](research-agent-handoff.md). Original receipts and
fresh live state override dated progress narratives.

## Actual scores

| Candidate | Submission | Public AUC | Meaning |
|---|---:|---:|---|
| Released S6, exact offline inference | **57300990** | **0.950** | COMPLETE/noerror; reproduced external weights and inference |
| Released nartaa CoAtNet | 57258866 | 0.949 | Reproduced external weights; not our source training |
| Our whole94 parent adaptation, epoch1 | 57294905 | 0.949 | No gain over unchanged parent established |
| Our independent N3, epochs4/8/12/16 | 57256756/57259376/57262392/57265916 | 0.821/0.869/0.890/**0.894** | Own MRI training from general ImageNet initialization |
| Context / full-field / regional additions | 57287572/57289392/57291182 | 0.948 each | No demonstrated gain over0.949 |
| Earlier own DINO | 57182737/57182547 | 0.872 | Historical own-model reference |

Public leaderboard scores are not private-final or clinical accuracy. Last saved
leaderboard first place was0.969; no current leading method is inferred.
[Score proofs](../initiative/audits/s6-actual0950-result-20261017-v1/README.md),
[leaderboard proof](../initiative/audits/github-current-state-refresh-20261017-v2/actual-public-leaderboard.json).

## New actual progress

**The exact public0.958 reference completed one interactive two-T4 MRI run.**
Both CoAtNet stages, original four-foldR6v1 and the three-checkpoint2.5D reader
executed. R6 scored3/3 with zero fallback and both worker exits0. Final830-byte
CSV passes3rows/13columns, exact publictest UID order and finite[0,1] values;
SHA5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b.
The session was explicitly stopped at about17minutes. CPU945812 reconciled all
8unchanged saved source cells and actual success under the canonical claim.
**No committed offline version or competition entry exists for this copy.**
Our139199164v1 remains source-only; never submit it. Keep R6Datasetv1 pinned:
its v2 contains different R7 weights under the same names. This is public
qualification, not our0.958 score or hidden-runtime proof.
[Exact qualification and controls](../initiative/audits/public0958-reference-20261017-v1/interactive-v1/README.md).

**Our whole94 adaptation completed all3epochs/1,458finite updates.** Original
139172537v1 is terminal success; CPU945622 verified complete final state and
privately preserved all5objects at HF1ea56fa5f9a263597df0a10cc87d98fe2dc1da7b.
All three epochs are already verified; do not republish. Sole final-serving and
entry owners17301/17317 remain live; no finalE3 score exists.
[Original terminal and final preservation](../initiative/audits/whole94-fixed3-final-20261017-v1/README.md).

**A matched source-label experiment is prepared and admitted on CPU.** S6v5's
released labels differ from our Stevenv6 labels across all3,887fit studies;
Fracture rank correlation is0.51446. CPU945573 verified the real full-array
label-only path. CPU945599 created private Dataset12653444v1 once and fully
re-downloaded/hash verified all5objects. CPU945706 funded4GiB checkpoints plus
16MiB controls, preserving all old liabilities/reserves. Twenty-four whole94
and8sampled CPU engineering checks pass. The fixed comparison uses the unchanged
uniformIID16 recipe, changing only source labels. **No treatment GPU recovery,
fit, VM reservation or score exists yet.**
[Protocol](../initiative/experiments/competitive-v1/s6-source-label-contrast-v1/README.md),
[actual data/storage proofs](../initiative/audits/s6-source-label-admission-20261017-v1/README.md).

The author's86missing studies were local disk deletions, not host removal or a
quality filter. S6's own recipe uses4263fit/16epochs/64windows/denseBCE/fixedE16,
noEMA/SWA and noOAI. Our3epoch25%source/75%parent adaptation is a different recipe.
[Original released-source comparison](../initiative/audits/s6-training-release-20261017-v1/README.md).

## Current owned work

| Work | Original identity | Latest actual state |
|---|---|---|
| Whole94 adaptation and preservation | 139172537v1;96833/96866 | **Terminal success**, all1458updates/all3private epochs verified |
| Final whole94 serving / entry | 17301/17317; standby28090 | Owned/waiting actual capacity, acceptance and allowance |
| UniformIID16 full fit | 139192503v1; sole observer34538 | RUNNING; epochs1/2 privately verified |
| InfluenceIID16 full fit | 139181903v1 | Epoch2 complete972/private verified; final role remains owned by34538 |
| S6 T4 fitting32 scout | **139210569v1**, sole34237 | Pushed once; APIstatusnull/UIqueued at last observation, not terminal |
| Independent scout analysis | 34308 | Waits original scout; no GPU/entry authority |
| S6 CPU32 scout | **139205901v2**,40709 | RUNNING; no new MRI result yet |
| Public0.958 reference | 139199164v1 | Interactive qualification complete/stopped; source-only saved version |
| Old equal-rank blend | old27596 | **Deprioritized before any GPU push/entry**, source preserved |
| Older R7 reference | old31309 | Deprioritized before push/entry; original assets preserved |
| Expanded own N3 | Slurm938284 | COMPLETED0:0/5184updates; all16epochs privately verified |
| S6 H100 pilot | original942047 | Last pending; never duplicate based on observation loss |

Fresh authenticated allowance at**10:54UTC:5used/0available**. No additional
competition entry was made this turn. Actual two-batch-GPU-session capacity
still governs batch work; Kaggle separately allocated our now-stopped interactive
session. Do not infer capacity or hidden scores from a missing API status.

## Evidence and limits

Expanded N3 fixed-final weak macro is0.9174315801 versus0.9209253491 original,
delta−0.00349377, conditional95%[−0.01331711,+0.00607252]: no established gain.
Earlier neighbor/weighting/averaging/stratification and new context/field/regional
results remain in the research brief. Protected clinical values remain unused.
Released encoder exposure prevents independent whole-model OOF claims on its
training cohort. Weak-label agreement and green engineering checks are not
clinical truth or predicted Kaggle AUC.

The original detailed10:15UTC guide remains recoverable from
[the preceding Git snapshot](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/ca90fa5e347310dbbb35765e5dcc5c1b07e04910/docs/current-state.md); original receipts
and archived source contracts are never replaced with new narratives.

The source9c083a4b milestone is now privately mirrored/full37object download/SHA
verified at HF7a14fc051d913b14b8e2895cc8db60fa6166e2ce.
[Exact private preservation](../initiative/audits/hf-source-labels-exact0958-final3-20261017-v1/README.md).
