# Patch-attention fold-1 fixed-final result

Slurm job 847119 completed ten epochs and 2,850 optimizer updates on 2,278
training studies. The fixed final epoch scored **0.7214765382 macro AUC** on
1,125 held-out outer-fold-1 studies with accepted report-derived labels. The
matched global-CLS fold-1 model scored **0.7128332913** on the same outer
fold, an absolute difference of **+0.0086432469**. These are internal results,
not Kaggle scores, and the outer labels were not used to select an epoch.

The predeclared lateral-meniscus target scored 0.6930797721 versus 0.6691644682
for global CLS, a gain of 0.0239153039. Fracture fell from 0.4520547945 to
0.2557077626. The fold-0 comparison had a lower patch-attention macro AUC
than its matched CLS baseline, while lateral meniscus improved. Consequently
the two-fold evidence does not justify replacing the global head wholesale.

The exact final result, predictions and run binding are preserved in
`result.json`, `outer_predictions.csv` and `run.json`. Their hashes were checked
against the Newton run before archiving. `epoch-publication/` contains the
single-attempt intent, commit, manifest and full private download verification
for each of ten immutable epoch checkpoints at `GodlyDonuts/RSNA-Data`. Each
epoch has four verified private objects. The epoch-10 revision is
`7bc52efec2141405be9e3a0fddb973398df9da1f`. After verification, a
one-CPU retention job removed only redundant local epochs 1–9 `.pt` copies,
reclaiming 10,954,517,985 bytes. The final local checkpoint, result files,
sidecars and all private epoch copies remain; see
`local-retention-fold1-result.json`. This experiment has not been submitted to
Kaggle.

The [epoch training trajectory](epoch-training-trajectory.json) evaluates all
ten completed epochs against the original 2,850-update optimizer log (521,656
bytes, SHA-256 `0f4db3061473ec9a8af1b748d965d3d61e789cf35c40d1a58a71242a4c9cf4dc`).
Each epoch has exactly 285 consecutive finite updates and is joined to its
privately verified checkpoint SHA and Hugging Face revision. Mean training loss
fell from 0.226790 in epoch 1 to 0.157921 in epoch 10. These are **training
diagnostics**, not independent validation AUCs or evidence of a Kaggle score.
The held-out outer-fold result above remains the predeclared final-epoch
evaluation; the outer labels were never used to choose a checkpoint.

The 49 small result, journal and source objects were separately mirrored to
the same private Hugging Face dataset and fully re-downloaded/hash verified at
revision `6055fd9507ae45893b7aff083c4dc4423d292b32`; see
`private-mirror-receipt.json`. The large checkpoints remain in their ten
immutable per-epoch private paths.
