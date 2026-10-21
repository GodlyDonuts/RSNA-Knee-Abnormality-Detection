# C100 score regression: first evidence, not a claimed root cause

October 21, 2026. Exact Newton C100 submission57491555 completed successfully
and scored **0.692**, versus external R6's0.958 and earlier adaptations'0.949.
[Original scored receipt](../predictive-C100-final-entry-20261021-v1/actual-scored.json).
This is a model regression. The prior hidden exceptions no longer explain this
particular scored result; the cause of those older exceptions remains unproven.

One-CPU1002780 read the original four terminal identities and log filenames.
Distinct one-CPU1002793 then read the four existing complete optimizer histories
once. It verified each original job/binding and the same fitting roster, all
2,916 contiguous steps, six epochs and 3,887 study exposures per epoch. It ran
no model, loaded no checkpoint or labels, and used no GPU or competition slot.
The [actual receipt](actual-existing-loss-comparison.json) binds the executed
[read-only observer](passive.py) and original log hashes.

| Original branch | Epoch 1 fitting exam BCE | Epoch 6 fitting exam BCE |
| --- | ---: | ---: |
| B: direct classifier + generic reconstruction | 0.616301 | 0.575507 |
| C: predictive evidence, density coefficient0.1 | 0.616277 | 0.583854 |
| C0: same predictive graph, coefficient0 | 0.616118 | 0.598655 |
| C100: coefficient100 | 0.616397 | 0.564260 |

These are study-weighted **pre-update training-trajectory averages**, not
fixed-final evaluation or held-out scores. C100 has the lowest last-epoch
fitting BCE among these runs, yet its actual competition score is poor. The
evidence therefore does not support choosing another model or longer training
simply because its training loss falls further. The other three original
competition entries failed and have no scores; their fitting losses cannot
rank their competition performance.

Every last-epoch B/C/C100 update triggered the existing global gradient clip1;
C0 did so for88.89%. This describes optimization and does not establish that
clipping caused the poor score. C100's previously saved seven-study last-update
diagnostic had lower BCE with its density correction than without it, but that
small fitting observation also cannot establish generalization.

Next useful distinction: evaluate the fixed final model's context classifier
and density-corrected readout separately, explicitly distinguishing fitting
diagnostics from a competition score. A successful inference path for the
existing fixed B/C/C0 controls would supply the originally intended model
comparisons. Do not retrain healthy originals, retune protected labels, infer
a density-correction root cause from this receipt, or run either proposed
C100 ensemble after its standalone-score gate failed. Colab's already-trained
independent final and original preservation/deployment remain unchanged.

Search hygiene: while locating an old R7 owner, an overly broad numeric text
search incidentally displayed existing CSV rows, including one protected gold
row. Those values were not used in this analysis, selection, fitting, or a
metric. No row/UID is reproduced here. Subsequent operational searches must
restrict file types/paths rather than search label tables. This incident does
not authorize protected-label evaluation or tuning.
