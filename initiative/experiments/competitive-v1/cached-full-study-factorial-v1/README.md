# Label source × complete-study coverage on a frozen MRI representation

**Question:** does the sixteen-window training bag hide useful study information,
and does this interact with the new source labels? This is a cheap controlled
readout test, motivated by complete-context feature banks, not a claim of a new
invention or a0.97model. Existing full-encoder label training remains unchanged.

[Long-Term Feature Banks](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Long-Term_Feature_Banks_for_Detailed_Video_Understanding_CVPR_2019_paper.html)
uses stored features to supply context beyond a short video clip. That motivates
retaining the whole MRI bag while separating encoder cost; its video results do
not establish MRI accuracy. A possible later hybrid would refresh selected CNN
features while retaining the other original study features; that mechanism is
not implemented/admitted by this first counterfactual.

## Fixed four-way experiment

| Role | Source label values | Training windows |
|---|---|---:|
| original16 | Existing Stevenv6 accepted targets |16IID with replacement|
| released_S6_16 | Admitted S6v5 overlay |16IID with replacement|
| original94 | Existing Stevenv6 accepted targets |all94|
| released_S6_94 | Admitted S6v5 overlay |all94|

All four use the same3887fitting studies/order, both existing anatomical views,
accepted masks, original full94view-specific teacher, pretrained CNN features,
initial norm/attention/classifiers, effective8including the actual final7,
fixed3epochs/1458updates, headLR3e-5,WD.02,10%warmup/cosine-to1%,clip1 and
25%source/75%detached teacher objective. The source and observation policies are
the two explicit factors. The16draws/view variate match the current uniform
sampler; full94keeps its view variate while retaining all original windows.

The raw pre-normalization feature cache already exists; no new CNN inference,
MRI decode or feature transfer is needed. Only the exact279832norm/attention/
classifier parameters train. All four use CPUFP32 and zero attention dropout.
The original full CNN is fixed. These precision/frozen-encoder/dropout choices
also differ from the live CNN experiment, so that cross-family comparison is
not a clean encoder-only causal claim. Inference must use all94and mean of
original/mirrored probabilities. No new pixel or label file is modified.

ActualCPU947802bound the original full3887feature/old-new target data and
exactly replayed32initial FP32readouts. Maximum probability gap to cachedFP16
was0.0000847017; this is numerical acceptance, not disease performance. The
new256MiBnode33recovery/training/publication cap is separately funded beyond the
old611754508347required free bytes, giving612022943803. Shared new bytes/files0.
It does not borrow the current label campaign's checkpoints or serving budget.

Before full fitting, publish this exact source, then require actual independent
fresh full/stop1/resume processes for all four roles on16fitting engineering
studies plus16disjoint fitting-only prediction cases. Complete decoded model,
AdamW,LR/cursors/allRNG/history/every completed epoch snapshot and16×12 prediction
bytes must match. Old/new labels must cause actual learned differences. These
small runs cannot be submitted or establish accuracy. Full fitting requires
that actual recovery and a new exact one-time dispatch; no fit is started here.

Preserve all12immutable production epochs privately exactly once with licence,
source, parent, target/mask/teacher, runtime and roster provenance and full
re-download/size/SHA verification. ParentCNNweights remain referenced once;
only small learned readouts are stored. Recovery states remain original.
No best-epoch selection, internal Gold/confirmation tuning or whole-model OOF
claim is allowed. All3887rows were exposed to the released encoder. Fitting loss
is not generalization; any candidate requires complete original full-MRI T4/
runtime/offline/output qualification and an actual competition outcome. A
negative result is retained. No existing job, owner, Dataset or entry changes.
