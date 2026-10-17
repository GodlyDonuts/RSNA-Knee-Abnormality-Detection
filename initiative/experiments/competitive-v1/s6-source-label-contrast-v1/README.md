# Matched S6 source-label contrast

The owner's target is 0.97. Actual best remains 0.950. This experiment tests a
measured supervision difference rather than assuming a larger model will help.

## Mechanism and fixed comparison

The newly released S6 v5 labels differ from the Steven v6 labels currently used
by our adaptation. The fitting-only comparison covered all 3,887 studies and
46,644 cells; Fracture rank correlation was 0.51446. Neither source is clinical
truth. See the [original source comparison](../../../audits/s6-training-release-20261017-v1/README.md).

Use the unchanged **uniform IID16** full-encoder engine as the control. Keep
the same 3,887 fitting studies, parent initialization, study order, random draws,
views, masks, detached parent predictions, 25% source / 75% parent objective,
optimizer, effective batch eight, three epochs / 1,458 updates and full94
two-view inference. Replace only source-label values with the admitted S6
overlay. The existing uniform run remains untouched. This choice was made
before any treatment fitting; it saves compute relative to a second whole94
training arm and permits the closer matched comparison.

`source_overlay.py.txt` changes only labels and binds both source roles to
exact array/file/population hashes. `sampled_fit.py.txt` delegates to the
unchanged original uniform trainer. Cross-source checkpoint resume is refused.
Protected clinical values, confirmation and gold are not used. Released parent
exposure prevents calling this internal whole-model OOF.

## Actual acceptance, October 17

- 24 synthetic CPU checks exercise the real whole94 trainer: control complete
  state/prediction identity, full/stop1/resume equality, negative input checks
  and a nonzero treatment effect on learned parameters.
- Eight additional checks exercise the **actual unchanged sampled trainer**:
  repeated draw order, view/mask/teacher identity, complete state and full94
  predictions, both source-role recoveries and the singleton final batch.
- CPU945573 admitted the real full3,887 array path. Existing UIDs/masks/two-view
  teacher arrays and original target bytes are unchanged. No new MRI rendering,
  encoder, optimizer or GPU execution occurred in that admission.
- CPU945599 created the one private Dataset
  `godlydonuts/rsna-s6-fit-labels-5e34bbf40f4f`, **12653444v1**, and fully
  re-downloaded/size/SHA verified all five objects. Its 613,297-byte target file
  SHA is `5e34bbf40f4f5b21638d2a1dad853912dee98533964a5236a8f5328bd3d388a9`.
  Do not recreate or upload it again.
- CPU945706 funded a distinct node-local **4 GiB checkpoint/publication plus
  16 MiB control** envelope. All previous liabilities and reserves remain.
  This does not reserve GPU time or authorize an unaccepted production wrapper.

These checks establish engineering properties, not MRI accuracy. Actual GPU
recovery and treatment production have **not** run. Require a distinct immutable
wrapper, fresh all-future VM funding/capacity and actual full/stop1/resume MRI
acceptance before treatment fitting. Keep the fixed-final comparison, preserve
all immutable epochs once, and judge the treatment by actual eligible scores
and appropriately exposed held-out evidence. Preserve a negative result.

The [data/publication/funding receipts](../../../audits/s6-source-label-admission-20261017-v1/README.md)
contain exact pins. No private labels, case membership or pixel arrays are in Git.

## Actual real-payload recovery preparation, 12:02UTC

`gpu_recovery.py.txt` compilerSHA404d8cc7… uses the unchanged accepted private
recoverya01dd964… and exact overlayf6414308…/sampledadapter5621ee32…. It preserves
all ten original numerical modules; both source roles use the original uniform
recipe. Control complete state/predictions must equal the actual oldT4result,
not merely a second new run. Both roles require complete full/stop1/resume learned/
AdamW/LR/cursor/allrankRNG/scaler/history/epoch and16×12full94prediction equality.

Actual CPU946080 compiled source0a6f6c567901…/worker7b7a1bf3e5a8… and verified the
real32case array facade against original input bridge939362. No MRI/GPU ran.
`check_gpu_recovery.py.txt` has7adverse compiler/data checks;
`check_recovery_contract.py.txt` has5adverse contract groups, including missing
old obligations, insufficient quota, missing T4/control proof and changed proposals.
`recovery_contract.py.txt` has no cloud mutation method or private payload.

Single finite owner46281 can push one diagnostic under fresh funding/capacity
and observe it. Its first actual check passed110405.97/66000VMseconds but both
batch lanes were occupied; no push/VMreservation existed. It cannot launch a
full fit or competition entry. A separate actual GPU pass and immutable production
wrapper are still required. [Original actual receipts](../../../audits/lockin-097-execution-20261017-v2/README.md).

## Full-population template, not executed production

`production.py.txt` is now actually CPU-admitted by946417 on the original
private uniform production source and all3887source arrays. All13numerical
modules remain byte exact; the two accepted label helpers are the only additions.
The builder requires the actual new GPU recovery's COMPLETE/source/version and
complete receipt. Original MRI parent/order/draws/optimizer/1458step schedule and
continuous per-epoch state stay fixed. Six adverse compiler groups pass.
`epoch_verify.py.txt` adds strict source-label/teacher/mask/recovery bindings and
retains all original numerical state checks; three synthetic adverse groups pass.
No executable epoch, full fit, GPU acceptance or accuracy result is claimed.
