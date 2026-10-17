# Current project state

**Verified October 17, 2026, 13:48 UTC.** Our target is **0.97**. Best measured
project public AUC remains **0.950**. The target has not been reached.

| Candidate | Submission | Public AUC | What was demonstrated |
|---|---:|---:|---|
| Released S6, exact offline inference | **57300990** | **0.950** | External weights and inference reproduced; COMPLETE, no error |
| Released nartaa CoAtNet | 57258866 | 0.949 | External weights reproduced |
| Our whole94 parent adaptation, epoch 1 | 57294905 | 0.949 | No improvement over unchanged parent established |
| Independent N3, epochs 4/8/12/16 | 57256756/57259376/57262392/57265916 | 0.821/0.869/0.890/**0.894** | Own MRI training from general ImageNet initialization |
| Context / full-field / regional additions | 57287572/57289392/57291182 | 0.948 each | No demonstrated public gain |
| Earlier own DINO | 57182737/57182547 | 0.872 | Historical own-model reference |

Public AUC is not private-final performance or clinical accuracy.
[Score receipts](../initiative/audits/s6-actual0950-result-20261017-v1/README.md).

## Training that is actually running

The **source-label comparison** is training all 73,136,096 parameters on the
same 3,887 fitting studies as the completed uniform control. Source labels are
the only scientific change: images, masks, teacher, sampling, optimizer and the
fixed three-epoch/1,458-update recipe remain matched. Neither label source is
clinical ground truth.

- **139234107 v1** is RUNNING on two T4s. Exact source/runtime/inputs were
  verified; live logs already showed 258 finite optimizer updates.
- Its actual GPU recovery **139230981 v1** passed complete old-control replay,
  both-label full/stop/resume state equality and prediction-byte equality.
- **60358** owns epoch 1. **61037** verifies and privately preserves each completed
  epoch before resuming epochs 2/3. No completed treatment epoch or new score
  is claimed. Scientific source remains `990c8ded` throughout the campaign.

[Actual recovery, training and continuation](../initiative/audits/097-label-campaign-continuation-20261017-v1/README.md).

## Completed models awaiting scoring

Whole94, uniform16 and influence16 each completed three epochs and 1,458 updates;
all nine immutable checkpoints are privately fully download/size/SHA verified.
Do not repeat their training or preservation.

Uniform/influence final inference sources are now prepared by **CPU946894**;
**CPU946946** validated both actual preserved snapshots through their serving
adapter. They retain original full94, two-view inference and reuse immutable
notebook weights. They have **not** run their T4 qualification/public inference
or received competition scores. New label-final serving helpers also remain
prepared until the actual final model and private preservation exist.
[Exact scoring preparation](../initiative/audits/matched-final-scoring-preparation-20261017-v1/README.md).

The stronger public reference **139199164 v2** remains Queued on its actual
logs page; API status is absent. It is unqualified and unscored. Its separate
interactive run passed, which is not a hidden result. Retain R6 Dataset v1;
v2 contains different R7 weights under the same filenames. Never repeat Save,
submit source-only v1, or upload the public three-row CSV.

## Other actual outcomes

The CPU32 scout **139205901 v2** is ERROR and observer40709 finished. Its
classifier actually scored all 32 studies without fallbacks in 9,618.9 seconds;
the outer wrapper then failed a final retained-byte/RSS assertion. Resource
acceptance remains unproven; no repeat run was launched. Original private
outputs were preserved, and only aggregate failure/cost evidence was collected.

H100 pilot **942047** remains pending with zero runtime on evc33, where both GPUs
were allocated. Immediate evc40 device-only request947064 could not allocate;
no new CUDA result exists. Healthy Kaggle training remains untouched.

## Resources and handoff

Authenticated competition allowance at **13:50 UTC**: **5 used, 0 available**. Refresh it
before any entry. Single canonical ledger/claim remain on Newton; credentials
stay in ignored0600.env/RAM/stdin, and large artifacts stay private.

The new serving controls use 4 MiB of the existing node33 future allowance,
leaving **4,128,768 other future bytes** within the unchanged 32 MiB envelope.
Preserve the unpaid part of that 4 MiB allocation as well as other liabilities;
deduct paid bytes once. Original 20 GiB and 5,000-file reserves remain.

Source `01bda353` is privately mirrored at
**HF6795cebc6623fb5fab9c3d6ffde3dc2fbd4303c2**, all 43 objects verified. Do not
repeat it. [Preservation receipt](../initiative/audits/hf-097-label-full-campaign-20261017-v1/README.md).

Read the [research brief](research-brief.md), [execution handoff](codex-continuation.md)
and [research-agent handoff](research-agent-handoff.md). Accepted masks/folds and
protected gold/confirmation remain unchanged. Released encoders saw internal
studies; head CV on their features is not independent whole-model OOF. Negative
results and original evidence remain in the research archive.
