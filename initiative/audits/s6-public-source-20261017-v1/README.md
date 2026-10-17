# Actual source-only S6 admission

At October 17 03:02–03:03 UTC, the public Kaggle pull endpoint returned
`nartaa/rsna-knee-s6fall-solo`, **138285808 v2**, source **193,159 bytes**, SHA
`6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe`.
The original source is retained locally without execution. This audit publishes
its identity and static contract, not a copied external source tree.

Dataset metadata identifies **12589750 v3**, public,
`nartaa/rsna-knee-s6fall-weights`, 200,500,299 bytes. Two named JSON manifests
were actually downloaded (1,128 and 52,962 bytes) with a one-MiB per-file cap;
no model or pixel bytes were acquired. Requested Dataset version 3 and original
content hashes are in [the receipt](manifest-receipt.json).

The checkpoint manifest pins `E_prior_s42_fall.pt` SHA
`0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12`:
ConvNeXt-Small, spatial/window attention, 4,263 source training studies, 16 epochs,
64 training windows, full-FOV and 100-mm anatomy crops. The author's reported
0.950 is **not yet reproduced by this project**. The code actually uses W_A=0,
W_B=1, despite an inherited blend description; unreadable S6 cases can still use
CoAtNet. The current main routine unconditionally executes both complete model
pipelines; W_A=0 does not save CoAtNet inference time. Do not classify the output
or compute budget from the inherited docstring alone.

Weight permission is CC BY-NC 4.0 with nartaa/Heliosli attribution; third-party
components retain their licences. The Dataset metadata preserves that statement.
The anatomy checkpoint SHA is
`5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e`,
from the attached public Heliosli Dataset. All actual model/source/rights and
runtime/output checks remain necessary before our reproduction. No new Dataset,
notebook push, GPU run, teacher predictions or competition entry occurred.

Primary sources: [author's account](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359),
[exact public notebook](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo),
[weight Dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights).
[Assessment and next decision](../../research/owner-deadline-program-20261017-v1/README.md).

Later actual progress: [the original checkpoint is now privately acquired once
and SHA/header verified](../s6-checkpoint-admission-20261017-v1/README.md).
The source-only observations above retain their 03:02–03:03 timestamp. No MRI/GPU
acceptance or new competition score is inferred.
