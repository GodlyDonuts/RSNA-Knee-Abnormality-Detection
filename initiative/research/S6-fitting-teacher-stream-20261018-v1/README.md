# Complete fitting-cohort complementary MRI teacher stream

The objective is all3887 original fitting studies, not another scout. The original frozen S6 anatomy/render/classifier code and weights are reused. New original-source acquisition stays on evc33 and uses the already verified complete private source index; there is no permanent501GB image copy, new label source or optimizer here.

## Actual evidence and execution choice

The original CPU bridge961406 exactly reproduced all248 images/meta tensors from the smallest-source retained study. Original H100960986 accepted the unchanged classifier FP32/FP16 on eight cached studies. Original CPU32 completed finite predictions, recovered960928 despite its final resource assertion failure. Its numerical/input proofs are valid; its resource failure remains a failure. Those jobs must not be repeated.

Use FP32. Reuse existing H100 FP32 logits for its eight cases and original CPU FP32 logits for the other24; record their distinct provenance and small accepted hardware discrepancies, rather than inventing independent OOF accuracy. The other3855 fitting studies get new source/render/H100 inference once. Raw logits drive teacher targets; rounded six-decimal CSV values do not.

A persistent original CPU renderer is safe to reuse across studies: actual source inspection961680 shows StudySlices keeps only the resolver, clears its cache each build, and retains no pixel arrays on self. The original CStudies already supports repeated studies. The worker explicitly releases completed local tensors and trims libc heap after saving the private result.

## Pipeline and failure semantics

Four CPU workers (two Torch threads each) acquire/hash-check original DICOMs with six bounded HTTP streams each, verify every header identity, enforce source1GiB/native/window/heap bounds, and request a weighted decoder-memory lease before rendering. The unchanged anatomy/window renderer returns private uncompressed MRI tensors. One H100 runs the original main in FP32. An observational root-forward hook records exactly twelve finite raw logits without replacing output; the original main's dumped logits must equal the durable records exactly at phase end. Class/constructor/forward mathematics are untouched.

All source buffers stay in at most four16GiB slots, aggregate64GiB RAM scratch; decoder heap broker80GiB; the GPU phase requests10CPU/224GiB/oneH100. CPU-only first-four production prefill requests9CPU/160GiB/zeroGPU. The original2-GPU physical evc33 is permitted. Both H100s were allocated at the latest scheduler read; prefilling actual new studies makes progress without reserving an idle GPU. No4-/8-GPU nodes.

The CPU prefill is part of the full stream and its tensors are reused by the GPU consumer. It is not a disease-model pilot or a scored result. GPU phases use four-hour allocations, with the original inference deadline at3h20m. Completed predictions are fsynced once to a capped6MiB journal; ready prefetched tensors remain for the next phase. A distinct next phase may continue only after actual successful Slurm terminal/receipt and fresh resource review. Failed/uncertain phases never retry automatically. Malformed/partial journal lines refuse replay. An atomic execution lock prevents simultaneous prefix writers.

The original stream8MiB control/future-small-output envelope funds source code, journal and final teacher NPZ; old S6 control16MiB is unchanged. New64GiB transient RAM, broker and host allocations are separately admitted under fresh global reserves. Original datasets/models/index and unrelated healthy jobs are never deleted or restarted. Only newly owned source/render scratch may be reclaimed after its source/tensor proof and finite logit are durably recorded. Per-study details stay private.

## Acceptance and next experiment

Source/contract scaffolding passes five focused tests, including an actual Torch forward observer that cannot replace the model output even if its callback returns a tensor. This does not establish real full-stream execution or accuracy. CPU staging, actual first-four prefill, GPU context/progress and full cohort completion must be reported from their original receipts.

Only all3887 unique identities, finite3887x12 raw logits/probabilities, explicit original source/model/runtime/geometry/label ordering and a successful final phase admit separate equal-rank target construction. No partial empirical ranks, protected-label reads, target-specific weights or claims of gold labels. A later matched full-encoder contrast judges usefulness; there is no new clinical/Kaggle gain yet. Existing paired/LR fitting/scoring owners remain unchanged.
