# Multimodal evidence teacher: one bounded, image-first experiment

The owner suggested coupling our MRI classifier to a multimodal LLM and accepted
the official MedGemma terms. Authenticated access to the pinned official
`google/medgemma-1.5-4b-it` revision91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b
actually returned200 after the initial403. The original public manifest binds
15files/8639646105bytes and original LFS digests. No model download or accuracy
result is implied by access. Use safetensors and local-only loading with no remote
repository code. Derived evidence and MRI packets stay private.

This is a hypothesis about **complementary localized evidence**, not an assumption
that a language model knows better. Google's model card reports volumetric MRI
support and64.7% macro accuracy on an internal balanced ten-condition MRI task;
that is not RSNA AUC or independent knee performance. Its source training includes
licensed internal MRI, so it is not an independently held-out/OOF teacher by
assertion. [Official model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card).

## First execution and decision

1. Acquire the official pinned model once under a new9GiB private node-local
   envelope, plus128MiB dependencies and256MiB private input/output. Preserve all
   originals, current32MiB metadata and20GiB physical/shared/5000file reserves;
   historical unstarted bulk work remains deferred. Acquisition uses one CPU,
   zero GPUs and credentials only in stdin. Partial/uncertain transfers stop for
   original-state reconciliation rather than automatic replay.
2. Reuse complete original MRI packets already assembled for Daenerys. Freeze
   all image hashes and panel-to-original-slice mappings. Technical pilot packets
   are fitting data, not an accuracy holdout. No accepted labels, reports,
   classifier scores or previous reviews enter the initial prompt. Request all
   twelve targets with present/absent/uncertain/not-visible distinctions and
   concrete sheet/panel evidence. Record every model preprocessing resize.
3. Measure actual H100 latency, peak memory, structured-output validity, complete
   input coverage, and whether references resolve to supplied MRI. A plausible
   narrative or valid JSON is not a correct finding. Compare report-informed
   output only after freezing the image-only response. Treat contradictions and
   insufficient evidence as uncertainty, not a majority vote.
4. If the model can ground useful evidence, prioritize classifier-disagreement
   studies with balanced target coverage. Derive candidate slice/sequence
   supervision separately from disease labels; require adjacent-slice and
   cross-sequence support where applicable. Test evidence-guided training against
   the identical classifier without that evidence on untouched, component-split
   evaluation. Existing all3887-fitting predictions are not a clean evaluation.
   No protected gold/confirmation use or accepted-label overwrite.

The primary deliverable remains a fast MRI-only classifier. A direct VLM hybrid
also needs an actual incremental accuracy and offline2T4/9h runtime comparison
before deployment. Distillation avoids paying generative-inference cost for
every hidden study, but is not automatically more accurate. Continue the actual
complete S6 teacher/consensus experiment and higher-LR Kaggle comparison in
parallel; do not replace them with another indefinite annotation project.
