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

## Actual admission and earlier-slot execution

CPU968270 downloaded/verified all15official files/8639646105bytes, including
original LFS hashes, on33. Runtime Torch2.14.0+cu130/Transformers5.6.1 imports the
actual Gemma3 conditional-generation class; no new runtime overlay was needed.
CPU968339 admitted all195native slice crops/5series/34027570bytes from the exact
existing displays, with no image resizing, report, accepted labels or model
predictions. Every source-sheet and slice PNG digest matched; source displays and
DICOM originals remain unchanged. The model processor's own resizing is disclosed.

Original one-H100 pilot968352 had a late estimated opening on33. Actual same-runtime
CPU968374 on26 and first direct private relay968399/968400 independently verified
all214objects/8673759365bytes. The first26copy has a distinct10GiB envelope above
20GiB physical/shared/5000file reserves; no source network reacquisition or original
deletion occurred. Slot probing predicted an earlier30minute opening. Server-side
pending-only cancellation then confirmed968352 CANCELLED/00:00:00; distinct pilot
968429 is submitted on26,1H100/2CPU/64GiB/30minutes. It changes only host/runtime
location; original model/prompt/image processing/pilot source remain identical.
No MRI inference or accuracy result is inferred from admission/dispatch. Inspect
actual Slurm and private terminal/control files before any next action.

## Actual allocation failures and source-node continuation

The26 batch968429 and scheduler-step968456 both FAILED1:0 before model load/MRI
with CUDA101(invalid ordinal), after15/13seconds. Strict exact-assigned-index probe
968494 failed its NVML index0 query(exit6) before CUDA/model/MRI. This is unresolved
allocated-device visibility, not a demonstrated defective physical GPU or a model
failure. No unassigned device was selected; cgroups/visibility were not widened.
All admitted weights/images remain private and unchanged on both nodes.

Distinct source-node assigned-step968562 is now submitted on33,1H100/2CPU/64GiB/
30minutes, using original pilot source and previously accepted scheduler-step
mechanism. It remains pending at latest observation. Do not retry26 blindly, repeat
a cancelled submission, or claim blank-control/evidence/clinical success from
staging. The original sources and all failures/dispatch receipts are preserved.
