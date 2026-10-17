# Matched teacher-guided observations for full-encoder learning

Question: can16 selected windows deliver more useful full-model updates than
16uniform observations, using one already completed full94 teacher cache?
The predeclared label-free screen on all3887 fitting studies reduced teacher
readout approximation MSE45.6%, every target improved. That is teacher fidelity,
not disease accuracy. [Actual screen](../../../audits/licensed-MIL-sampling-20261016-v1/README.md).

This connects [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html),
[importance sampling for deep learning](https://proceedings.mlr.press/v80/katharopoulos18a.html)
and [attention sampling of large images](https://proceedings.mlr.press/v97/katharopoulos19a.html).
Those papers motivate mechanisms; they do not prove this MRI variant or its gain.
This is an adaptation, not a claim of a new invention.

Both roles initialize the same complete licensed73,136,096-parameter CoAtNet.
Both keep the accepted3887fitting studies, source/teacher targets, original96x384
pixels,94center observer, effective8batch, encoder/headLR3e-6/3e-5, frozenBNbuffers,
AdamW.02,10%warmup/cosinefinal1%,clip1 and FP16/scaler1024/FP32loss. No protected
cohort is read. Fixed3epochs/1458updates per role are the proposed full endpoint;
no full fit is admitted or dispatched by this document.

Each study presents16IID windows WITH replacement. A role-independent case/epoch
seed supplies identical uniform variates and anatomical view to both arms. The
control uses uniform q=1/94. The candidate uses the fixed screened proposal:
25%uniform plus75%normalized teacher influence, averaged over both views and
normalized per target before averaging12targets. Duplicates remain actual draws.
Subtract centered log(q) from attention scores and normalize over sampled draws.
Both arms use FP32 attention softmax, cast to feature dtype for the same pooled
readout. All encoder and head parameters learn;8-window checkpoint recomputation
bounds activation memory. Full94 inference delegates to the original parent.

The original unique-window reader stays byte-identical. A new wrapper reads the
unique selected windows and restores draw order/duplicates, preserving original
pixels, targets and view-specific full-teacher logits. Role and complete proposal
SHA bind optimizer checkpoints; changes cannot silently resume an old fit.

Current whole94 training remains healthy and unchanged. It is a contextual
reference, not a strict sampling-only causal control: it uses more windows/compute
and a different head softmax precision path. The two new16-window roles are the
matched scientific contrast. Measure actual throughput/memory and full learning;
94/16 is not an observed speedup. Self-normalized ratios are biased at finite bag
size, frozen influence can become stale, and a wrong teacher can amplify mistakes.
A fidelity gain need not improve disease classification.

First require actual real-MRI twoT4 full/stop1/resume recovery for BOTH roles:
16fitting+16disjoint engineering cases,2epochs/4updates; exact complete model/
AdamW/LR/cursors/allrankRNG/scaler/history/epoch snapshots and identical16x12
original full94 two-view predictions. Compare initial full learned graph and all
RNG between roles; both must actually learn distinct tensors. Tiny predictions
are engineering-only and cannot be submitted. Reuse original private32 pixels,
features, teacher and source labels; do not acquire or decode raw MRI again.

After actual acceptance, admit a fixed full3887proposal table and original pixel/
target bindings, fresh all-liability storage and GPU VM budgets, then run both
fixed-final full roles. Preserve all immutable epochs once. Actual competition
comparisons need their exact learned-state/T4/public-MRI/runtime acceptance and
fresh allowance; the current context/field/regional/E1/E3 plan remains unchanged.
No independent localOOF can be claimed for derivatives of this exposed parent.
