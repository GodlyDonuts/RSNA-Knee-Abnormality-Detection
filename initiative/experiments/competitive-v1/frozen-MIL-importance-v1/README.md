# Label-free Monte Carlo fidelity screen for MRI MIL

A weak study target can be presented with an uninformative small image bag.
Before paying for another full CNN fit, ask whether a frozen strong teacher can
guide sampling toward windows that stabilize its whole-study readout. The idea
connects Monte Carlo importance sampling to weakly supervised MRI bags. This is
not a novelty claim, clinical localization or proof that the teacher is correct.

The exact public0.949 parent has a finding-specific softmax-weighted feature
mean and linear class readout. Cache its two-view94-window features once; extract
only its279832-parameter MIL head with no MRI/labels/GPU. For each view/target,
write attention a, instance score v, and full mean mu=sum(a*v). Define influence
magnitude a*abs(v-mu), average views, normalize each of12targets separately, then
average targets. Proposal q mixes75%of this distribution with25%uniform coverage.
All coefficients are fixed before the real32-study screen.

Draw16windows IID with replacement,128repeats, seed20261016+case ordinal. Use the
same uniform random numbers for the uniform and adaptive proposal. Correct each
selected attention weight by1/q, equivalent to subtracting log(q) from its logit.
Duplicates are valid draws. Normalize within the selected bag, evaluate both
views and average probabilities. The self-normalized finite-bag ratio can be
biased; full support does not guarantee improvement. Report unique-window counts.

Primary engineering measure: mean squared deviation from this teacher's full
FP64-moment readout over32original fitting-study features and all12targets.
Predeclared go-to-full-cohort screen only if adaptive MSE is at most80%of uniform
MSE and no target with uniform MSE>1e-12 doubles its error. Secondary: paired
teacher-order disagreement across the32cases/128draws and per-target errors.
Compare the FP32CPU teacher head and FP64decomposition with the original cached
FP16 probabilities; report numerical gaps rather than silently equating them.

These32cases are the already fixed lexical fitting engineering subset, not a
representative validation cohort. No report, disease label, Gold, confirmation,
hidden-test record, AUC or model fitting is read or run. Passing justifies only
a label-free full3887feature repeat. It cannot promote a clinical recipe or
competition model. A later matched CNN training contrast would need identical
parent initialization/data/labels/compute and actual GPU recovery; none is staged
or dispatched here. Attention/influence is model behavior, not verified anatomy.

Ten synthetic tests pass: moment reconstruction, log-proposal equivalence with
duplicates, constant-teacher fallback, full support, deterministic RNG and a toy
sparse-vote variance reduction. Actual source/MRI results are not implied. The
licensed model/source provenance stays explicit; all case-level arrays remain
private on the owning node/Kaggle. No new CNN pass or competition slot is used.
