# Actual cached-MRI screen: parent-delta pooling

CPU954924 completed the fixed3887-study,64draw,all12-target screen in44seconds.
The stable paired-logit residual improved approximation to simulated dense
student pooling at every predeclared drift scale. This is **not actual learned
student behavior, clinical accuracy, held-out evidence or a Kaggle gain**.

| Simulated drift scale | Stable MSE / ordinary sampled MSE | Raw-moment invalid draws / 248768 |
| --- | ---: | ---: |
| 0 | 1.39983e-30 | 0 |
| 0.05 | 0.00195118 | 0 |
| 0.2 | 0.0308044 | 4 |
| 0.5 | 0.177514 | 385 |
| 1 | 0.525858 | 3156 |

Stable-method error uses **all** draws; the raw-moment error in actual.json is
explicitly conditional on positive denominator draws. Reject unguarded raw-moment
training: already4 invalid draws at0.2 and3156at1.0. No tail was silently clipped.

At zero drift the stable forward agrees within4.45e-16 in this FP64 moment
calculation. The raw formula has exact initial expected gradients in enumerated
synthetic checks, but the stable formula retains sampled-gradient bias. Neither
property establishes exact equality for the real stochastic FP16 trainer.

The practical stable candidate is a training-only correction:
`corrected_logit = student_sampled_logit + parent_full_logit - parent_same_bag_logit`.
Full94serving cancels the two parent terms, retaining the original single student
and unchanged inference cost. Because each study/epoch bag and mirror are fixed
by the existing sampler, deterministic teacher corrections could be computed
once from accepted cached parent features. Three3887x12FP32tables are559728bytes
**before** schedules/provenance/packaging: arithmetic, not a new storage admission.

Before any trainer/GPU dispatch: bind actual scheduled windows/mirror, parent
readout/cache/teacher precision, source order and correction bytes. Review active
dropout/stochastic-depth and FP16 differences; fixed-BN is not full determinism.
Require actual real-MRI full/stop/resume evidence and a matched fixed-final
comparison. No extra source labels, coverage, inference or simultaneous sampling
change. Current family/finding fits and their evaluation owners stay unchanged.
