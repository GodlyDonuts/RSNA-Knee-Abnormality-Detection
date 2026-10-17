# Actual fixed N3 source-domain AUC decomposition

**CPU949671completed** the exact within/between-domain AUC decomposition on all
1,156 original N3E16 development predictions. It reuses the immutable prediction
SHA2ed6f4be… and original verified one-header index, reading only this roster's
weak target/mask fields and manufacturer/model metadata. No model, new MRI
inference, gold/confirmation or fitting executed. Macro AUC reconstructs the
independent original **0.9209253491221031** to <1e-12.

There are12manufacturer domains and42manufacturer/model domains on this roster.
For manufacturer/model, same-domain pairs comprise only2.3–10.0% of each target's
pair mass. This is a case-distribution fact, not evidence that scanner identity
caused the model's AUC. Conditional within/between AUC compare different pairs.

| Target | All AUC | Within manufacturer/model | Between | Positive exams |
|---|---:|---:|---:|---:|
| ACL | .9549 | .9540 | .9550 | 85 |
| Medial meniscus | .9484 | .9458 | .9486 | 412 |
| Lateral meniscus | .8475 | .8491 | .8474 | 149 |
| Lateral OA | .8349 | .7981 | .8366 | 9 |
| Effusion | .9159 | .9042 | .9162 | 9 |

Common ACL/meniscus discrimination does not collapse within these coarse
scanner domains. Lateral-OA/effusion differences merit attention, but each has
only nine positives; MCL/fracture have two. None establishes a causal domain
failure, a source-independent patient split or clinical-label accuracy.
The endpoint is repeatedly inspected development and weak labels. One header
per study also cannot prove per-series scanner consistency or actual site.
No equal-domain metric is substituted for the competition metric.

[All12 targets/both stratifications/exact reconstruction and limitations](actual.json).
Only aggregate matrices' cell counts are retained publicly, no case arrays or
scanner/case mappings. The math was unchanged between the two observers:
original949650computed/reconstructed successfully but stopped before acceptance
because its detailed pair-matrix serialization exceeded2MiB. Distinct v2 removes
that detailed serialization and preserves the original failed intent/subcap.
It passed in949671; never repeat either analysis. No new score or causal claim.

## Original spatial scout observation in the same allocation

CPU949671also observed original H100949626 after its actual PREEMPTED terminal:
26/32complete private feature files,1,912windows,1,069,121,828bytes, no final scout
acceptance. Its frozen original forward checked <=1e-6 base-logit fidelity before
each completed file write. The remaining six must be completed by a distinct
recovery that first verifies/reuses these files, not repeats completed forwards.
