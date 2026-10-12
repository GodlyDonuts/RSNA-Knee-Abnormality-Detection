# Fitting-only supervision source differences

Exact2247 fitting studies only; parquet row filtering excludes evaluation/protected studies. OverlaySHAc8fd2e86 binds admitted source/release/folds. No model/config/accepted label changes. Source agreement is not clinical truth or accuracy;0.5 is descriptive only, not selected training policy.

| Target | Accepted positives / observed | Public ≥0.5 disagreements on jointly observed | Newly observed public cells |
| --- | ---: | ---: | ---: |
| ACL | 171 / 1853 | 107 / 1853 | 394 |
| MCL | 5 / 1821 | 187 / 1821 | 426 |
| Medial Meniscus | 832 / 1914 | 7 / 1914 | 333 |
| Lateral Meniscus | 297 / 1862 | 3 / 1862 | 385 |
| Medial OA | 46 / 957 | 109 / 957 | 1290 |
| Lateral OA | 18 / 994 | 101 / 994 | 1253 |
| PF OA | 38 / 842 | 127 / 842 | 1405 |
| Effusion | 16 / 1273 | 727 / 1273 | 974 |
| Synovitis | 243 / 253 | 0 / 253 | 1994 |
| Baker's | 6 / 756 | 252 / 756 | 1491 |
| Contusion | 121 / 792 | 3 / 792 | 1455 |
| Fracture | 3 / 443 | 1 / 443 | 1804 |

The important difference is concentrated in severity-sensitive targets and masked coverage. Accepted MCL has5 positives, Effusion16, Baker6 and Fracture3; Synovitis has243 positives but only10 observed negatives. This is a sparse and selectively observed learning problem even with2247 studies. Public probabilities preserve accepted positive evidence strongly (mean about.95–.99 across targets) but broaden supervision and disagree strongly for Effusion/Baker/MCL/cartilage. Meniscus and contusion joint binary agreement is much closer.

The frozen v2 rubric explicitly requires high-grade MCL, moderate/large effusion/Baker and cartilage severity/extent. Thus broader source probabilities can represent different target semantics or better report coverage; this aggregate audit cannot distinguish those explanations. No gold/confirmation labels were read. Existing full public-versus-accepted fixed10 contrast remains unchanged; no blended target/source choices made from these statistics. Need actual matched final outcomes before promotion, and clinical/source review before claiming label correctness.
