# Teacher anchoring: an actual fitting-only negative

CPU946541 measured all3,887fixed fitting studies, both unchanged full94teacher
views and all12targets. No new MRI decoding, encoder, optimizer, GPU fit, Gold,
confirmation or clinical report was used. No live recipe changed.

The hypothesis was that the75%parent anchor necessarily prevents useful source-
label ranking corrections. In the original loss,
`q = alpha * source_label + (1 - alpha) * sigmoid(parent_logit)`.
CPUautograd on32actual fitting rows verified that the derivative is exactly
`(sigmoid(student_logit) - q)/(N*12)` for the predeclaredalpha grid
0.25/0.5/0.75/1.0; maximum observed gradient error was0.0.
For source-positivei/negativej with teacher-opposed ranking, the pointwise target
reverses their ordering when
`alpha > (p_j-p_i) / ((y_i-y_j) + (p_j-p_i))`.

Before reading outcomes, strong source labels were defined as>=0.9and<=0.1.
The table is **agreement with those weak source labels**, not clinical error,
held-out AUC or the learned sampled-student trajectory. The parent saw these
fitting studies. The unconstrained pointwise optimum is not a hard constraint
on a shared CNN's dynamics/generalization. Pair counts are heavily dependent.

| Target | Plain teacher-opposed / strong-source pairs | Reordered byalpha0.25 | Mirrored opposed / pairs | Reordered byalpha0.25 |
|---|---:|---:|---:|---:|
| ACL | 1865 / 950798 | 98.82% | 2735 / 950798 | 98.24% |
| MCL | 393 / 127080 | 99.75% | 579 / 127080 | 100.00% |
| Medial Meniscus | 32406 / 2141972 | 86.83% | 41214 / 2141972 | 85.96% |
| Lateral Meniscus | 12619 / 1175994 | 91.32% | 18895 / 1175994 | 89.89% |
| Medial OA | 13058 / 999648 | 92.05% | 15961 / 999648 | 91.01% |
| Lateral OA | 11514 / 519196 | 93.96% | 13884 / 519196 | 93.27% |
| PF OA | 14399 / 1044490 | 94.60% | 16813 / 1044490 | 94.21% |
| Effusion | 12547 / 1074393 | 92.92% | 14506 / 1074393 | 92.58% |
| Synovitis | 645 / 46230 | 98.14% | 718 / 46230 | 97.63% |
| Baker's | 3268 / 473909 | 90.09% | 4548 / 473909 | 88.68% |
| Contusion | 7399 / 847530 | 91.15% | 10181 / 847530 | 89.96% |
| Fracture | 595 / 69531 | 100.00% | 774 / 69531 | 98.97% |

**The strong blocker hypothesis is not supported by this fitting screen.**
Alpha0.25already reorders86.83–100%of plain-view teacher-opposed strong-source
pairs. Fracture has only49strong-source positive and1419negative studies;
all595plain-view opposed pairs reorder at0.25. The low overall old/new Fracture
rank correlation therefore does not by itself establish a large high-confidence
error barrier. At0.5, only2of32,406plain-view opposed MedialMeniscus pairs remain
inverted; other targets' opposed strong-source pairs all reorder. These are
analytical target preferences, not achieved ranking or an accuracy improvement.

Preserve the new label-only contrast unchanged. Do not remove its anchor or
pick a new mixing weight from this screen. A future separately matched anchor-
weight experiment can still be justified by actual model/generalization evidence;
this result neither proves weak labels correct nor rules out unseen teacher bias,
intermediate-confidence effects or insufficient optimization. No data, recipe,
checkpoint or competition submission changed. All24target-view outputs and
threshold quantiles, including limits/undefined cases, are retained inactual.json.
