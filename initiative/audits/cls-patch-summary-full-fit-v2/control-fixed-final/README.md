# Actual fixed-final ten-epoch control result

Original job878853 finished its **control arm**, then began the CLS-plus-mean-patch candidate. The joint GPU job remains running; the candidate has no final outcome yet. The completed control is the current rank32/native384/plain-attention model with the zero patch-summary branch. It trained ten fixed epochs/2810 contiguous finite optimizer updates on2247 fitting studies and evaluated1156 disjoint weak-label development studies.

The original fixed-final result is **macro AUC0.8786705965399949**, all12 targets scored. Independent replay from the exact251975-byte prediction CSV and only the matching accepted development rows reproduces every target AUC within1e-12. Source0183ef3, actual model/result binding, exact roster digest, original role return0, empty stderr, prediction SHA and privately verified epoch10 checkpoint SHA/revision are bound in [collection.json](collection.json) and [metric-replay.json](metric-replay.json). No model weights, MRI pixels or protected gold/confirmation rows were collected.

| Target | AUC | Observed positives | Observed negatives |
| --- | ---: | ---: | ---: |
| ACL | 0.882340 | 85 | 879 |
| MCL | 0.706660 | 2 | 946 |
| Medial Meniscus | 0.873747 | 412 | 580 |
| Lateral Meniscus | 0.754903 | 149 | 821 |
| Medial OA | 0.981659 | 23 | 499 |
| Lateral OA | 0.817849 | 9 | 531 |
| PF OA | 0.864991 | 19 | 445 |
| Effusion | 0.938732 | 9 | 671 |
| Synovitis | 0.980165 | 121 | 5 |
| Baker's | 0.910122 | 3 | 382 |
| Contusion | 0.925545 | 61 | 371 |
| Fracture | 0.907336 | 2 | 259 |

This is an **internal report-supervision AUC**, not a Kaggle score, clinical diagnostic accuracy, paired candidate gain or forecast. Rare targets have very sparse observed positives/negatives; the detailed support matters. Do not select the candidate's epoch or modify its recipe using this control result. The CLS-plus-mean-patch question requires its original fixed-final candidate and paired whole-leakage-component comparison on these same1156 studies.

The old strict uniform internal0.8606 is historical context with recipe/source/precision/seed differences, not a controlled causal attribution for the new0.8787. Best actual solo Kaggle remains0.804. The newly completed model must receive exact source/runtime/weight acceptance before any later competition deployment; no submission was made here.

All ten control epochs are privately preserved/download-hash verified. Final epoch10 revision0637080cdd64bcf3b9beed10ee9e813fad2d9b70; checkpointSHA2465648917b579eaa7960d1c54cc3482975d4b11d8a0ec4d1684cbd9a2ac97ec. Original full joint source/outputs/jobs remain unchanged. The first local replay failed before label decoding due tuple/list schema handling; corrected read-only replay uses explicit target-column lists, same original data and same metric. No training run was repeated or failed.
