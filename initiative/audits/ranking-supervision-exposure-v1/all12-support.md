# Actual fitting-only comparison opportunities

| Target | Observed fitting + / − | Eligible effective updates, seed20260922 | Seed20261004 |
| --- | ---: | ---: | ---: |
| ACL | 171 / 1682 | 390/843 (46.26%) | 389/843 (46.14%) |
| MCL | 5 / 1816 | 15/843 (1.78%) | 15/843 (1.78%) |
| Medial Meniscus | 832 / 1082 | 826/843 (97.98%) | 820/843 (97.27%) |
| Lateral Meniscus | 297 / 1565 | 568/843 (67.38%) | 591/843 (70.11%) |
| Medial OA | 46 / 911 | 128/843 (15.18%) | 128/843 (15.18%) |
| Lateral OA | 18 / 976 | 53/843 (6.29%) | 51/843 (6.05%) |
| PF OA | 38 / 804 | 109/843 (12.93%) | 99/843 (11.74%) |
| Effusion | 16 / 1257 | 47/843 (5.58%) | 47/843 (5.58%) |
| Synovitis | 243 / 10 | 22/843 (2.61%) | 16/843 (1.90%) |
| Baker's | 6 / 750 | 18/843 (2.14%) | 17/843 (2.02%) |
| Contusion | 121 / 671 | 288/843 (34.16%) | 281/843 (33.33%) |
| Fracture | 3 / 440 | 5/843 (0.59%) | 7/843 (0.83%) |

Each forward microbatch contains one real study, so within-microbatch
same-target positive/negative pairs are **zero for all twelve targets**.
The effective batch combines eight such forwards. Counts above span three
epochs and exclude masked padding and same-leakage-component comparisons.
The Effusion row uses the original control16 positives; the reviewed31-cell
candidate is not replayed here. All other original targets remain unchanged.
No model, MRI, optimizer, ranking loss, ordered-concept annotation or protected
evaluation was used. These counts cannot establish an accuracy root cause.
