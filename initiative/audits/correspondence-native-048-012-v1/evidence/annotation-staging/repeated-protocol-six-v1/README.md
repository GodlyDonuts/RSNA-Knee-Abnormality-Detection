# Six repeated-protocol correspondence candidates

The complete original report text for all six cases was personally read and frozen in `source-only-preread.json` before candidate-specific technical materials were opened. The assignment had exposed coarse series/side/repetition flags; that order is disclosed. No clinical review annotations or official target values were accessed.

All six report strings are identical to the preserved current authenticated official CSV and archived source CSV. The complete technical check covers 56 series, 112 lexical first/last source headers and 1,657 inventory slices. Selected endpoint path/SHA/size/UID membership matches the corrected inventory; every cache metadata JSON matches its frozen recovery lineage and complete source SOP/hash membership. All 112 endpoint positions, orientations, spacing and dimensions exactly match their source-specific cached geometry. There were no native or cached pixel decodes, network requests or original file changes.

| Native review priority | Case | Series / slices | Raw bytes | Correspondence assessment |
|---|---|---:|---:|---|
| 1 | 075:14 | 11 / 337 | 274,262,744 | Unsided short source; separate blank and L-tagged complete acquisition sets. Native review has highest value for resolving unexpected two-location content. |
| 2 | 048:12 | 10 / 359 | 302,931,850 | Explicit integrated bilateral report; two complete five-series sets with all tags blank. |
| 3 | 086:13 | 10 / 300 | 238,906,448 | Explicit integrated bilateral report; L-tagged and blank complete five-series sets. |
| Defer | 001:18 | 9 / 282 | 83,035,438 | All L-tagged; multiple exact-geometry repeat families in one protocol region. |
| Defer | 021:03 | 8 / 231 | 179,783,174 | One unsided report; repeats and narrower/adjusted acquisition within the same recorded region. |
| Defer | 064:14 | 8 / 148 | 65,747,012 | One unsided report; repeats and modestly changed positioning, without a separate complete shifted protocol. |

All repeated series have distinct native pixel fingerprints: none is an exact pixel-overlap candidate in preserved metadata. Repeated geometry alone is insufficient for deleting a series. Every study has one consistent endpoint PatientID hash; each series has a distinct anonymized frame hash. Neither property proves patient identity, shared global registration, side or source correspondence. Blank side fields remain unknown. Official plane fields match the dominant IOP axis for all 56 series, but this is not native anatomical validation.

`source-header-assessment-and-native-plan.json` contains the full individual decisions, exact acquisition-set UIDs and proposed bounded native workflow. The first three cases merit one-study-at-a-time full-source native checks, a 20 GiB disk reserve, at most two requests, and independent visual freezes before exchanging conclusions. Native acquisition is planned, not performed. Root coordinates escalation. The lower-priority three need no automatic full download from present evidence.

`collect_technical.py` rebuilds immutable source/header/cache evidence in a new output directory after verifying the frozen preread and original bindings. `compare_geometry.py` separately checks exact endpoint geometry against each corresponding cache SOP and compares repeated protocol metadata. Both scripts are local read-only except their new staging outputs. Existing endpoint coverage remains historical prefix-header evidence, and all-slice cached geometry remains historical strict-decoder evidence; neither is represented as fresh all-slice header or native verification.
