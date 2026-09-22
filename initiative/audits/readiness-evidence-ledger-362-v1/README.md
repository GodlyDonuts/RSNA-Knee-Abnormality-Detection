This is a reusable evidence ledger for registry362, not a clinical decision or label export. It preserves all 4,407 original reports / 52,884 target cells, the frozen rubric and canonical source fingerprint. The stable cutoff includes wording 331, cartilage 260, bone 000–019 (1,000 reports), root cartilage 000–005, root wording 000/001 source and dispositions, and root wording 002 full 46 source readings. Root 002's later comparison/adjudication and bone 020 remain excluded. The completed bone 007 canonical-context normalization is bound separately without changing its original proposal.

The ledger preserves 187,755 records across 416 stages: 52,884 original first-pass judgments; 52,884 original second-reader prereads; 52,884 final reviewed records; 13,504 supplemental frozen source judgments; 11,104 complete supplemental postfreeze opinions plus 493 explicit difference-only records; 2,277 root source judgments; and 1,725 root postfreeze dispositions. These are records of earlier work, not 187,755 independent reviews. Repeated stages by one author do not add an independent reader. Missing original session identifiers remain null in 2,076 records; their actual author IDs and exact artifact selectors remain present. Five stage headers lack explicit sessions, so the ledger does not invent them.

Use these files:

- `ledger-final/cell-ledger.jsonl`: one UID+target row per canonical cell, with original reviewed state, actual362 state, current correction references, related evidence IDs, inventory memberships and non-gold report-mask obligations.
- `ledger-final/evidence-records.jsonl`: exact recorded states/quotations, actual declared author/session, source hash, stage and original JSON pointer or line. Original prereads, later reviews, supplemental freezes and reconsiderations remain distinct. Early wording 000–003 postfreeze difference records keep their multiple original fields; no final quotation is synthesized.
- `ledger-final/artifact-index.json`, `stage-index.json`, `source-report-index.jsonl`: exact file hashes, source/index/UID identities and actual target scope. No original full-report archive is copied.
- `ledger-final/scope-memberships.jsonl`: exact cell membership for every stage and inventory/attention scope.
- `worklists/exact-worklist-memberships.jsonl`:46,999 explicit membership rows across37 scopes, adding original historical cartilage noncandidate-negative and masked-candidate partitions. Each row links supplemental/root source and postfreeze IDs, current accepted case and mask requirement. A source/postfreeze record from another topic is a reuse opportunity requiring the appropriate context comparison, not automatic gate clearance.
- `ledger-final/summary.json` and `worklists/worklist-summary.json`: counts and their interpretation; `replay-validation.json` and `guard-test-validation.json`: validation receipts.

The key bounded reuse counts are:

| Fixed scope | Cells / reports | Matching supplemental pair | Remaining targeted disposition without that pair |
|---|---:|---:|---:|
| Cartilage residual negative candidates |2,407 /1,204|408 cells|1,999 cells /975 reports|
| Historical cartilage negative cells outside proxies |3,002 /1,188|287 cells|2,715 cells /1,055 reports|
| Bone priority |2,304 /1,549|1,539 cells|765 cells /536 reports|
| Focused qualified wording |539 candidate cells /331 reports|539 cells|0 in this narrow candidate subset|

Across all targets, 6,454 cells have two recorded supplemental source authors in the same actual target family. All unmatched cells still have the original two-pass evidence; no fresh reread requirement is inferred. Root attention scope is 255 bone reports and 216 wording 002–006 reports. Within it, 20 bone cells already link to root source judgments; wording attention links 591 root source judgments, including 552 from completed root 002. Target-level links prevent a 3-target OA reading from becoming a 12-target reading by inference.

Current 362 correction objects replay exactly. The mask ledger separately records 18 enforced whole-report correspondence exclusions and 24 unresolved reports requiring all 12 non-gold report-loss masks to remain zero; all 24 remain clinically masked at this cutoff. Official gold values and final gold masks are not accessed or assigned. Three semantic gates remain pending.

Reproduce with a new output directory, from the workspace root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python work/annotation-staging/readiness-evidence-ledger-362-v1/build_ledger.py --workspace . --output <new-ledger-directory>
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python work/annotation-staging/readiness-evidence-ledger-362-v1/verify_ledger.py --workspace . --ledger <new-ledger-directory> --output <new-replay-receipt.json>
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python work/annotation-staging/readiness-evidence-ledger-362-v1/derive_worklists.py --workspace . --ledger <new-ledger-directory> --output <new-worklist-directory>
```

The separate mechanical replay checked 978 artifact hashes, all 187,755 record quotes/source identities, 416 reconstructed target scopes, all 52,884 cell links, all 362 corrections, exact membership counts and mask predicates. Eleven guard tests passed. The builder refuses existing outputs, source scope/hash mismatches, invalid quotations, absent authors, changed registry and insufficient disk reserve. Missing session IDs are not invented. No repository software tests or image downloads/copies were required. The preliminary unhanded-off duplicate ledger build was removed; only final data and necessary code/validation remain. No R files, source artifacts, registry entries, QA statuses or clinical judgments changed.
