# Proposed seventh-group correspondence extension — not integrated

This proposal extends Git `13c224a7c301f225e9fef56861de7c6b3fa0c10e` from 13 to 16 whole-study report-supervision exclusions. Only `048:09`, `048:38`, and `053:47` are added. Each original root case record explicitly states `required_export_action = exclude_whole_study_report_supervision_preserve_originals_and_gold`. This is an unambiguous existing instruction; no new root adjudication or fabricated boolean is needed. `048:31` has the explicit retain/no-exclusion action and is excluded from this proposal.

## Reviewable files

- `overlay/rsna/correspondence.py`: proposed runtime change only; `production-proposal.patch` is the diff against the pinned production base.
- `overlay/initiative/audits/correspondence-release-dispositions-v4.json`: three source-bound decisions, each twelve false report-loss masks. The registry overlay appends its exact hash while preserving all three prior references.
- `overlay/initiative/audits/correspondence-source-bindings-v4/`: three wrappers, each preserving the original root action string and binding exact source, case selector, root replay, independent visual assessment, root visual freeze, both full-source freezes and joint handoff.
- `overlay/tests/test_correspondence_actions.py`: production-ready dedicated tests. `run_tests.py` loads the proposed runtime without changing the live checkout; 37 new tests plus 78 existing correspondence tests pass. The earlier staging-only test harness is retained for preparation traceability.
- `preparation.json`: exact evidence anchors and candidate hashes. `source-validation.json`: all 4,407 original source reports validated, canonical source hash unchanged, four policies / sixteen cases. `validation-root/` contains byte-identical source/proof copies used for the nonmutating validation.

## Explicit mapping and proof checks

The new `reviewed_cache_action_aggregate` kind accepts only the original `/required_export_action` pointer with the exact exclusion string. Verified status, other pointers, partial text, booleans, retain/no-exclusion actions and whitespace variants do not substitute for that decision. Existing v1–v3 acceptance is preserved. Policy `proof_binding_version: 2` still prevents a wrapper from downgrading itself to a bare verified-status assertion.

The runtime checks the exact blinded batch hash, source index/UID/case, report hash and contiguous source quotations. It also reads the original root and independent visual artifacts, verifies their hashes against the root replay and joint handoff, requires distinct canonical assessor roles, verifies both original full-source reports and structural quotations, and checks that recorded source freezes precede visual freezes, which precede the joint handoff and completed root replay. Both independent/handoff lineages must bind the same two source freezes.

These checks authenticate preserved content and recorded chronology. They do not prove access order beyond the original assessor attestations, independently reread pixels, regenerate displays, or rerun the root numeric audit. Root visual observations retain their bounded three-sample scope. The original joint handoffs still say numeric replay was pending at their creation; they are not rewritten. The later, separately bound completed root replay supplies the explicit authoritative actions. No literal side tag alone causes an exclusion.

Synthetic label tests demonstrate that only report-derived training masks and their provenance change; reports, clinical state/evidence/label values, groups, splits and synthetic gold evaluation are preserved. No official target columns or prior clinical annotations were opened for this proposal. The normalizer claims no new blinded clinical or technical pass.

## Validation and promotion boundary

115 focused tests pass, Ruff check/format pass under repository configuration, and mypy passes for the proposed runtime. All live production files, initiative policies/evidence, Git and HF remain unchanged by this task. Root must review and promote the overlay before it becomes effective; the candidate policy remains explicitly marked `staged_for_root_review_not_integrated`. Training readiness remains false.
