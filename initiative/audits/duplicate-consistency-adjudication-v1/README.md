# Seven independently adjudicated report corrections

The independent assessor `/root/duplicate_consistency_review` explicitly accepted
all seven changes proposed by `/root`. The source-first freeze covers all nine
neutral reports and 108 target decisions. Aggregate proposal categories disclosed
by required project instructions are documented; the per-case proposals and
original clinical rows were opened only after the full source freeze.

| Reports | Target | Accepted change | Report-loss effect |
| --- | --- | --- | --- |
| 026:49, 045:19, 011:48, 057:36 | Effusion | negative → uncertain | Observed negative becomes masked |
| 000:44, 072:27 | Fracture | uncertain → not_mentioned | Remains masked |
| 072:27 | Contusion | uncertain → not_mentioned | Remains masked |

Qualified fluid significance does not supply an explicit absent/trace/small amount
under the frozen operational rubric. An unspecified patellar osteochondral lesion
informs PF cartilage uncertainty but does not assert contusion or fracture.

The complete pre-acceptance evidence package is copied byte for byte under
`evidence/`, including its original manifest, disclosure, source freeze,
all-target comparison, assessment, validation receipt, and assembly scripts.
Its historical status text remains unchanged. `accepted-decision.json` is the
later explicit decision authored by the actual independent adjudicator.

The registered case is
`initiative/reviews-v2/adjudications/cases/duplicate-qualified-negation-nine-report.json`.
The real `inspect_adjudications` check passed against all 4,407 blinded reports:
34 total accepted target corrections, including this case's seven, and all three
global semantic QA issues still pending. The 27 existing adjudication tests pass.
No new tests or implementation changes were needed.

`registry-before.json` preserves the original registry. Original-artifact hashes
cover all 89 blinded inputs, 89 drafts, 89 reviews, six prior correction cases,
three pending QA files, and the frozen rubric. They remained unchanged.
`integration-receipt.json` records preservation, exact bindings, and verification.

This bounded acceptance does not resolve full-release cartilage, bone terminology,
or qualified-negation/duplicate semantic QA. No effective label export, official
gold inspection, rubric change, Git commit/push, or network publication occurred.
The archive manifest and continuation documents remain for the coordinating task.
