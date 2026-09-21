# Report-derived weak labels — protocol v2.0

Prepared 2026-09-20 from organizer guidance, without inspecting gold target values. This supersedes the semantics of the recovered workflow's rubric, not its immutable historical outputs. It is a versioned operational weak-supervision protocol; reports cannot reproduce an image reread.

## Binding source and compact rubric

The host's [official overview and label rubric](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) specifies single-knee examination labels, two independent MSK readers with a third adjudicator, and negative treatment of borderline image findings.

| Target | Official positive criterion / explicit exclusions |
| --- | --- |
| ACL | Complete disruption or high-grade partial tear (>50% fibers). Degeneration/thickening alone without discontinuity is negative. |
| MCL | Acute high-grade partial/complete tear with fiber disruption and edema. Low-grade sprain or chronic stress change is negative. |
| Medial / lateral meniscus | Corresponding meniscal signal reaching a surface on ≥2 images, or abnormal morphology. Isolated internal degeneration is negative. |
| Medial / lateral / PF OA | Corresponding compartment with roughly ≥1 cm cartilage-loss region, >50% thickness lost. |
| Effusion | Moderate/large distending joint fluid. |
| Synovitis | Inflamed, thickened synovial lining. |
| Baker's | Moderate/large fluid collection in the characteristic posterior location. |
| Contusion | Impact-related marrow edema without a discrete fracture line. |
| Fracture | Acute cortical disruption or fracture line. |

No additional postoperative/graft or subchondral/osteochondral exception was found in the checked rubric. Keep any proposed handling visibly separate from organizer facts.

## Why v1 cannot be promoted unchanged

The recovered `workflows/report_review.py` permits any ACL/MCL injury, trace/minimal effusion, and any Baker cyst; it uses an OA diagnosis/arthrosis heuristic instead of cartilage severity and extent. These do not implement the official target definitions. Its blanket fracture-subtype treatment and intact-reconstruction uncertainty rule are not organizer statements. Preserve all v1 reader/reviewer outputs with their original provenance. An independent v1 review does not certify v2 eligibility.

## Two distinct uncertainty layers

The organizers' negative policy concerns uncertain **image findings seen by their readers**. A weak labeler often lacks severity, extent, acuity, or laterality in the report. That absence is **missing evidence**, and must not automatically become a reliable negative for model training. This protocol therefore keeps the existing four-state output interface while recording why each state was chosen.

- `positive`: current, correctly localized report evidence explicitly establishes the criterion. An unspecified “partial tear,” “OA,” “effusion,” “Baker cyst,” or “marrow edema” does not establish an omitted severity/extent/causal requirement.
- `negative`: explicit normal/absent structure or explicit below-threshold finding under the relevant rule. Examples: explicitly low-grade MCL sprain, trace effusion, or isolated intrasubstance meniscal degeneration. Record `explicit_absence` versus `below_threshold` in notes.
- `uncertain`: report mentions the target but is equivocal, contradictory, incorrectly localized, or insufficient to establish or exclude the criterion. Record the exact missing requirement or ambiguity. Mask from the training loss.
- `not_mentioned`: no informative mention of that target/structure. Empty evidence and masked loss. Never infer a negative from silence.

These mappings are conservative extraction decisions for this project. They are not new clinical definitions and must be validated on non-gold review data before committing large training runs.

## Reading rules

1. Read each original report, preserving language, anatomy, laterality, negation scope, current finding versus history/indication, and severity. Do not use old labels, public pseudo-label tables, or gold columns to fill missing decisions.
2. Match ACL versus PCL and MCL versus LCL explicitly. A broad normal description can support negatives only for structures clearly within its scope. Do not infer contusion absence from “no fracture.”
3. Keep all three cartilage compartments distinct. “Chondromalacia” is neither automatic positive nor automatic negative: the recorded severity, extent, and compartment must support the selected state. Likewise, a bare OA diagnosis lacks the specific image-threshold evidence.
4. A definite current diagnosis of a tear of the named meniscus is accepted as report-level positive evidence: a clinical report need not restate the image-count criterion used to make that diagnosis. Specific tear morphology (for example bucket-handle, radial, displaced or horizontal tear) also supports positive. This is an explicit weak-supervision inference, not proof of the organizer's image criteria. Equivocal tearing stays uncertain; isolated intrasubstance degeneration is negative. Never infer an unmentioned compartment. Direct diagnostic conclusions can likewise support synovitis, traumatic bone contusion, or current acute fracture, while preserving their separate anatomy and causal requirements. Severity-qualified targets (ACL, MCL, OA, effusion, Baker's) still need the qualifying evidence described above.
5. Effusion does not establish synovitis. Other cysts do not establish a Baker cyst. Nonspecific or degenerative marrow edema does not establish traumatic contusion. A fracture subtype alone does not establish the required acute criterion.
6. For postoperative or bilateral ambiguity, retain `uncertain` and a specific note unless the current, relevant knee and criterion are supported. There is no verified organizer rule that every graft is positive, negative, or uncertain. Do not manufacture one.
7. Provide one contiguous original-language source quotation for every nonmissing state. The quote must include sufficient local context for identity, negation, and severity; validate that it is an exact source substring. Quotes prove traceability, not correctness.

## Artifact and review protocol

Use original target order: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.

Preserve the existing JSONL arrays (`study_uid`, 12 `states`, 12 `evidence`, `notes`) if needed for compatibility. Bind every v2 artifact in separate provenance to `rubric_version`, the rubric SHA-256, source-file SHA-256, exact UID list, reader/reviewer identities, stage, and parent artifact hashes. Do not silently overwrite v1 outputs or mix rubric versions in a supposedly final manifest.

Every v2-eligible report needs a blinded reader and a separate reviewer who rereads the source, not merely the draft. Review every target, preserve disagreement history, and retain unresolved items as masked. Reuse v1 work as an explicitly versioned draft only; require a new source-based v2 pass before promotion. Gold stays evaluation-only and is never changed from report evidence, per the [host's image-versus-report clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826).

Start with a predetermined non-gold pilot drawn across report languages and high-risk semantic categories. Measure reviewer disagreement, uncertainty reasons, class coverage, and evidence validity without opening gold values. If strict report evidence makes an entire target unusable, revise the weak-supervision design explicitly and test that revision independently; do not relax the published criterion silently.

Release requires full manifest/evidence coverage and clear masking, preserved gold and duplicate-group isolation, technical image acceptance, and an actual loader/training smoke check. Report quality measurements as weak-label extraction results. Neither compliance with this protocol nor agreement with report text certifies image diagnostic accuracy or competitive AUC.

## Frozen run identity

Protocol version: `report-weak-v2.0`. Preparation run: `codex-20260920-rubric-v2`. Read gold-blinded original batch inputs only. All 4,407 reports receive two independent passes; official evaluation rows remain excluded from training regardless of report extraction. Preserve separate files for this version under `initiative/reviews-v2/`. The existing v1 archive remains unchanged.
