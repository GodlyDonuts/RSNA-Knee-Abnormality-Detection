# Qualified wording and severity: candidate discovery only

This read-only audit processes all 89 final reviewed batches, all 4,407 blinded
reports, and 52,884 target cells. It validates exact evidence and binds every
input/review file to its checksum. All 34 registered corrections are applied only
to an in-memory inspection copy. No new clinical state is inferred or proposed.

After those corrections, all 54 normalized duplicate-report groups (204 studies)
have matching target states: zero conflicting groups and zero conflicting
group/target pairs. This does not establish semantic correctness.

The broader multilingual lexical and positive-threshold screens select 10,285
target candidates across 3,173 reports; 3,135 selected reports are outside the
normalized duplicate groups. These are deliberately inclusive retrieval results,
including findings that may fully support their existing states:

| Priority | Target candidates | Reports |
| --- | ---: | ---: |
| Observed state with direct evidence match or positive-threshold screen | 4,718 | 2,551 |
| Observed state linked by source-wording proximity | 797 | 574 |
| Already masked target | 4,770 | 2,083 |

Report counts overlap across priorities. Another 360 unassigned qualifier cues
are retained for scope review. The 260 excluded cues are all the literal word
`relevant` in the explicit prior-comparison phrase; exclusions retain their source
excerpts. The lexical rules are not exhaustive multilingual semantic review.

Two narrower views help select the next independent source pass:

- Direct qualified wording inside observed positive/negative evidence: **539
  target candidates in 331 reports**, with seven source-only batches.
- Identical normalized complete evidence quotations with differing states:
  **60 target/quotation groups in 409 reports**. All have an observed state and
  a masked counterpart; none has opposite observed positive/negative polarities.
  Different full-report context can explain such differences, so these remain
  candidates, not corrections.

The largest mixed-state wording buckets concern Spanish qualified marrow
negation, Spanish joint-fluid/popliteal-cyst negation, English qualified fluid
negation, and English gross-fracture negation. Broadly matched cartilage grades,
depth/extent language and fluid amounts are retained separately. Target-family
anchors are retrieval associations, not clinical localization: a cartilage cue
may nominate all three compartments, and fluid proximity does not prove an
intra-articular or Baker-cyst location.

## Start a separate independent source reading

Use the frozen `protocols/report_weak_v2.md`. Read complete source reports and
freeze source-based decisions before opening any candidate-detail files.
Membership in these packets is selected by this audit and is not a random or
fully blinded sample. No target, priority, clinical annotation, evidence quote,
or proposed correction is included inside the source-only files.

- `focused-neutral-full-report-packet.json` and `focused-neutral-batches/`:
  the 331-report observed/direct-qualifier priority packet.
- `exact-wording-neutral-full-report-packet.json`: the 409 reports represented
  in mixed-state exact-wording groups.
- `neutral-full-report-packet.json` and `neutral-batches/`: all 3,173 selected
  reports, also split into 64 source-only batches.

After that independent freeze, inspect the matching separate detail artifact:

- `focused-qualified-candidate-details.json`: the 539 direct observed targets.
- `exact-wording-state-consistency-candidates.json`: all 60 mixed-state buckets
  with exact source/base bindings and existing states.
- `candidate-details.jsonl`: every target candidate with original and inspection
  states, exact evidence, rule IDs, and original-language source excerpts/offsets.

`selection-method.json` and the two scripts preserve deterministic selection,
normalization, target association, priorities and known limitations.
`full-coverage-scan.jsonl` accounts for every report, including those not selected.
`audit-summary.json` binds all 178 input/review artifacts. The existing review
index, frozen rubric and adjudication registry are also pinned.

`validation-receipt.json` records coverage, exact source-locator checks, source-only
packet checks, and preservation checks. `artifact-manifest.json` binds the complete
staged deliverable. All three global semantic QA gates remain pending. No original
source, review, registry, documentation, release manifest, or official gold was
changed or published. These are discovery results, not 4,407 personally reread
clinical reports and not a final training release.
