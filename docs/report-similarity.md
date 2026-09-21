# Reviewed report-similarity leakage control

Reports 038:44 and 045:15 differ only by one redundant full stop and adjacent
space. Existing case/Unicode/whitespace normalization gives them different
`report_group` values. The label exporter now joins this reviewed pair in the
leakage graph before assigning splits or quarantining neighbors of official gold.
Exact-report and verified pixel edges participate in the same transitive graph.

`initiative/audits/report-similarity-registry.json` is required by `rsna.labels`.
It must list every preserved `report-similarity-dispositions-v*.json` exactly once,
with matching file hashes. There is no skip switch. An empty registry is valid
only when no policies exist. Every accepted edge binds canonical blinded source
batches, both complete raw reports, two separately identified source assessments,
their joint comparison, and the preserved difference proof. Missing evidence,
changed text, disagreement, duplicate edges, or invalid admission blocks export
before creating output files.

The supported relation is deliberately narrow: one local `. . ` becomes `. `,
with every other character unchanged. The exporter does not remove punctuation
globally or perform fuzzy matching. Source-first admission is an explicit reviewed
attestation linked to immutable readings and their exposure disclosures; it is
not an automated interpretation of prose or proof of real-world identity.

Original reports, normalized report groups, reviewed states, evidence and official
gold values remain unchanged. The added edge may change split membership, and
report-supervision exclusions retain their graph connections. The exporter writes
`report_similarity_trace.json` with the registry/policy/evidence hashes and final
component membership and splits. The same trace is included in `label_audit.json`.

Independent synthetic regression tests exercise transitive gold quarantine,
unchanged annotations, order independence, tampered inner evidence with rehashed
outer references, and successful and blocked exports. Separate source-only
validation covers all 4,407 blinded reports and the real registered pair.

This is leakage control, not evidence of shared patient, hospital or pixels.
The full final label export, remaining correspondence policies and training checks
are still required before declaring the data ready.
