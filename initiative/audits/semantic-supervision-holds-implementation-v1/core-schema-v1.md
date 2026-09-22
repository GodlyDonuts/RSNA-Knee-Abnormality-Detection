# Semantic report-supervision contract v1

Frozen schema/interface for the real-data builder, authored by `/root/quality_temporal_review`. This defines validation mechanics, not real hold membership, clinical acceptance, QA resolution, or a release approval. The implementation tests use synthetic approvals only. Full coverage is the exact canonical UID set multiplied by the 12 targets in `rsna.schema.TARGETS`; production therefore has 52,884 cells. No sampled or empty-list replacement is accepted.

## Encoding and references

`FileRef = {path, sha256}` binds exact file bytes at an artifact-root-relative path. Paths must resolve inside that root. JSON and `.json.gz` are supported; hashes cover compressed bytes. `RecordRef = {artifact: FileRef, format: json|jsonl, selector, sha256}` binds a raw original object. JSON selectors are JSON Pointers; JSONL selectors are zero-based integer line numbers. The record/annotation/context digest is SHA256 of UTF-8 `json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`, **without a terminal newline**. Historical digests using a newline remain historical metadata and must not be substituted for this projection digest. Duplicate JSON fields, malformed pointers, and nonfinite values fail.

Every schema version is integer 1, never boolean. Evidence IDs must be stable within their bound catalog and may namespace an existing preflight completion and index; IDs alone never replace raw file+selector bindings.

## Acyclic artifacts

Fixed required registry path: `initiative/reviews-v2/semantic-supervision/registry.json`.

Registry has exactly `schema_version, context, holds, ledger, proposal, assessment, decision, previous_registry`; all except schema and nullable previous_registry are FileRefs. Holds has exactly `schema_version, context, records`. Ledger has exactly `schema_version, context, holds, evidence_catalog, rows`. Catalog has exactly `schema_version, context, records`. Context is computed by `inspect_semantic_supervision`, not invented by a candidate adapter.

Context binds the canonical report fingerprint and complete source files; original reviewed files; all original draft and preread files; frozen rubric; whole accepted correction objects and correction case map; exact correspondence exclusions; exact known near-report edge pairs; and independently pinned required inputs. Required input names are exactly `review_provenance, retirement_manifest, correspondence_policy, similarity_policy, split_policy, technical_policy, scope_manifest, authenticated_train_csv, near_report_dispositions`. All are FileRefs. The caller must obtain these expected pins independently of the proposed registry. The authenticated train CSV is hashed opaque at inspection; its gold values are used only at the later release boundary to verify gold, never to select holds.

The context deliberately excludes the complete mutable adjudication/QA registry hash. Current adjudications are freshly inspected; context binds their stable correction objects/case map instead. Thus context → holds → ledger → policy approval → semantic registry → QA resolutions → final release has no hash cycle. Original case and review maps cannot change unnoticed. Required resolution checks augment the existing adjudication validator.

## Original evidence catalog

Each entry has `id, study_uid, target, source_report_sha256, rubric_sha256, agent_id, session_id, kind, layout, annotation_sha256, record:RecordRef, identity:RecordRef`.

Kinds are `original_source, supplemental_source, postfreeze, proposal, rejection, resolution`. Source identities count only actual distinct `agent_id` values. Identity wrappers do not create another reader. Identity record has actual `agent_id` and nullable `session_id` (optional `identity_agent_pointer` and `identity_session_pointer` select authentic alternative/nested original fields; absent default session is null), and must be in the authored artifact or explicitly bind its SHA through a preserved original field. Recognized default ownership fields are `sha256`, or nested `authored_artifact/source_assessment/assessment/artifact.sha256`. Optional `ownership_sha256_pointer` selects another exact SHA from that same hash-bound identity record, such as an original `independent_preread_sha256`. It must equal the authored raw artifact SHA.

Supported raw projections preserve the entire selected original object hash:
- `arrays12`: original UID row with states/evidence in fixed 12-target order.
- `arrays_scoped`: states/evidence aligned to row `target_order`, or optional `target_order_record:RecordRef` plus `target_order_pointer` (default `/target_order`) from the authored artifact or its bound identity freeze. No copied order without original binding.
- `decisions`: original UID parent with decisions list, exactly one matching target entry containing state/evidence.
- `scalar`: original UID+target row; optional `annotation_pointer` selects the state/evidence object.
- `scalar_fields`: same, but required explicit `state_field` and `evidence_field` select flat original fields (for example root_postfreeze_state/root_postfreeze_evidence).

Each projected state is canonical. Nonmissing evidence is nonempty and an exact contiguous substring of the full original report; not-mentioned evidence is empty. Metadata-only normalization never acquires clinical authorship.

## Finite obligations, holds, and ledger

Scope manifest has exactly `schema_version, source_reports_sha256, rubric_sha256, target_order, obligations`. Each obligation has unique `id`, study_uid, target, category, scope_kind (`target` or `whole_report`); whole-report obligations carry `report_scope_id` and explicitly enumerate all 12 targets. Categories are `state_disagreement, evidence_reliability, incomplete_reliability_review, report_scope_ambiguity`. Every known obligation must appear once in its canonical ledger cell. Discovery obligations are not automatically reviewed errors. Absence from a lexical queue is not evidence of clearance.

Hold record: `id, study_uid, target, category, scope_kind, reason, current_annotation_sha256, evidence_ids, obligation_ids, origin_id, lifecycle, predecessor_sha256`. scope_kind is `text_intrinsic` or `record_specific`. Every reason and its bound evidence/obligations remain explicit; a display category cannot drop overlapping reasons. Current annotation digest covers exact `{state,evidence}`. New holds must be active with null predecessor. Prior holds cannot disappear. Changed records require exact predecessor object digest and transition_reason; releases require resolution_evidence_ids. Prior registries and objects remain immutable. No implicit release occurs when annotations or sources change.

Text-intrinsic active holds must enumerate all normalized exact-report copies for the same target/origin; normalization is NFKC, casefold, whitespace collapse. Propagated records carry the original origin_id and exact local annotation/evidence bindings. Record-specific defects stay local. Known near-report edges require the separate bound near_report_dispositions object: schema_version, source_reports_sha256, edges. Each exact known edge (`left`, `right`) enumerates all 12 target_dispositions with target, action (`no_semantic_transfer` or `separately_held`), reason, and nonempty raw RecordRefs in evidence. Separately-held requires active holds on both records. Similarity alone is never a clinical-equivalence claim.

Each ledger row: `study_uid, target, source_report_sha256, original_annotation_sha256, current_annotation_sha256, evidence_ids, hold_ids, obligations, retention_reason`. hold_ids includes all active/released history IDs on that cell. Obligations carry id and one disposition:
- `held`: an active hold explicitly covers it.
- `already_masked`: current state is uncertain or not-mentioned.
- `correspondence_excluded`: UID is in the separately validated exact policy.
- `resolved`: requires `resolution_evidence_ids` with same-cell resolution annotations matching the retained current state, plus explicit `discharged_evidence_ids` (possibly empty) naming the exact source opinions discharged by this obligation. A historical or quote-only resolution cannot silently settle unrelated/later dissent.

Retained observed cells require at least two actual distinct source authors, exact current-state support, nonempty retention_reason, and explicit resolution coverage for every cited contrary source opinion. Quote variants are not automatically defects; known evidence reliability objections remain separate obligations. The approved finite scope must retain all known disagreements; omitting an objection from the candidate adapter is not a clinical resolution.

## Concrete policy approval

Proposal/assessment/decision each bind exact context, holds, ledger, actual agent_id/session_id, and reason. Statuses respectively: `proposed_for_report_supervision`, `endorsed_for_report_supervision`, `accepted_for_report_supervision`. Assessment binds proposal; decision binds proposal and assessment. Assessor must differ from proposer and decider. These are actual policy decisions about fixed membership, not fabricated approvals or agreement counts. Empty holds are possible only if the complete canonical ledger and every bound obligation justify them, with actual explicit approval; an empty registry or omitted scope fails.

## Application and verified boundary

`inspect_semantic_supervision(reports, reviews_folder, rubric_path, artifact_root, adjudications, required_inputs, *, correspondence_excluded_uids, near_report_pairs)` receives a fresh validated adjudication inspection and independently verified policy memberships. It returns an immutable digest-bound complete audit; it does not resolve gates.

`apply_semantic_supervision(studies, audit, *, gold_uids, technical_eligible_uids)` preserves clinical states, quotes, provenance and numeric targets. Non-gold numeric values must agree with the preserved state. It only intersects existing masks with approved semantic eligibility, permitted split and independently validated technical eligibility. Gold membership comes from authenticated source membership and preserves official values/masks. State/correspondence exclusions remain prior masks. Per-target semantic hold IDs and all reasons are exported. Held cells have exactly zero loss/gradient.

`validate_qa_binding(resolution, audit)` supplements existing QA validation: all three gates require resolved records bound to exact review map, correction map and registry/context/holds/ledger/decision. It never writes a resolution.

`verify_release_boundary` verifies persisted full-table identity, fresh bound artifacts, independently authenticated source/gold membership/target values, exact masks, technical eligibility, all three QA bindings, and the approved semantic contract. Train/resume and labeled inference must validate the complete manifest before selecting subsets. Public-test inference retains the checkpoint training-release identity without reopening training data. Downstream class-count/coverage approval and deterministic diagnostic selection are release/CLI concerns, not changes to this frozen membership schema. Counts are measured after fixed membership; they never select holds using gold values.
