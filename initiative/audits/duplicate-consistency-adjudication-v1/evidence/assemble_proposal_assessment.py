"""Assemble post-freeze comparisons and independently authored dispositions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "work/recovered"
OUT = Path(__file__).resolve().parent
PROPOSAL = REPO / "initiative/audits/post-review-consistency4407-v1/root-duplicate-proposed-errata.json"
FREEZE = OUT / "source-preread-frozen.json"
IDENTITY = "/root/duplicate_consistency_review"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_new(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")

freeze = json.loads(FREEZE.read_text())
receipt = json.loads((OUT / "source-freeze-receipt.json").read_text())
for name, expected in receipt["files"].items():
    assert sha(OUT / name) == expected
proposal = json.loads(PROPOSAL.read_text())
assert proposal["agent_id"] != IDENTITY
assert proposal["session_id"] != IDENTITY
assert sha(PROPOSAL) == "8e0d4addcb1d7bf09205e03fb6c496580cb2b353005ef60a2192e952babe492e"
for key in ("rubric_sha256", "source_reports_sha256"):
    assert freeze[key] == proposal[key]

# Individual semantic conclusions authored after reading the proposal and exact
# original review rows. These preserve why each original differs from my freeze.
reasons = {
    ("batch-026", 49, "Effusion"): "Agree with negative to uncertain. This report's only fluid description is 'No significant knee effusion.' The original note treats significance as a fluid-negative criterion, but the frozen protocol requires explicit absence or an explicitly below-threshold amount. Neither the qualifying word nor 'Normal study' supplies that amount. My already-frozen reading independently retained uncertainty; the unchanged quote is exact.",
    ("batch-045", 19, "Effusion"): "Agree with negative to uncertain. The original note calls no-significant fluid below_threshold, but the report supplies no absent/small/trace grade that establishes this under the frozen rubric. The general normal-study conclusion adds no quantitative fluid information. My frozen per-source reading is uncertain with the identical source quote.",
    ("batch-011", 48, "Effusion"): "Agree with negative to uncertain. The original assertion that no-significant fluid excludes moderate/large distention is an inference beyond the report's stated amount. The no-significant-abnormality conclusion repeats a clinical-significance qualification and does not resolve the missing severity information. My frozen source interpretation therefore supports a masked uncertain state.",
    ("batch-057", 36, "Effusion"): "Agree with negative to uncertain. 'Pas de liquide significatif' is a qualified fluid statement, not an explicit absent/trace/small amount. The original below_threshold interpretation and normal-examination conclusion do not establish the required grade. My French source reading was frozen as uncertain before proposal access; this decision follows that report rather than selecting a duplicate's annotation.",
    ("batch-000", 44, "Fracture"): "Agree with uncertain to not_mentioned and empty evidence. The original note seeks acute fracture characterization from a patellar osteochondral lesion, but that generic lesion does not state fracture, cortical disruption, or fracture line. The missing acuity rule applies after a fracture finding is actually described; it should not create a fracture mention here. PF OA remains uncertain because the named lesion is informative for patellar cartilage but omits depth/extent. The Fracture loss mask remains zero.",
    ("batch-072", 27, "Contusion"): "Agree with uncertain to not_mentioned and empty evidence. The source supplies a patellar osteochondral lesion and Hoffa impingement, with no marrow edema, bone bruise, or impact injury. The original note's missing-impact-cause reasoning presupposes an informative contusion/marrow finding that is absent. My frozen reading already marked Contusion unmentioned; no negative is inferred and the mask remains zero.",
    ("batch-072", 27, "Fracture"): "Agree with uncertain to not_mentioned and empty evidence. The report never asserts fracture or a fracture line; the nonspecific osteochondral lesion cannot create such a mention. Unknown fracture identity/acuity in the old note is not a source statement. This repeats the same source-based reasoning independently used for 000:44, including preservation of PF OA uncertainty and a zero Fracture mask.",
}

sources = {r["study_uid"]: r for r in freeze["records"]}
originals = {}
bindings = []
comparisons = []
for uid, source in sources.items():
    path = REPO / "initiative/reviews-v2/reviewed" / (source["batch"] + "-reviewed.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [r for r in rows if r["study_uid"] == uid]
    assert len(matches) == 1
    original = matches[0]
    originals[uid] = original
    assert sha(REPO / source["source_path"]) == source["source_sha256"]
    binding = {"study_uid": uid, "batch": source["batch"], "index": source["index"], "source_path": source["source_path"], "source_sha256": source["source_sha256"], "artifact_path": str(path.relative_to(REPO)), "artifact_sha256": sha(path)}
    bindings.append(binding)
    for i, d in enumerate(source["decisions"]):
        original_quote = original["evidence"][i]
        assert original["states"][i] in {"positive", "negative", "uncertain", "not_mentioned"}
        assert (not original_quote) if original["states"][i] == "not_mentioned" else original_quote in source["report"]
        comparisons.append(dict(binding, target=d["target"], original_state=original["states"][i], original_evidence=original_quote, independent_source_state=d["state"], source_evidence=d["evidence"], reason=d["reason"], state_agreement=original["states"][i] == d["state"], evidence_agreement=original_quote == d["evidence"]))

records = []
for proposed in proposal["proposed_errata"]:
    uid = proposed["study_uid"]
    source = sources[uid]
    target = proposed["target"]
    key = (proposed["batch"], proposed["index"], target)
    reason = reasons[key]
    d = next(d for d in source["decisions"] if d["target"] == target)
    b = next(b for b in bindings if b["study_uid"] == uid)
    i = freeze["target_order"].index(target)
    original = originals[uid]
    for field in ("source_path", "source_sha256", "artifact_path", "artifact_sha256"):
        assert proposed[field] == b[field]
    assert proposed["before_state"] == original["states"][i]
    assert proposed["before_evidence"] == original["evidence"][i]
    assert proposed["proposed_state"] == d["state"]
    assert proposed["source_quote"] == d["evidence"]
    before_mask = int(proposed["before_state"] in ("negative", "positive"))
    after_mask = int(d["state"] in ("negative", "positive"))
    assert proposed["observed_mask_changes"] == (before_mask != after_mask)
    records.append(dict(b,
        target=target,
        before_state=original["states"][i],
        before_evidence=original["evidence"][i],
        proposed_state=proposed["proposed_state"],
        proposed_evidence=proposed["source_quote"],
        independent_source_state=d["state"],
        source_evidence=d["evidence"],
        disposition="agree_with_proposed_change",
        agrees_with_proposal=True,
        changed_from_source_freeze=False,
        original_observed_mask=before_mask,
        assessed_observed_mask=after_mask,
        observed_mask_changes=before_mask != after_mask,
        reason=reason,
        contextual_source_quote=source["report"] if d["state"] == "not_mentioned" else d["evidence"],
        contextual_quote_note="For not_mentioned, the full source is retained only as context; source_evidence remains empty as required."
    ))

assert len(records) == 7
assert len({r["study_uid"] for r in records}) == 6
assert len(comparisons) == 108
assert sum(not r["state_agreement"] for r in comparisons) == 7
assert {(r["study_uid"], r["target"]) for r in comparisons if not r["state_agreement"]} == {(r["study_uid"], r["target"]) for r in records}
for c in comparisons:
    if c["state_agreement"] and not c["evidence_agreement"]:
        assert c["batch"] == "batch-026" and c["target"] in {"Contusion", "Fracture"}
        c["evidence_disposition"] = "Both quotations are exact, anatomically explicit normal-bone findings. The independent freeze uses the whole sentence; no correction to the original shorter quotation is requested."

common = {"schema_version": 1, "agent_id": IDENTITY, "session_id": IDENTITY, "identity_kind": "codex_agent_task", "source_reports_sha256": freeze["source_reports_sha256"], "rubric_sha256": freeze["rubric_sha256"], "rubric_version": "report-weak-v2.0"}
comparison = dict(common, stage="post_freeze_all_target_original_comparison", completed_at_utc=datetime.now(timezone.utc).isoformat(), source_preread_sha256=sha(FREEZE), proposal_sha256=sha(PROPOSAL), n_studies=9, n_targets=108, n_state_agreements=101, n_state_disagreements=7, n_same_state_quote_extent_differences=2, original_reviews_preserved=True, records=comparisons)
write_new("all-target-original-comparison.json", comparison)
assessment = dict(common,
    stage="independent_source_assessment",
    status="assessment_complete_all_seven_proposals_supported_no_correction_registered",
    completed_at_utc=datetime.now(timezone.utc).isoformat(),
    source_preread_path=str(FREEZE.relative_to(WORKSPACE)),
    source_preread_sha256=sha(FREEZE),
    source_preread_jsonl_sha256=sha(OUT / "source-preread-frozen.jsonl"),
    source_freeze_receipt_sha256=sha(OUT / "source-freeze-receipt.json"),
    proposal_path=str(PROPOSAL.relative_to(REPO)),
    proposal_sha256=sha(PROPOSAL),
    all_target_comparison_path=str((OUT / "all-target-original-comparison.json").relative_to(WORKSPACE)),
    all_target_comparison_sha256=sha(OUT / "all-target-original-comparison.json"),
    historical_exposure=freeze["historical_exposure"],
    independent_identity_note=freeze["identity_note"],
    runtime_session_id=freeze["runtime_session_id"],
    runtime_thread_id=freeze["runtime_thread_id"],
    access_order="Full rubric/instructions and nine neutral full reports read; all 108 source decisions frozen and hash-receipted; only then the proposed errata and nine exact original review rows were opened. Root's preread and the candidate discrepancy file were not opened.",
    scope="Bounded assessment of nine supplied reports, all 108 source decisions and seven proposed changes across six studies. This is not all-release semantic QA, an image reread, or clinical diagnostic certification.",
    n_source_studies=9,
    n_source_target_decisions=108,
    n_proposed_changes=7,
    n_supported_changes=7,
    n_rejected_changes=0,
    n_reconsidered_source_freeze_states=0,
    n_observed_to_masked_changes=4,
    n_masked_to_masked_clarifications=3,
    original_artifacts_preserved=True,
    gold_values_inspected=False,
    rubric_changed=False,
    correction_registration_performed=False,
    release_qa_resolved=False,
    source_and_original_bindings=bindings,
    records=records,
    notes="All seven proposed states and exact after-evidence match the independent source freeze. The other 101 states match the original review rows; two unchanged bone-target quotes in 026:49 differ only in the extent of the exact normal-bones sentence and require no correction. Agreement on the two original Spanish Fracture uncertainty states did not make them source-supported. No annotation, official gold value, rubric, or correction registry was altered. The independent assessment supports the seven proposals; an explicitly authored accepted-decision artifact and validated integration are separate steps. Whole-release cartilage, bone terminology, and qualified-negation/duplicate QA remain outside this bounded assessment."
)
write_new("independent-assessment.json", assessment)
print(json.dumps({"assessment_sha256": sha(OUT / "independent-assessment.json"), "comparison_sha256": sha(OUT / "all-target-original-comparison.json"), "n_supported_changes": 7, "n_state_agreements": 101, "n_state_disagreements": 7}, indent=2))
