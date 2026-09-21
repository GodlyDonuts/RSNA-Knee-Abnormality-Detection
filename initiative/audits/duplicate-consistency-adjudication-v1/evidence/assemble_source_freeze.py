"""Assemble personally reasoned decisions after all nine full reports were read.

The four explicit decision lists encode reasoning authored by the assessor, not
rules for predicting labels. Every source member was read and considered for all
12 targets before this script was written. Repeated decisions are intentional.
"""
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "work/recovered"
OUT = Path(__file__).resolve().parent
NEUTRAL = WORKSPACE / "work/annotation-staging/post-review-readiness4407-v1/duplicate-consistency-neutral-sources.json"
RUBRIC = REPO / "protocols/report_weak_v2.md"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IDENTITY = "/root/duplicate_consistency_review"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def decision(target, state, evidence, reason):
    return {"target": target, "state": state, "evidence": evidence, "reason": reason}

# English reports were each read in full, including the normal-study conclusion.
english_normal_study = [
    decision("ACL", "negative", "Normal ACL, PCL, MCL and LCL.", "explicit_absence: ACL is explicitly included in the named normal ligament list; no disruption or qualifying tear is described."),
    decision("MCL", "negative", "Normal ACL, PCL, MCL and LCL.", "explicit_absence: MCL is explicitly normal; no acute fiber disruption or high-grade tear is described."),
    decision("Medial Meniscus", "negative", "Normal medial and lateral menisci. No evidence of tears.", "explicit_absence: the medial meniscus is named normal and the following tear negation covers it."),
    decision("Lateral Meniscus", "negative", "Normal medial and lateral menisci. No evidence of tears.", "explicit_absence: the lateral meniscus is named normal and the following tear negation covers it."),
    decision("Medial OA", "negative", "Normal tibiofemoral and patellofemoral articular cartilage.", "explicit_absence: normal tibiofemoral cartilage covers the medial tibiofemoral compartment and excludes threshold cartilage loss in the report."),
    decision("Lateral OA", "negative", "Normal tibiofemoral and patellofemoral articular cartilage.", "explicit_absence: normal tibiofemoral cartilage covers the lateral tibiofemoral compartment and excludes threshold cartilage loss in the report."),
    decision("PF OA", "negative", "Normal tibiofemoral and patellofemoral articular cartilage.", "explicit_absence: the patellofemoral cartilage is explicitly normal."),
    decision("Effusion", "uncertain", "No significant knee effusion.", "Qualified negation: significant is not an explicit amount, absence, or trace/small grade. Clinical significance is not itself the frozen moderate/large distention criterion; the precise fluid threshold remains unspecified, so retain a mask."),
    decision("Synovitis", "not_mentioned", "", "No informative statement describes synovial lining, thickening, or synovitis. Fluid language and a generic normal-study conclusion do not establish a synovial finding."),
    decision("Baker's", "not_mentioned", "", "No posterior popliteal/Baker collection is identified or explicitly excluded. The generic surrounding-soft-tissue statement does not clearly delimit the characteristic recess."),
    decision("Contusion", "negative", "Normal visualized bones and surrounding soft tissues.", "explicit_absence: the direct normal-bones finding has broad osseous scope and supports absence of a described marrow contusion. This does not rely on fracture negation or on the general conclusion."),
    decision("Fracture", "negative", "Normal visualized bones and surrounding soft tissues.", "explicit_absence: explicitly normal visualized bones support absence of a report-level acute fracture; no conflicting osseous finding is present."),
]

# These two reports use a different conclusion, which does not add findings.
english_no_significant_abnormality = [
    decision("ACL", "negative", "Normal ACL, PCL, MCL and LCL.", "explicit_absence: ACL is individually named within the normal ligament list."),
    decision("MCL", "negative", "Normal ACL, PCL, MCL and LCL.", "explicit_absence: MCL is individually named within the normal ligament list."),
    decision("Medial Meniscus", "negative", "Normal medial and lateral menisci. No evidence of tears.", "explicit_absence: medial meniscal morphology is called normal and tears are expressly denied."),
    decision("Lateral Meniscus", "negative", "Normal medial and lateral menisci. No evidence of tears.", "explicit_absence: lateral meniscal morphology is called normal and tears are expressly denied."),
    decision("Medial OA", "negative", "Normal tibiofemoral and patellofemoral articular cartilage.", "explicit_absence: medial tibiofemoral cartilage is within the expressly normal tibiofemoral cartilage scope."),
    decision("Lateral OA", "negative", "Normal tibiofemoral and patellofemoral articular cartilage.", "explicit_absence: lateral tibiofemoral cartilage is within the expressly normal tibiofemoral cartilage scope."),
    decision("PF OA", "negative", "Normal tibiofemoral and patellofemoral articular cartilage.", "explicit_absence: patellofemoral articular cartilage is named normal."),
    decision("Effusion", "uncertain", "No significant knee effusion.", "The finding is explicitly qualified by significance, without a measured/described amount or direct absence. The nonspecific conclusion adds no fluid severity information; moderate/large threshold classification is not established."),
    decision("Synovitis", "not_mentioned", "", "No synovial lining or synovitis finding is supplied; neither effusion language nor generic soft-tissue normality clearly covers it."),
    decision("Baker's", "not_mentioned", "", "No Baker/popliteal collection or anatomically scoped posterior absence is described."),
    decision("Contusion", "negative", "Normal visualized bones and surrounding soft tissues.", "explicit_absence: normal visualized bones broadly exclude a report-level contusion finding in the assessed knee; the conclusion is not the evidence basis."),
    decision("Fracture", "negative", "Normal visualized bones and surrounding soft tissues.", "explicit_absence: normal bones provide direct osseous evidence against fracture, without relying only on a nonspecific normal conclusion."),
]

french = [
    decision("ACL", "negative", "Ligament croisé antérieur : Normal.", "explicit_absence: the anterior cruciate ligament is specifically normal."),
    decision("MCL", "negative", "Ligament collatéral médial : Normal.", "explicit_absence: the medial collateral ligament is specifically normal."),
    decision("Medial Meniscus", "negative", "Ménisque médial : Normal.", "explicit_absence: the medial meniscus is specifically normal."),
    decision("Lateral Meniscus", "negative", "Ménisque latéral : Normal.", "explicit_absence: the lateral meniscus is specifically normal."),
    decision("Medial OA", "negative", "Changements dégénératifs :\nAucune.", "explicit_absence: the unqualified whole-knee denial of degenerative changes includes the medial compartment. No compartmental positive or cartilage lesion contradicts it."),
    decision("Lateral OA", "negative", "Changements dégénératifs :\nAucune.", "explicit_absence: the broad degenerative-changes section denies changes throughout the knee, including the lateral compartment; it is not silent on degeneration."),
    decision("PF OA", "negative", "Changements dégénératifs :\nAucune.", "explicit_absence: the whole-knee denial of degeneration also includes the patellofemoral compartment, with no stated exception."),
    decision("Effusion", "uncertain", "Épanchement articulaire : Pas de liquide significatif.", "Qualified negation: no significant fluid does not specify absence, trace/small amount, or a reproducible relation to moderate/large distention. Preserve uncertainty instead of translating significance into an unstated volume grade."),
    decision("Synovitis", "not_mentioned", "", "No synovial lining/thickening finding or synovitis diagnosis is stated. The fluid section and generic normal conclusion do not provide it."),
    decision("Baker's", "negative", "Kyste de Baker : Aucun.", "explicit_absence: a Baker cyst is expressly absent."),
    decision("Contusion", "negative", "Signal de la moelle osseuse :\nNormal.", "explicit_absence: normal marrow signal directly excludes reported marrow edema/contusion; this is separate from the fracture finding."),
    decision("Fracture", "negative", "Fractures :\nAucune.", "explicit_absence: the fracture section directly states none."),
]

spanish = [
    decision("ACL", "not_mentioned", "", "Neither the anterior cruciate ligament nor a scoped ligament finding is described."),
    decision("MCL", "not_mentioned", "", "Neither the medial collateral ligament nor a scoped ligament finding is described."),
    decision("Medial Meniscus", "not_mentioned", "", "The medial meniscus is not discussed; patellar and Hoffa findings do not describe it."),
    decision("Lateral Meniscus", "not_mentioned", "", "The lateral meniscus is not discussed; no unmentioned meniscal state is inferred."),
    decision("Medial OA", "not_mentioned", "", "No medial tibiofemoral cartilage or degenerative finding is provided. The patellar lesion cannot localize here."),
    decision("Lateral OA", "not_mentioned", "", "No lateral tibiofemoral cartilage or degenerative finding is provided. The patellar lesion cannot localize here."),
    decision("PF OA", "uncertain", "Lesión osteocondral de la rótula.", "The osteochondral lesion is definitely patellar and therefore informs the PF cartilage target, but depth, surface area/extent, and loss severity are not given. Neither positive nor below-threshold negative is established."),
    decision("Effusion", "not_mentioned", "", "No joint-fluid finding is present. Hoffa fat-pad impingement does not mention effusion."),
    decision("Synovitis", "not_mentioned", "", "Hoffa fat-pad impingement is not a diagnosis or description of inflamed/thickened synovial lining, and no separate synovial finding appears."),
    decision("Baker's", "not_mentioned", "", "No posterior/Baker cyst or collection is mentioned."),
    decision("Contusion", "not_mentioned", "", "An unspecified osteochondral lesion does not itself mention marrow edema, bone bruise, or impact injury. No informative contusion finding is supplied, so absence is not inferred and uncertainty is not created from a generic osseous word."),
    decision("Fracture", "not_mentioned", "", "The report says osteochondral lesion, not fracture, cortical disruption, or fracture line. There is no fracture assertion requiring acuity adjudication; no fracture negative is inferred."),
]

# Explicit per-report selection records what was independently reasoned for each
# full source. No selection is derived from text matching or old annotations.
authored = [
    ("1.2.826.0.1.3680043.8.498.17769942698581399165275319502954100325", english_normal_study, "026:49: personally read the full normal-study report. All 12 decisions above account for the qualified fluid statement, named normal structures, broad normal bones, and unmentioned synovial/Baker targets."),
    ("1.2.826.0.1.3680043.8.498.41810826648175757249770990269365542940", english_normal_study, "045:19: personally read this complete source separately. Its identical findings and conclusion supply the same per-target evidence; identical wording, rather than another review's labels, explains agreement."),
    ("1.2.826.0.1.3680043.8.498.75577846246542841539613557336025271434", english_normal_study, "069:35: personally read this entire source and reconsidered each of the 12 targets. The normal-study conclusion does not resolve fluid qualification or add missing synovial/Baker anatomy."),
    ("1.2.826.0.1.3680043.8.498.11725843001854130477287484865097535511", english_no_significant_abnormality, "011:48: personally read the complete findings and no-significant-abnormality conclusion. The latter is not used to overwrite target-specific uncertainty or silence."),
    ("1.2.826.0.1.3680043.8.498.32179217289301203414651193280580594771", english_no_significant_abnormality, "037:39: personally read this complete source separately and considered all 12 targets. Named normal structures support negatives; no-significant fluid remains imprecise."),
    ("1.2.826.0.1.3680043.8.498.25505153402978560400776672230220616863", french, "032:22: personally read every French section and the conclusion. Direct normal marrow supports Contusion negative, explicit no fracture supports Fracture negative, and significant-fluid wording remains masked."),
    ("1.2.826.0.1.3680043.8.498.59511171909586924989766685687032462369", french, "057:36: personally read the entire French source, including its final trailing space. Each target has the same substantive evidence as the other French source; trailing whitespace is preserved in source binding."),
    ("1.2.826.0.1.3680043.8.498.10133843191898064967257276100779798471", spanish, "000:44: personally read the full Spanish technique, results and repeated impression. The patellar lesion informs PF cartilage uncertainty only; Hoffa impingement adds none of the other target findings."),
    ("1.2.826.0.1.3680043.8.498.79087227780630716132147615846840128474", spanish, "072:27: personally read this complete Spanish source separately. Every target was considered; repeating the impression does not add marrow impact or fracture evidence to the unspecified patellar lesion."),
]

assert sha(RUBRIC) == "138c5cc33b92901d57dbf5fbe7e3a9878b73d25b0af6934908d0052cd6786014"
groups = json.loads(NEUTRAL.read_text())
sources = {r["study_uid"]: dict(r, report_group=g["report_group"]) for g in groups for r in g["source_records"]}
assert len(sources) == len(authored) == 9

# Verify the canonical blinded fingerprint without reading official target files.
all_reports = {}
for p in sorted((REPO / "initiative/reviews/inputs").glob("batch-*.json")):
    for r in json.loads(p.read_text()):
        assert set(r) == {"study_uid", "report"}
        assert r["study_uid"] not in all_reports
        all_reports[r["study_uid"]] = r["report"]
fingerprint_bytes = json.dumps([{"study_uid": u, "report": all_reports[u]} for u in sorted(all_reports)], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
fingerprint = hashlib.sha256(fingerprint_bytes).hexdigest()
assert fingerprint == "5c18838a09e9eec6522966555694fca83a9bae67e4d7c5c68592207ee4b60ed3"

records = []
annotations = []
for uid, decisions, note in authored:
    source = sources[uid]
    p = REPO / source["source_path"]
    assert sha(p) == source["source_sha256"]
    original = json.loads(p.read_text())[source["index"]]
    assert original == {"study_uid": uid, "report": source["report"]}
    assert all_reports[uid] == source["report"]
    assert [d["target"] for d in decisions] == TARGETS
    for d in decisions:
        assert d["state"] in {"positive", "negative", "uncertain", "not_mentioned"}
        assert (d["evidence"] == "") if d["state"] == "not_mentioned" else (d["evidence"] and d["evidence"] in source["report"])
    records.append(dict(source, report_sha256=hashlib.sha256(source["report"].encode()).hexdigest(), decisions=deepcopy(decisions), source_read_note=note))
    annotations.append({"study_uid": uid, "states": [d["state"] for d in decisions], "evidence": [d["evidence"] for d in decisions], "notes": note + " " + " ".join(d["target"] + ": " + d["reason"] for d in decisions)})

freeze = {
    "schema_version": 1,
    "stage": "independent_source_preread_frozen_before_proposal_access",
    "agent_id": IDENTITY,
    "session_id": IDENTITY,
    "identity_kind": "codex_agent_task",
    "runtime_session_id": os.environ.get("CODEX_SESSION_ID"),
    "runtime_thread_id": os.environ.get("CODEX_THREAD_ID"),
    "identity_note": "The actual canonical agent-task identity is used for agent_id/session_id per protocols/agent_tasks.md. Runtime IDs are separately disclosed and may be inherited from the coordinating process; they are not asserted to prove separate provider sessions.",
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "workflow_run_id": "codex-20260920-rubric-v2",
    "rubric_version": "report-weak-v2.0",
    "rubric_sha256": sha(RUBRIC),
    "source_reports_sha256": fingerprint,
    "neutral_sources_path": str(NEUTRAL.relative_to(WORKSPACE)),
    "neutral_sources_sha256": sha(NEUTRAL),
    "scope": "Only the nine supplied neutral reports, all 12 targets each; 108 personally reasoned decisions. Not a whole-release semantic assessment.",
    "historical_exposure": "No historical case-level clinical annotations, proposal, discrepancy records, root assessment, or official gold values were visible in this subagent's provided context or opened before this freeze. The assignment stated that seven proposals existed. Required docs/codex-continuation.md disclosed their aggregate directions: four qualified-effusion masks, one Contusion and two Fracture uncertainty-to-unmentioned changes. docs/adjudications.md disclosed seven proposals across six studies. This is a source-first pass before per-case proposal/base access, with those aggregate cues explicitly disclosed; it is not claimed fully blind to the existence or broad categories of proposed corrections.",
    "method": "Personally read each complete neutral report as displayed before writing the decision lists. Individually considered all targets using the frozen rubric. The assembly script copies authored equivalent decisions for separately read duplicate sources and validates evidence/hashes; it does not infer states from text or prior labels.",
    "n_studies": 9,
    "n_target_decisions": 108,
    "target_order": TARGETS,
    "records": records,
}
for filename, content in [("source-preread-frozen.json", json.dumps(freeze, ensure_ascii=False, indent=2) + "\n"), ("source-preread-frozen.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in annotations))]:
    p = OUT / filename
    with p.open("x") as f:
        f.write(content)

receipt = {"status": "source_freeze_completed_before_opening_proposal_or_original_clinical_rows", "agent_id": IDENTITY, "session_id": IDENTITY, "frozen_at_utc": freeze["frozen_at_utc"], "n_studies": 9, "n_target_decisions": 108, "files": {name: sha(OUT / name) for name in ["source-preread-frozen.json", "source-preread-frozen.jsonl", "assemble_source_freeze.py"]}, "rubric_sha256": sha(RUBRIC), "neutral_sources_sha256": sha(NEUTRAL), "source_reports_sha256": fingerprint, "exact_source_binding_and_evidence_checks": "passed"}
with (OUT / "source-freeze-receipt.json").open("x") as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(receipt, indent=2))
