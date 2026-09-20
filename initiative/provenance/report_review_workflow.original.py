import asyncio
import csv
import json
from pathlib import Path


SOURCE = Path("/home/ubuntu/rsna-hf-repo/train.csv")
OUTPUT = Path("/home/ubuntu/rsna-readiness/report_review_results.json")
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
SCHEMA = {
    "type": "object",
    "properties": {
        "artifact_url": {"type": "string"},
        "n_studies": {"type": "integer"},
        "notes": {"type": "string"},
    },
    "required": ["artifact_url", "n_studies", "notes"],
}
RUBRIC = """
You are preparing research-only weak labels for the RSNA 2026 Knee Abnormality
Detection competition, https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data.
Your task is to PERSONALLY READ AND REASON THROUGH EVERY PROVIDED RADIOLOGY REPORT,
including its original language, to annotate all twelve study-level findings.
You must not use regex/NLI models, copy public pseudo-labels, or write an automatic
labeler as a substitute for reading each report. Python may ONLY assemble/validate
the explicit annotations you reasoned through. Do not inspect gold labels or other
batches. This is report extraction, not image verification or clinical diagnosis.

Targets, in exact order: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

Definitions: ACL and MCL refer to injury of the anterior cruciate and medial
collateral ligaments, respectively. Do not confuse PCL/LCL with them. Meniscus
targets mean TEARS of the specified meniscus, not isolated degenerative signal.
OA targets mean osteoarthritis of the named medial/lateral tibiofemoral or
patellofemoral compartment. Effusion is excess joint fluid. Synovitis is synovial
inflammation/thickening, not automatically present with effusion. Baker's is a
popliteal/Baker cyst, not another cyst. Contusion means bone bruise/contusion;
isolated marrow edema from OA, fracture, or nonspecific causes is not definitely a
contusion. Fracture includes a stated subchondral/insufficiency fracture but not
automatically osteonecrosis, contusion, or marrow edema.

Each target must have one state:
  positive: definite current/relevant finding explicitly supported by report.
  negative: explicit absence or explicit normal description of this structure.
  uncertain: possible/equivocal, conflicting, insufficient compartment detail,
    ambiguous postoperative status, or ambiguous mapping to the target.
  not_mentioned: no informative mention of the target or its structure.

Never turn silence into a negative. A report saying all ligaments intact supports
negative ACL/MCL. A report saying no bone abnormality supports negative fracture
and contusion, but no fracture alone does not rule out contusion. Trace/minimal
effusion counts positive if called an effusion; physiologic fluid is negative.
Do not equate isolated chondromalacia/cartilage signal with definite OA: label
uncertain unless explicit OA or compartment-localized degenerative arthrosis
with cartilage loss/osteophytes supports it. Do not infer the lateral compartment
from unspecified 'tibiofemoral OA'. An intact ACL reconstruction without recurrent
injury is uncertain (history alone is not current tear). Distinguish negation
scope, differential diagnosis, clinical indication, and actual imaging findings.
Translate reasoning internally but quote evidence verbatim in original language.

Create JSONL with ONE record per input study:
{"study_uid":"original UID",
 "states":["positive","negative",... exactly 12],
 "evidence":["verbatim short quote",... exactly 12],
 "notes":"brief ambiguity notes or empty"}
Use empty evidence ONLY for not_mentioned. For other states evidence must be an
exact contiguous substring of the source report, long enough to convey negation
and anatomy (generally 10-240 characters). Never invent quotations. Notes can be
English. Read all reports and make all target decisions; do not just annotate
the first ones and fill the rest with defaults.

Create output through apply_patch in manageable groups, validate every input UID
appears exactly once, all arrays have 12 entries, allowed states, and quotes are
substrings. Upload the final JSONL with upload_attachment and return artifact_url,
n_studies, and concise notes in the required structured output. Do NOT create a
PR or send messages to the user. Do not print or store credentials.
"""


async def main():
    with SOURCE.open() as handle:
        reports = [
            {"study_uid": row["StudyInstanceUID"], "report": row["Report"]}
            for row in csv.DictReader(handle)
        ]
    reports.sort(key=lambda row: row["study_uid"])
    batches = [reports[start:start + 50] for start in range(0, len(reports), 50)]
    labels = [f"batch-{index:03d}" for index in range(len(batches))]
    await register_workflow({
        "name": "rsna-evidence-review",
        "description": "Blinded per-study reading and independent review of all 4,407 radiology reports.",
        "soft_time_limit_minutes": 25,
        "phases": [
            {"title": "annotate", "detail": "Read each report and quote evidence for 12 findings.", "labels": labels},
            {"title": "review", "detail": "Independently check every report and correct the draft.", "labels": labels},
        ],
    })
    semaphore = asyncio.Semaphore(6)

    async def process(index, batch):
        name = labels[index]
        source = json.dumps(batch, ensure_ascii=False, sort_keys=True)
        async with semaphore:
            log(f"Annotating {name}: {len(batch)} reports")
            draft = await agent(
                RUBRIC + f"\nBatch {name}. Save as {name}-draft.jsonl.\nREPORTS:\n" + source,
                phase="annotate", schema=SCHEMA, label=name,
            )
        async with semaphore:
            log(f"Reviewing {name}")
            reviewed = await agent(
                RUBRIC
                + "\nYou are the independent REVIEWER. First read each source report and "
                "reason through the 12 targets, then check the attached draft. Correct "
                "all missed positives, false positives, incorrect negations, unsupported "
                "negatives, and fabricated quotes. Do not just validate its schema: "
                "re-read EVERY source report. Preserve a study only after checking it. "
                "Return the complete corrected JSONL, not a patch or summary.\n"
                + f"Batch {name}. Save as {name}-reviewed.jsonl.\n"
                + f'ATTACHMENT:"{draft["artifact_url"]}"\nREPORTS:\n'
                + source,
                phase="review", schema=SCHEMA, label=name,
            )
        if reviewed["n_studies"] != len(batch):
            raise ValueError(f"{name}: wrong review count {reviewed['n_studies']}")
        log(f"Finished {name}")
        return {"batch": name, "draft": draft, "reviewed": reviewed}

    log(f"Starting {len(batches)} batches; six agents maximum concurrently")
    results = await asyncio.gather(
        *(process(index, batch) for index, batch in enumerate(batches)),
        return_exceptions=True,
    )
    successful = [result for result in results if not isinstance(result, BaseException)]
    failed = [
        {"batch": labels[index], "error": str(result)}
        for index, result in enumerate(results) if isinstance(result, BaseException)
    ]
    OUTPUT.write_text(json.dumps({"results": successful, "failed": failed}, indent=2))
    log(f"Recorded {len(successful)} reviewed batches and {len(failed)} failures")
    if failed:
        raise RuntimeError("Some batches failed; resume this run rather than omitting studies")


asyncio.run(main())
