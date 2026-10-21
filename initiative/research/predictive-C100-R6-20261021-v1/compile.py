"""A separate C100/R6 candidate built from exact, already executed child sources.

Pure preparation only. Original B/C/C0 conditional experiment stays unchanged.
No remote source execution, account operation, GPU run or entry here.
"""

import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "predictive-R6-additive-20261020-v1"
C100 = "055a8f8ed7e0d64f47537fb25480a0f08c7ac947ec8f0472e22f7dcdc117d6e9"
R6 = "affa44363c530eed37878d083707d4fc50dcf1571d8dfcce12fd1c5a7d4bc581"
MODEL = "e606d974f4aa8bcc2a4e07375353e41f1e4d31d7151f7d223380624c8116d8d6"
PUBLIC_CSV = "5c42889e95cefe7003d2b4bc3629794c1ef4fcbf607bc99fc68579b402db327e"
PUBLIC_ROSTER = "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b"


def digest(b):
    return hashlib.sha256(b).hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replace(s, old, new):
    require(s.count(old) == 1, "Exact frozen parent source anchor required")
    return s.replace(old, new)


def supervisor_source(prior):
    # Keep physical Kaggle memory limits and original public qualification, but
    # do not reintroduce the old rule that discards a complete hidden CSV merely
    # because historical RSS exceeded the tighter qualification threshold.
    s = replace(
        prior,
        '        and receipt["models"][0]["memory_accepted"] is True,',
        '        and (roster_sha256 != PUBLIC_ROSTER or receipt["models"][0]["memory_accepted"] is True),',
    )
    s = replace(s, "    stages = []\n", "    stages = []\n    watchdog_seconds = TOTAL_SECONDS\n")
    s = replace(
        s,
        "        seconds = TOTAL_SECONDS - (time.monotonic() - started)",
        "        seconds = watchdog_seconds - (time.monotonic() - started)",
    )
    s = replace(
        s,
        "        roster_sha = digest(canonical(uids))",
        """        roster_sha = digest(canonical(uids))
        if roster_sha == PUBLIC_ROSTER:
            watchdog_seconds = binding["public_watchdog_seconds"]
            require(watchdog_seconds == 900, "Bounded actual public experiment required")
            timer.cancel()
            timer = threading.Timer(max(0.001, remaining()), expire)
            timer.daemon = True
            timer.start()""",
    )
    s = replace(
        s,
        '            "runtime_guarantee": False,',
        '            "effective_watchdog_seconds": watchdog_seconds,\n            "runtime_guarantee": False,',
    )
    ast.parse(s)
    return s


def build(c_source, r6_source, qualification, public, *, supervisor_text, fusion_text):
    require(
        digest(c_source) == C100 and digest(r6_source) == R6,
        "Exact already-executed C100 and complete scored R6 sources required",
    )
    tree = ast.parse(c_source)
    bindings = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "BINDING"
    ]
    require(len(bindings) == 1, "One original source binding required")
    original = bindings[0]
    require(
        original["mode"] == "competition"
        and original["branch"] == "C100"
        and original["qualification"] == qualification
        and len(original["models"]) == 1,
        "Original C100 training and actual qualification binding required",
    )
    chosen = original["models"][0]
    require(
        chosen["model"]["sha256"] == MODEL and chosen["architecture"]["context_cap"] == 24,
        "Original fixed-final C100 model and training context required",
    )
    require(
        qualification["status"] == "actual_trained_predictive_public_T4_qualification_accepted"
        and len(qualification["models"]) == 1
        and qualification["models"][0]["trained_export_sha256"] == MODEL
        and qualification["models"][0]["candidate_sha256"] == digest(canonical(chosen))
        and qualification["models"][0]["precision_accepted"] is True
        and qualification["models"][0]["memory_accepted"] is True,
        "Original trained T4 qualification required",
    )
    require(
        public["status"] == "actual_predictive_exact_test_roster_submission_created"
        and public["mode"] == "competition"
        and public["actual_test_studies"] == 3
        and public["test_roster_sha256"] == PUBLIC_ROSTER
        and public["CSV_sha256"] == PUBLIC_CSV
        and public["submission_csv_absent"] is False
        and len(public["models"]) == 1
        and public["models"][0]["trained_export_sha256"] == MODEL
        and public["models"][0]["candidate_sha256"] == digest(canonical(chosen))
        and public["models"][0]["prediction_sha256"]
        == qualification["models"][0]["prediction_sha256"],
        "Actual standalone C100 complete public prediction proof required",
    )
    runtime = {k: qualification[k] for k in ("torch", "timm", "pydicom", "pinned_Docker")}
    serving = {
        k: qualification[k]
        for k in (
            "serving_worker_sha256",
            "serving_contract_sha256",
            "serving_helpers_sha256",
            "model_dataset_ref",
        )
    }
    require(
        all(public[k] == v for k, v in {**runtime, **serving}.items()),
        "Unchanged standalone runtime and serving source required",
    )
    require(
        all(original[k] == v for k, v in serving.items()), "Exact child source binding required"
    )
    payload = {
        "C": c_source,
        "R6": r6_source,
        "fusion": fusion_text.encode(),
        "supervisor": supervisor_source(supervisor_text).encode(),
    }
    binding = dict(
        schema="independent_C100_complete_R6_fixed_rank_v1",
        mode="candidate",
        source_sha256={k: digest(v) for k, v in payload.items()},
        C_model_sha256=MODEL,
        C_candidate_sha256=digest(canonical(chosen)),
        C_runtime=runtime,
        C_serving=serving,
        trained_qualification_sha256=digest(canonical(qualification)),
        standalone_C_acceptance_sha256=digest(canonical(public)),
        standalone_C_public_CSV_sha256=PUBLIC_CSV,
        total_watchdog_seconds=32400,
        public_watchdog_seconds=900,
        fixed_reference_weight=0.75,
        fixed_candidate_weight=0.25,
        internet=False,
        standalone_submission=57491555,
        minimum_standalone_score=0.949,
        standalone_score_known=False,
        new_competition_entry=False,
    )
    encoded = {k: base64.b64encode(v).decode() for k, v in payload.items()}
    source = (
        "# PRIVATE licensed sources: never commit this compiled file.\n"
        "import base64,hashlib,json,os\nos.umask(0o077)\n"
        f"_binding=json.loads({json.dumps(binding, sort_keys=True)!r})\n"
        f"_payload={encoded!r}\n"
        "_source={k:base64.b64decode(v,validate=True) for k,v in _payload.items()}\n"
        "assert all(hashlib.sha256(v).hexdigest()==_binding['source_sha256'][k] for k,v in _source.items())\n"
        "_namespace={'__name__':'_private_C100_R6_supervisor'}\n"
        "exec(compile(_source['supervisor'],'C100_R6_supervisor','exec'),_namespace)\n"
        "_namespace['run'](_source['C'],_source['R6'],_source['fusion'],_binding)\n"
    )
    ast.parse(source)
    return source, dict(
        status="C100_R6_prepared_not_executed",
        source_sha256=digest(source.encode()),
        source_bytes=len(source.encode()),
        binding=binding,
        actual_combined_GPU_run=False,
        competition_entry=False,
        new_Dataset=False,
    )


def score_trigger(submission):
    """Fixed before C100 outcome; a successful score is necessary, not sufficient."""
    import math

    if submission.get("ref") != 57491555:
        return False
    try:
        score = float(submission["publicScore"])
    except (ValueError, TypeError, KeyError):
        return False
    return (
        submission.get("status") == "COMPLETE"
        and not submission.get("errorDescription")
        and math.isfinite(score)
        and 0.949 <= score <= 1
    )
