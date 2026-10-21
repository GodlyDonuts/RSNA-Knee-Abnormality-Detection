"""Pure private compilation of two actually qualified, fixed-final C100 children.

Inputs must be collected from the original serving owners on private storage.
This module does not allocate, copy weights, publish, or submit anything.
"""

import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEWTON_SOURCE = "055a8f8ed7e0d64f47537fb25480a0f08c7ac947ec8f0472e22f7dcdc117d6e9"
NEWTON_MODEL = "e606d974f4aa8bcc2a4e07375353e41f1e4d31d7151f7d223380624c8116d8d6"
BRANCHES = {"newton": "C100", "colab": "Colab-C100-seed20261021"}
PUBLIC_ROSTER = "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b"
FIELDS = (
    "torch",
    "timm",
    "pydicom",
    "pinned_Docker",
    "serving_worker_sha256",
    "serving_contract_sha256",
    "serving_helpers_sha256",
    "model_dataset_ref",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(name):
    spec = importlib.util.spec_from_file_location("two_seed_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child(role, item):
    raw, qualification, public = item["source"], item["qualification"], item["public"]
    require(digest(raw) == item["actual_source_sha256"], "Actual child source differs")
    if role == "newton":
        require(digest(raw) == NEWTON_SOURCE, "Exact original Newton source required")
    tree = ast.parse(raw)
    assignments = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "BINDING"
    ]
    require(len(assignments) == 1, "One exact original child binding required")
    binding = assignments[0]
    require(
        binding["mode"] == "competition"
        and binding["branch"] == BRANCHES[role]
        and binding["qualification"] == qualification
        and len(binding["models"]) == 1,
        "Actual standalone final qualification binding required",
    )
    chosen = binding["models"][0]
    require(
        chosen["arm"] == "C"
        and chosen["architecture"]["context_cap"] == 24
        and chosen["provenance"]["epoch"] == 6
        and chosen["provenance"]["completed_updates"] == 2916,
        "Complete fixed-final 24-context child required",
    )
    if role == "newton":
        require(
            chosen["model"]["sha256"] == NEWTON_MODEL, "Exact actual Newton final model required"
        )
    for receipt, status in (
        (qualification, "actual_trained_predictive_public_T4_qualification_accepted"),
        (public, "actual_predictive_exact_test_roster_submission_created"),
    ):
        require(
            receipt["status"] == status and len(receipt["models"]) == 1,
            "Actual standalone qualification and public receipts required",
        )
        model = receipt["models"][0]
        require(
            model["trained_export_sha256"] == chosen["model"]["sha256"]
            and model["candidate_sha256"] == digest(canonical(chosen))
            and model["memory_accepted"] is True
            and model["precision_accepted"] is True,
            "Actual complete trained child evidence differs",
        )
        require(all(receipt[k] == qualification[k] for k in FIELDS), "Standalone runtime differs")
    require(
        public["mode"] == "competition"
        and public["actual_test_studies"] == 3
        and public["test_roster_sha256"] == PUBLIC_ROSTER
        and public["submission_csv_absent"] is False
        and public["CSV_sha256"] == digest(item["public_csv"])
        and public["CSV_bytes"] == len(item["public_csv"]),
        "Original public CSV bytes required",
    )
    reader = csv.reader(io.StringIO(item["public_csv"].decode(), newline=""), strict=True)
    next(reader)
    uids = [row[0] for row in reader]
    require(digest(canonical(uids)) == PUBLIC_ROSTER, "Original public reference roster differs")
    load("fusion")._read_csv(item["public_csv"], tuple(uids))
    require(
        all(binding[k] == qualification[k] for k in FIELDS[4:]), "Source serving binding differs"
    )
    return dict(
        branch=BRANCHES[role],
        model_sha256=chosen["model"]["sha256"],
        candidate_sha256=digest(canonical(chosen)),
        runtime_and_serving={k: public[k] for k in FIELDS},
        public_CSV_sha256=public["CSV_sha256"],
        qualification_sha256=digest(canonical(qualification)),
        public_receipt_sha256=digest(canonical(public)),
        provenance_sha256=digest(canonical(chosen)),
    )


def build(items):
    require(set(items) == set(BRANCHES), "Two original standalone owners required")
    children = {role: child(role, items[role]) for role in BRANCHES}
    require(
        children["newton"]["model_sha256"] != children["colab"]["model_sha256"],
        "Independent full learned states required",
    )
    require(
        all(
            children["newton"]["runtime_and_serving"][k]
            == children["colab"]["runtime_and_serving"][k]
            for k in FIELDS[:4]
        ),
        "Same accepted runtime and Docker required",
    )
    payload = {role: items[role]["source"] for role in BRANCHES}
    payload.update(
        {name: (HERE / (name + ".py")).read_bytes() for name in ("fusion", "supervisor")}
    )
    binding = dict(
        schema="complete_independent_C100_probability_mean_v1",
        mode="candidate",
        children=children,
        source_sha256={k: digest(v) for k, v in payload.items()},
        total_watchdog_seconds=32400,
        public_watchdog_seconds=900,
        weights=[0.5, 0.5],
        internet=False,
        new_models_or_Datasets=False,
    )
    references = {role: items[role]["public_csv"] for role in BRANCHES}
    source = (
        "# PRIVATE original sources and public reference predictions; never commit this file.\n"
        "import base64,hashlib,json,os\nos.umask(0o077)\n"
        f"_binding=json.loads({json.dumps(binding, sort_keys=True)!r})\n"
        f"_encoded={ {k: base64.b64encode(v).decode() for k, v in payload.items()}!r}\n"
        f"_references={ {k: base64.b64encode(v).decode() for k, v in references.items()}!r}\n"
        "_payload={k:base64.b64decode(v,validate=True) for k,v in _encoded.items()}\n"
        "assert all(hashlib.sha256(v).hexdigest()==_binding['source_sha256'][k] for k,v in _payload.items())\n"
        "_refs={k:base64.b64decode(v,validate=True) for k,v in _references.items()}\n"
        "_namespace={'__name__':'_two_seed_supervisor'}\n"
        "exec(compile(_payload['supervisor'],'two_seed_supervisor','exec'),_namespace)\n"
        "_namespace['run']({k:_payload[k] for k in ('newton','colab')},_payload['fusion'],_refs,_binding)\n"
    )
    ast.parse(source)
    return source, dict(
        status="two_seed_source_prepared_not_executed",
        binding=binding,
        source_sha256=digest(source.encode()),
        source_bytes=len(source.encode()),
        actual_combined_GPU_run=False,
        competition_entry=False,
        accuracy_claim=False,
    )
