"""A separate C100 deployment capsule; the primary live owner stays unchanged.

This builder performs no account calls or dispatches. It adapts the original
single-phase deployment to the existing C100 compiler and original owning node.
"""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PIN = "fb9e3adf2d4f2f873adf10733bb310e7afa73be8"
ORIGINAL = "initiative/research/predictive-final-deployment-20261020-v1/"
SERVING = HERE.parent / "predictive-density100-serving-20261020-v1"
FULL_SOURCE = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
FULL_JOB = "995351"
EXTRA_KEYS = {"epoch_publications", "full_terminal", "full_scheduler"}


def git(name, pin=PIN):
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=REPO, text=True)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Original deployment derivation point differs")
    return text.replace(old, new)


def compiler():
    spec = importlib.util.spec_from_file_location(
        "C100_deployment_compiler", SERVING / "compile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def facade(options, evidence):
    return f"""import importlib.util
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("C100_capsule_implementation", HERE / "implementation.py")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
OPTIONS = {options!r}
EVIDENCE = {evidence!r}
adapter = implementation.C.Contract(implementation.originals()["contract.py"], **OPTIONS)
contract = types.SimpleNamespace(**{{k:v for k,v in adapter._base.__dict__.items() if not k.startswith("__")}})

def candidate(**entry):
    return adapter.candidate(**entry, **EVIDENCE)

contract.candidate = candidate
contract.validate_export = adapter.validate_export
contract.validate_qualification = adapter.validate_qualification
contract.check_predictions = adapter.check_predictions

def build(parent, entries, **kwargs):
    contract.require(len(entries) == 1, "One original C100 final is required")
    return implementation.build(parent, {{**entries[0], **EVIDENCE}}, contract_options=OPTIONS, **kwargs)
"""


def build(config):
    compiled = compiler()
    require = compiled.C.require
    options = json.loads(compiled.C.canonical(config["C100_contract_options"]))
    evidence = json.loads(compiled.C.canonical(config["C100_final_evidence"]))
    require(
        set(options) == compiled.OPTIONS and set(evidence) == EXTRA_KEYS,
        "Complete original C100 final and pilot bindings required",
    )
    adapter = compiled.C.Contract(compiled.originals()["contract.py"], **options)
    require(
        config["owning_node"] == "evc27"
        and set(config["original_runs"]) == {"C"}
        and config["original_runs"]["C"]["run_root"] == "/tmp/rsna-predictive-full-C100-20261020-v1"
        and config["root"].startswith("/tmp/rsna-predictive-final-deployment-20261020-v1-C100-"),
        "Deploy the original C100 final on its owning node",
    )
    terminal = evidence["full_terminal"]
    require(
        options["full_source_git"] == FULL_SOURCE
        and terminal["GPU_job"] == FULL_JOB
        and terminal["branch"] == "C100"
        and terminal["arm"] == "C"
        and terminal["epochs"] == 6
        and terminal["updates"] == 2916
        and terminal["source_git"] == options["full_source_git"]
        and terminal["density100_pilot_sha256"] == adapter.pilot_receipt_sha256
        and len(evidence["epoch_publications"]) == 6,
        "Actual original full C100 completion required before staging",
    )
    compiled.C.completed(evidence["full_scheduler"], FULL_JOB)
    files = {n: git(ORIGINAL + n) for n in ("policy.py", "cpu.py", "account.py")}
    files["policy.py"] = replace_once(
        files["policy.py"], 'ARMS = ("B", "C", "C0")', 'ARMS = ("C",)'
    )
    files["policy.py"] = replace_once(
        files["policy.py"],
        'FAMILY = "licensed-parent-full-epoch-serving:predictive-final-v1:"',
        'FAMILY = "licensed-parent-full-epoch-serving:predictive-C100-final-v1:"',
    )
    files["policy.py"] = replace_once(
        files["policy.py"],
        '"godlydonuts/rsna-predictive-" + mode.lower()',
        '"godlydonuts/rsna-predictive-c100-" + mode.lower()',
    )
    files["cpu.py"] = replace_once(
        files["cpu.py"], "All three actual finals required", "One actual C100 final required"
    )
    files["cpu.py"] = replace_once(
        files["cpu.py"],
        'status="actual_three_predictive_final_exports_prepared_CPU_only"',
        'status="actual_single_C100_final_export_prepared_CPU_only"',
    )
    files["cpu.py"] = replace_once(
        files["cpu.py"],
        '"godlydonuts/rsna-predictive-finals-"',
        '"godlydonuts/rsna-predictive-c100-final-"',
    )
    files["account.py"] = replace_once(
        files["account.py"],
        'kind="predictive_final_deployment_v1"',
        'kind="predictive_C100_final_deployment_v1"',
    )
    files["account.py"] = replace_once(
        files["account.py"], '"RSNA predictive final B C C0"', '"RSNA predictive C100 fixed final"'
    )
    files["policy.py"] += "\n" + (HERE / "primary_budget.py").read_text()
    files["account.py"] = replace_once(
        files["account.py"],
        "                actual_budget = p.budget(\n",
        "                plan = p.protect_primary_budget(plan, db)\n"
        "                actual_budget = p.budget(\n",
    )
    files.update({"serving/" + n: body for n, body in compiled.originals().items()})
    files["serving/C100_contract.py"] = (SERVING / "contract.py").read_text()
    implementation = replace_once(
        (SERVING / "compile.py").read_text(),
        'ORIGINAL = HERE.parent / "predictive-final-serving-20261020-v1"',
        "ORIGINAL = HERE",
    )
    implementation = replace_once(
        implementation,
        '"C100_compile_contract", HERE / "contract.py"',
        '"C100_compile_contract", HERE / "C100_contract.py"',
    )
    implementation = replace_once(
        implementation,
        'adapter_source = (HERE / "contract.py").read_text()',
        'adapter_source = (HERE / "C100_contract.py").read_text()',
    )
    files["serving/implementation.py"] = implementation
    files["serving/compile.py"] = facade(options, evidence)
    for name in (
        "workflow.py",
        "components.py",
        "publication.py",
        "gpu_capacity.py",
        "output_csv.py",
    ):
        files["common/" + name] = git("scripts/kaggle/" + name)
    for name, body in files.items():
        ast.parse(body, filename=name)
    return files
