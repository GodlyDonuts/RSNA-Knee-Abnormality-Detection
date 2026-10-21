"""Bind the Colab compiler to the existing canonical single-phase coordinator."""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PIN = "fb9e3adf2d4f2f873adf10733bb310e7afa73be8"
ORIGINAL = "initiative/research/predictive-final-deployment-20261020-v1/"
SERVING = HERE.parent / "colab-C100-final-serving-20261021-v1"
STORAGE = HERE.parent / "colab-private-kaggle-handoff-20261021-v1"
ROOT_PREFIX = "/tmp/rsna-predictive-final-deployment-20261020-v1-Colab-C100-"


def git(name, pin=PIN):
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=REPO, text=True)


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Pinned canonical source derivation point differs")
    return text.replace(old, new)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def facade():
    return """import importlib.util
import types
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("Colab_staged_compiler",HERE/"implementation.py")
implementation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
OPTIONS={"module_sha256":implementation.C.MODULE_SHA256}
adapter=implementation.C.Contract(implementation.originals()["contract.py"],**OPTIONS)
contract=types.SimpleNamespace(**{k:v for k,v in adapter._base.__dict__.items() if not k.startswith("__")})
contract.candidate=adapter.candidate
contract.validate_export=adapter.validate_export
contract.validate_qualification=adapter.validate_qualification
contract.check_predictions=adapter.check_predictions
def build(parent,entries,**kwargs):
    contract.require(len(entries)==1,"One actual independent Colab final required")
    return implementation.build(parent,entries[0],contract_options=OPTIONS,**kwargs)
"""


def build(config):
    compiler = load("Colab_deployment_original_compiler", SERVING / "compile.py")
    require = compiler.C.require
    require(
        config["owning_node"] == "evc27"
        and config["root"].startswith(ROOT_PREFIX)
        and ".." not in Path(config["root"]).parts,
        "Separate private evc27 deployment root required",
    )
    evidence = config["Colab_evidence"]
    terminal = json.loads(evidence["terminal_text"])
    require(
        terminal["status"] == "actual_full_C100_seed20261021_complete"
        and terminal["updates"] == 2916
        and terminal["epochs"] == 6
        and terminal["all_epochs_private_verified"] is True
        and len(evidence["epoch_publications"]) == 6,
        "Actual complete original Colab fit required before staging",
    )
    files = {n: git(ORIGINAL + n) for n in ("policy.py", "cpu.py", "account.py")}
    files["policy.py"] = once(files["policy.py"], 'ARMS = ("B", "C", "C0")', 'ARMS = ("C",)')
    files["policy.py"] = once(
        files["policy.py"],
        'FAMILY = "licensed-parent-full-epoch-serving:predictive-final-v1:"',
        'FAMILY = "licensed-parent-full-epoch-serving:Colab-C100-final-v1:"',
    )
    files["policy.py"] = once(
        files["policy.py"],
        '"godlydonuts/rsna-predictive-" + mode.lower()',
        '"godlydonuts/rsna-colab-c100-" + mode.lower()',
    )
    budget = HERE.parent / "predictive-density100-deployment-20261021-v1/primary_budget.py"
    files["policy.py"] += "\n" + budget.read_text()

    # Preserve original stage/funding/immutable-control checks; replace the
    # Slurm-final/ZIP preparation with actual existing Colab-Dataset evidence.
    cpu_source = files["cpu.py"]
    start = cpu_source.index('    if phase == "prepare":\n')
    end = cpu_source.index("\n    funding(v, root)\n    return result", start)
    cpu_source = (
        cpu_source[:start]
        + """    helper = load("Colab_private_preparation", root / "control/prepare.py")
    if phase == "prepare":
        result = helper.prepare(sys.modules[__name__], v, root, compiler)
    elif phase == "compile":
        result = helper.compile_final(sys.modules[__name__], v, root, compiler)
    else:
        raise ValueError("Explicit Colab stage, prepare or compile only")
"""
        + cpu_source[end:]
    )
    # runpy's __main__ namespace is not guaranteed to be sys.modules['__main__'].
    cpu_source = cpu_source.replace("sys.modules[__name__]", "types.SimpleNamespace(**globals())")
    files["cpu.py"] = once(cpu_source, "import sys\n", "import sys\nimport types\n")

    account = files["account.py"]
    account = once(
        account,
        'phase in ("dataset", "dataset-reconcile", "push", "status", "submit", "outcome")',
        'phase in ("push", "status", "submit", "outcome")',
    )
    start = account.index('            if phase.startswith("dataset"):\n')
    end = account.index('            dataset = cpu.read(root / "dataset-verified.json")', start)
    account = (
        account[:start]
        + """            adoption = cpu.load("Colab_canonical_adoption", root / "control/adopt.py")
            adoption.require_adopted(cpu, root, prepared, db)

"""
        + account[end:]
    )
    account = once(
        account,
        'dataset["status"] == "all_payload_bytes_verified"',
        'dataset["status"] == "original_Colab_bytes_verified_existing_Dataset_admitted"',
    )
    account = once(
        account,
        'dataset["downloaded_all_bytes"] is True',
        'dataset["original_Colab_downloaded_all_bytes"] is True',
    )
    account = once(
        account,
        'kind="predictive_final_deployment_v1"',
        'kind="independent_Colab_C100_final_deployment_v1"',
    )
    account = once(
        account,
        "                actual_budget = p.budget(\n",
        "                plan = p.protect_primary_budget(plan, db)\n                actual_budget = p.budget(\n",
    )
    files["account.py"] = account
    files.update({n: (HERE / n).read_text() for n in ("prepare.py", "adopt.py")})
    lease = json.loads((STORAGE / "actual-A10040-checkpoint-lease.json").read_text())["exact_lease"]
    files["original-Colab-lease.json"] = json.dumps(lease, sort_keys=True)
    original_storage = (STORAGE / "colab_storage.py").read_text()
    require(
        compiler.C.digest(original_storage) == compiler.C.CODE["colab_storage.py"],
        "Original checked Colab download implementation required",
    )
    files["colab_storage.py"] = original_storage
    files.update({"serving/" + n: body for n, body in compiler.originals().items()})
    files["serving/Colab_contract.py"] = (SERVING / "contract.py").read_text()
    implementation = once(
        (SERVING / "compile.py").read_text(),
        'ORIGINAL = HERE.parent / "predictive-final-serving-20261020-v1"',
        "ORIGINAL = HERE",
    )
    implementation = once(
        implementation,
        '"Colab_C100_compile_contract", HERE / "contract.py"',
        '"Colab_C100_compile_contract", HERE / "Colab_contract.py"',
    )
    implementation = once(
        implementation,
        'adapter_source = (HERE / "contract.py").read_text()',
        'adapter_source = (HERE / "Colab_contract.py").read_text()',
    )
    files["serving/implementation.py"] = implementation
    files["serving/compile.py"] = facade()
    for name in (
        "workflow.py",
        "components.py",
        "publication.py",
        "gpu_capacity.py",
        "output_csv.py",
    ):
        files["common/" + name] = git("scripts/kaggle/" + name)
    for name, text in files.items():
        if name.endswith((".py", ".py.txt")):
            ast.parse(text, filename=name)
        else:
            json.loads(text)
    return files
