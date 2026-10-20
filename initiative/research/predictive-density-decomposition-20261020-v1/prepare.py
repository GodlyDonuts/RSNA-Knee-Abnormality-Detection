"""Local-only capsule builder. Returns source/payload; never calls SSH or a scheduler."""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOADER_PIN = "7b277c4e75a3ee25cdf5247b3dfad93eb5f9eefa"
LOADER_SHA = "36a87becf4a79aeda7f6c491c82ff58c69b779902348f231648b5f00bec3aac0"
PRIOR_SHA = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
ENGINEERING_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"


def digest(body):
    return hashlib.sha256(body).hexdigest()


def git_file(pin, path):
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Exact committed source required")
    return subprocess.check_output(["git", "show", pin + ":" + str(path.relative_to(ROOT))],
                                   cwd=ROOT)


def prepare(source_git, c100_reference):
    """Caller supplies actual post-completion stat/terminal/acceptance/scheduler proof."""
    reference = dict(c100_reference)
    required = {"acceptance_sha256", "checkpoint_stat", "terminal_sha256", "scheduler"}
    if set(reference) != required:
        raise ValueError("Exact actual C100 final-reference schema required")
    for name in ("acceptance_sha256", "terminal_sha256"):
        value = reference[name]
        if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
            raise ValueError("Exact actual C100 SHA required")
    st = reference["checkpoint_stat"]
    if not isinstance(st, list) or len(st) != 4 or any(type(x) is not int or x < 1 for x in st):
        raise ValueError("Exact original checkpoint stat required")
    scheduler = reference["scheduler"]
    if any(scheduler.get(k) != v for k, v in {
        "GPU_job": "981620", "state": "COMPLETED", "exit_code": "0:0", "node": "evc27",
    }.items()):
        raise ValueError("Pending/failed pilot cannot admit a matched comparison")
    capsule = _base_capsule(source_git)
    capsule["payload"]["C100_reference"] = reference
    return capsule


def prepare_baseline(source_git):
    """Explicit C24-only capsule; no C100 reference or implied matched comparison."""
    capsule = _base_capsule(source_git)
    capsule["payload"]["execution_mode"] = "C24_only"
    capsule["execution"]["comparison_scope"] = "C24_only_no_C100_model_loaded"
    return capsule


def _base_capsule(source_git):
    bodies = {name: git_file(source_git, HERE / name) for name in ("diagnostic.py", "prepare.py")}
    if any(body != (HERE / name).read_bytes() for name, body in bodies.items()):
        raise ValueError("Freeze exact local diagnostic before preparing a capsule")
    prior_dir = HERE.parent / "predictive-objective-balance-20261020-v1"
    loader = git_file(LOADER_PIN, prior_dir / "diagnostic.py")
    prior = (prior_dir / "actual-result.json").read_bytes()
    engineering = (HERE.parent / "predictive-stable-warmup-20261020-v1/actual-accepted.json").read_bytes()
    if (digest(loader), digest(prior), digest(engineering)) != (
        LOADER_SHA, PRIOR_SHA, ENGINEERING_SHA,
    ):
        raise ValueError("Original loader/input/checkpoint evidence differs")
    payload = {
        "source_git": source_git, "diagnostic_sha256": digest(bodies["diagnostic.py"]),
        "original_loader_source": loader.decode(), "prior_result_text": prior.decode(),
        "original_engineering_receipt_sha256": ENGINEERING_SHA,
        "original_engineering_receipt": json.loads(engineering), "fitting_indices": [0, 1],
    }
    return {
        "program": bodies["diagnostic.py"].decode(), "payload": payload,
        "source_sha256": {name: digest(body) for name, body in bodies.items()},
        "execution": {
            "node": "evc27", "CPUs": 1, "GPUs": 0, "RAM_GiB": 12, "maximum_seconds": 1800,
            "python": "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python",
            "stdin_only": True, "new_remote_files": 0, "new_shared_files": 0,
            "actual_execution_or_training_authority": False,
        },
    }
