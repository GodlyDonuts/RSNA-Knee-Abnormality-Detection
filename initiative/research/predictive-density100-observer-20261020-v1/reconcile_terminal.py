"""Reconcile the completed original pilot without restarting its stopped observer.

The pilot binds the original decision bytes but writes the parsed decision using
its explicit sorted serializer. Check both exact serializations, not their hash
equality. Original numerical acceptance and all other collector checks remain.
"""

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = "2488a4086fc854af733c7accf84377ff69f9f6fc"
PILOT = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
ORIGINAL_DECISION_SHA = "81f4467be25e091fe1546204a587a04a81d19ccd4c00c8df4c086ec743055e15"
WRITTEN_DECISION_SHA = "580312fa2eb2db0dea7d1e9f0701492e604b6517a4f20a90b0f3ee80d0211605"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def exact_serializations(source, stored, bound_sha):
    assert sha(source) == bound_sha == ORIGINAL_DECISION_SHA
    expected = (json.dumps(json.loads(source), sort_keys=True, allow_nan=False) + "\n").encode()
    assert stored == expected and sha(stored) == WRITTEN_DECISION_SHA
    return {"original_bound_sha256": bound_sha, "original_stored_sha256": sha(stored),
            "exact_original_serializer_reproduced": True, "original_decision_unchanged": True}


def git_file(pin, name):
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=REPO).decode()


def program():
    import ast
    import inspect

    rel = HERE.relative_to(REPO).as_posix()
    original = git_file(SOURCE, rel + "/collect.py")
    contract = git_file(SOURCE, rel + "/contract.py")
    decision = git_file(PILOT, "initiative/research/predictive-density100-pilot-20261020-v1/decision.json")
    assert sha(decision.encode()) == ORIGINAL_DECISION_SHA
    old = '''        c.require(c.sha(read(root / "scientific-decision.json")) ==
                  value["scientific_decision_sha256"], "Original scientific decision differs")'''
    new = '''        out["scientific_decision_serialization"] = exact_serializations(
            original_decision.encode(), read(root / "scientific-decision.json"),
            value["scientific_decision_sha256"])'''
    assert original.count(old) == 1
    source = original.replace(old, new, 1)
    ast.parse(source)
    helpers = inspect.getsource(sha) + "\n" + inspect.getsource(exact_serializations)
    return ("import sys,types,hashlib,json\n"
            f"ORIGINAL_DECISION_SHA={ORIGINAL_DECISION_SHA!r}\n"
            f"WRITTEN_DECISION_SHA={WRITTEN_DECISION_SHA!r}\n"
            f"original_decision={decision!r}\n" + helpers +
            "\nmodule=types.ModuleType('observer_contract')\n"
            f"exec(compile({contract!r},'original_contract','exec'),module.__dict__)\n"
            "sys.modules['observer_contract']=module\n" + source)


def main():
    os.umask(0o077)
    old = json.loads((REPO / "artifacts" / HERE.name / "result.json").read_bytes())
    assert old["status"] == "observer_stopped_requires_reconciliation"
    assert subprocess.run(["ps", "-p", "15130"], capture_output=True).returncode != 0
    journal = REPO / "artifacts" / (HERE.name + "-terminal-reconciliation")
    journal.mkdir(mode=0o700, exist_ok=False)
    source = program()
    def once(path, value):
        with path.open("x") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    once(journal / "intent.json", {"GPU_job": "981620", "failed_reader": "985908",
         "original_observer_PID": 15130, "program_sha256": sha(source.encode()),
         "remote_writes": 0, "automatic_retry": False})
    command = ["srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
        "--mem=1G", "--time=00:03:00", "--immediate=60", "--job-name=rsna-C100-reconcile-read",
        "env", "CUDA_VISIBLE_DEVICES=", "TZ=UTC", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python", "-B", "-c",
        "import sys;exec(compile(sys.stdin.read(),'C100_reconcile_readonly','exec'))"]
    result = subprocess.run(["ssh", "-S", "/tmp/rsna-research-20261001-stable.sock", "-o", "BatchMode=yes",
        "newton", shlex.join(command)], input=source, capture_output=True, text=True, timeout=300)
    once(journal / "returned.private.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, "Retain original failed collection; inspect without replay"
    spec = importlib.util.spec_from_file_location("C100_original_observer", HERE / "observe.py")
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    value = json.loads(result.stdout)
    body = observer.collected(value)
    assert body is not None
    with (HERE / "actual-accepted.json").open("xb") as stream:
        stream.write(body)
    terminal = {k:v for k,v in value.items() if k != "acceptance_body"}
    terminal.update(original_observer_stopped=True, failed_reader_job="985908",
                    reconciliation_program_sha256=sha(source.encode()))
    once(HERE / "actual-terminal.json", terminal)
    once(journal / "result.json", terminal)
    print(json.dumps(terminal, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
