"""Continue existing final artifacts after an actual controller response-name bug.

No training, re-upload, repeated qualification compile, or changed serving code.
The original single-phase coordinator retains all account mutations and budgets.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "1dc078f7f5b6764d04231e4f7e2f80b275bc5bca"
EXPECTED_SOURCE = "fe364a3c8a9b7ea2d0edd0ca6678302255aed8f83e709c3254af6172d0fe60c6"
DATASET = "godlydonuts/rsna-predictive-c100-final-3d49d387375a"
DATASET_ID = 12762386


def read(path):
    return json.loads(path.read_text())


def source(pin, path):
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT, text=True
    )


def load(body, path, name):
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    m.__file__ = str(path)
    exec(compile(body, str(path), "exec"), m.__dict__)
    return m


def replace(body, old, new):
    assert body.count(old) == 1, "Exact original controller anchor required"
    return body.replace(old, new)


def phase_reader(prior, role):
    def result(phase, mode, actual):
        if phase != "compile" or actual.get("status") in {
            "waiting_existing_canonical_claim_no_mutation",
            "waiting_live_capacity_budget_or_allowance_no_mutation",
        }:
            return prior(phase, mode, actual)
        status, branch = (
            ("C100_final_offline_source_compiled_CPU_only", "C100")
            if role == "Newton"
            else ("Colab_C100_final_offline_source_compiled_CPU_only", "Colab-C100-seed20261021")
        )
        assert mode in ("qualification", "C")
        assert actual.get("status") == status and actual.get("branch") == branch
        assert actual.get("mode") == ("qualification" if mode == "qualification" else "competition")
        assert actual.get("new_Dataset_notebook_or_entry_mutations") == 0
        assert actual.get("actual_trained_GPU_execution") is False
        assert isinstance(actual.get("source_sha256"), str) and len(actual["source_sha256"]) == 64
        assert all(c in "0123456789abcdef" for c in actual["source_sha256"])
        return "done"

    return result


def newton_body(body):
    start = body.index("        def run(self):\n")
    end = body.index('            self.action("push")\n', start)
    removed = body[start:end]
    for action in ("stage", "prepare", "dataset", "dataset-reconcile", "compile"):
        assert removed.count(f'self.action("{action}")') == 1
    return body[:start] + "        def run(self):\n            _admit_existing(self)\n" + body[end:]


BOUNDARY = """    import hashlib, sqlite3
    prepared=json.loads((root/'prepared.json').read_text())
    verified=json.loads((root/'dataset-verified.json').read_text())
    compiled=json.loads((root/'compiled-qualification/compiled.json').read_text())
    assert verified['id']==12762386 and verified['private'] and verified['version']==1
    assert verified['downloaded_all_bytes'] and verified['files']==prepared['files']
    assert compiled['bundle_identity']==prepared['bundle_identity']
    assert hashlib.sha256((root/'compiled-qualification/source.private.py').read_bytes()).hexdigest()==compiled['source_sha256']=='fe364a3c8a9b7ea2d0edd0ca6678302255aed8f83e709c3254af6172d0fe60c6'
    assert not (root/'compiled-qualification/push-intent.json').exists()
    assert not (root/'compiled-C').exists()
    assert not result['canonical_claim_present']
    key='licensed-parent-full-epoch-serving:predictive-C100-final-v1:'+prepared['bundle_identity']
    with sqlite3.connect('file:/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3?mode=ro',uri=True) as db:
        row=json.loads(db.execute('SELECT value FROM meta WHERE key=?',(key+':dataset',)).fetchone()[0])
        assert row['state']=='predictive_dataset_fully_verified' and row['verified']==verified
        for suffix in (':qualification',':C'):
            assert db.execute('SELECT value FROM meta WHERE key=?',(key+suffix,)).fetchone() is None
    result['resume_compiled']=compiled
    result['resume_verified']=verified
    result['no_prior_push_or_submission']=True
"""


def make_owner(role, journal):
    fast_path = HERE.parent / f"predictive-fast-{role.lower()}-deployment-20261021-v1"
    run_once = load(
        source(PIN, fast_path / "run_once.py"), fast_path / "run_once.py", "frozen_phase"
    )
    saved = sys.modules.get("run_once")
    sys.modules["run_once"] = run_once
    try:
        fast = load(source(PIN, fast_path / "owner.py"), fast_path / "owner.py", "frozen_handoff")
    finally:
        if saved is None:
            sys.modules.pop("run_once", None)
        else:
            sys.modules["run_once"] = saved
    original_load = fast.load

    def adjusted_load(body, path, name):
        if role == "Newton":
            body = newton_body(body)
        else:
            body = replace(
                body,
                'paths = [JOURNAL / h.FILENAME, Path("/Users/sairamen/Downloads") / h.FILENAME]',
                'paths = [ORIGINAL_JOURNAL / h.FILENAME, JOURNAL / h.FILENAME, Path("/Users/sairamen/Downloads") / h.FILENAME]',
            )
        return original_load(body, path, name)

    fast.load = adjusted_load
    m = fast.implementation()
    old = ROOT / "artifacts" / fast_path.name / "owner"
    m.JOURNAL = journal
    m.ORIGINAL_JOURNAL = old
    factory_name = "original" if role == "Newton" else "base"
    factory = getattr(m, factory_name)

    def base():
        answer = factory()
        b = answer if role == "Newton" else answer[1]
        b.phase_result = phase_reader(b.phase_result, role)
        return answer

    setattr(m, factory_name, base)

    def admit_existing(owner):
        assert (
            read(old / "stopped.json")["message"]
            == "Unexpected phase response; reconcile, never retry"
        )
        transport = read(old / "transport-00005.json")
        assert transport["returncode"] == 0 and transport["phase"] == "compile"
        compiled = json.loads(transport["stdout"])
        phase_reader(lambda *_: None, "Newton")("compile", "qualification", compiled)
        assert compiled["source_sha256"] == EXPECTED_SOURCE
        assert not list(old.glob("push-*-intent.json")) and not list(
            old.glob("submit-*-intent.json")
        )
        config = read(old / "phase-00005.private.json")
        owner.evidence = config["C100_final_evidence"]
        assert owner.evidence == read(old / "actual-final-evidence.json")
        owner.contract_options = config["C100_contract_options"]
        original_observer = owner.observer
        owner.observer = replace(
            original_observer,
            "    return result\n\n\nif __name__",
            BOUNDARY + "    return result\n\n\nif __name__",
        )
        observed = owner.observe(DATASET)
        assert observed["dataset_ready"] and observed["dataset_id"] == DATASET_ID
        assert observed["resume_compiled"] == compiled and observed["no_prior_push_or_submission"]
        proof = read(old / "result-00004.json")
        assert observed["resume_verified"] == proof
        owner.base.once(journal / "existing-compiled-boundary-admitted.json", observed)
        owner.observer = original_observer

    m._admit_existing = admit_existing
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("Newton", "Colab"), required=True)
    p.add_argument("--source-git", required=True)
    p.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = p.parse_args()
    assert source(args.source_git, Path(__file__)) == Path(__file__).read_text()
    os.umask(0o077)
    pid = 3332 if args.role == "Newton" else 3333
    assert not subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True
    ).stdout.strip()
    old = ROOT / "artifacts" / f"predictive-fast-{args.role.lower()}-deployment-20261021-v1/owner"
    assert read(old / "owner-intent.json")["source_git"] == PIN
    if args.role == "Colab":
        assert read(old / "explicit-waiting-controller-stop.json")["PID"] == pid
        assert not list(old.glob("phase-*.private.json"))
        assert not list(old.parent.glob("*-intent.json"))
        assert not (old / "actual-original-final-evidence.json").exists()
    journal = ROOT / "artifacts" / HERE.name / args.role.lower()
    journal.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (journal / "continuation-source.json").open("x") as f:
        json.dump(
            {
                "source_git": args.source_git,
                "original_source_git": PIN,
                "role": args.role,
                "original_PID": pid,
                "PID": os.getpid(),
                "same_remote_root": True,
                "same_model_serving_and_budget": True,
                "automatic_mutation_retry": False,
            },
            f,
        )
    m = make_owner(args.role, journal)
    sys.argv = [str(m.HERE / "owner.py"), "--source-git", PIN, "--socket", args.socket]
    m.main()


if __name__ == "__main__":
    main()
