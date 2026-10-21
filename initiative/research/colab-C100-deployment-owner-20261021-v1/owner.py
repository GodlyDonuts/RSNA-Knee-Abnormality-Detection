"""Finite independent Colab final consumer; existing Dataset, no extra upload."""

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEPLOY = HERE.parent / "colab-C100-deployment-20261021-v1"
C100 = HERE.parent / "predictive-C100-deployment-owner-20261021-v1"
PIN = "7bd57083f0aab360dddca0d3087f8a559f3fa82b"
JOURNAL = ROOT / "artifacts" / HERE.name


def load(body, path, name):
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    m.__file__ = str(path)
    exec(compile(body, str(path), "exec"), m.__dict__)
    return m


def frozen(pin, path):
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT
    ).decode()


def base():
    m = load(frozen(PIN, C100 / "owner.py"), C100 / "owner.py", "Colab_original_C100_handoff")
    b = m.original()
    b.DEPLOY = DEPLOY
    b.MUTATIONS = b.MUTATIONS | {"adopt"}
    prior = b.phase_result

    def result(phase, mode, actual):
        if actual.get("status") in b.WAITING:
            return "wait"
        if phase == "prepare":
            m.require(
                actual.get("status")
                == "actual_Colab_fixed_final_existing_Dataset_prepared_CPU_only",
                "Actual Colab preparation required",
            )
            return "done"
        if phase == "adopt":
            m.require(
                actual.get("status") == "original_Colab_bytes_verified_existing_Dataset_admitted"
                and actual["private"] is True
                and actual["version"] == 1
                and actual["original_Colab_downloaded_all_bytes"] is True
                and actual["full_optimizer_downloaded_again"] is False,
                "Actual original private Colab Dataset admission required",
            )
            return "done"
        return prior(phase, mode, actual)

    b.phase_result = result
    return m, b


def observer_body(m, frozen_fit, jobs):
    body = m.observer_body(frozen_fit, jobs)
    extra = """
    import sqlite3
    primary = Path('/tmp/rsna-predictive-final-deployment-20261020-v1-C100-deadline')
    submitted = False
    if (primary/'prepared.json').is_file():
        prepared = json.loads((primary/'prepared.json').read_text())
        key = 'licensed-parent-full-epoch-serving:predictive-C100-final-v1:' + prepared['bundle_identity'] + ':C'
        with sqlite3.connect('file:/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3?mode=ro',uri=True) as db:
            found = db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        if found:
            row = json.loads(found[0])
            model = prepared['models'][0]
            submitted = (row.get('state') == 'predictive_final_submitted'
                and row.get('mode') == 'C' and row.get('version') == 1
                and type(row.get('submission_id')) is int and row['submission_id'] > 0
                and row.get('bundle_identity') == prepared['bundle_identity']
                and row.get('model_files') == {model['model_filename']:model['model']})
    result['original_Newton_C100_submitted'] = submitted
"""
    return m.replace(body, "    return result\n\n\nif __name__", extra + "\n    return result\n\n\nif __name__")


def configuration(m, o, observed, mode):
    inherited = m.config(o, observed, mode, o.policy)
    # Preserve the separate C1002GiB scope in addition to all older obligations.
    inherited["funding"]["other_node_unpaid_bytes"] += 2 * 2**30
    o.policy.storage(
        inherited["funding"],
        node_free=observed["node_free_bytes"],
        shared_free=observed["shared_free_bytes"],
        shared_files=observed["shared_free_files"],
    )
    original = inherited.pop("original_runs")["C"]
    inherited.pop("C100_contract_options")
    inherited.pop("C100_final_evidence")
    inherited.update(license_path=original["license_path"], Colab_evidence=o.evidence)
    if not observed["original_Newton_C100_submitted"]:
        inherited["runtime_plan"]["other_reserved_VM_seconds"] += 1920
    return inherited


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-git", required=True)
    p.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    m, b = base()
    for name in ("owner.py", "handoff.py", "decision.json"):
        m.require(
            frozen(args.source_git, HERE / name) == (HERE / name).read_text(),
            "Execute committed Colab handoff",
        )
    h = load(
        frozen(args.source_git, HERE / "handoff.py"),
        HERE / "handoff.py",
        "Colab_final_evidence_reader",
    )
    contract_path = HERE.parent / "colab-C100-final-serving-20261021-v1/contract.py"
    contract = load(
        frozen(args.source_git, contract_path), contract_path, "Colab_final_original_contract"
    )
    c = m.training()
    fit_args = SimpleNamespace(source_git=m.SOURCE, socket=args.socket)
    frozen_fit = c.frozen(fit_args)
    decision = json.loads((HERE / "decision.json").read_text())
    m.require(
        decision["status"] == "accepted_finite_Colab_final_handoff"
        and decision["root"]
        == "/tmp/rsna-predictive-final-deployment-20261020-v1-Colab-C100-deadline"
        and decision["fixed_epoch"] == 6
        and decision["runtime"]["other_reserved_VM_seconds"] == 0,
        "Exact original Colab handoff decision required",
    )

    class Owner(b.Owner):
        def __init__(self):
            self.pin, self.decision, self.socket, self.journal = (
                args.source_git,
                decision,
                args.socket,
                JOURNAL,
            )
            self.base = b
            self.policy = b.load_frozen(
                args.source_git,
                ROOT / "initiative/research/predictive-final-deployment-20261020-v1/policy.py",
                "Colab_runtime_policy",
            )
            self.transport = frozen_fit[2]
            self.launcher = DEPLOY / "run_once.py"
            self.observer = observer_body(m, frozen_fit, c.original_jobs())
            self.qualification = None
            self.remaining_public = 1
            self.counter = 0
            self.contract_options = {}
            self.evidence = {}

        def config(self, observed, mode):
            return configuration(m, self, observed, mode)

        def run(self):
            paths = [JOURNAL / h.FILENAME, Path("/Users/sairamen/Downloads") / h.FILENAME]
            require_marker = JOURNAL / "actual-original-final-evidence.json"
            while True:
                found = [p for p in paths if p.is_file()]
                if found:
                    break
                self.sleep()
            bundle = json.loads(h.read_small(found[0]))
            self.evidence = h.validate(bundle, contract)
            for other in found[1:]:
                m.require(
                    json.loads(h.read_small(other)) == bundle,
                    "Conflicting original completion handoffs",
                )
            b.once(require_marker, bundle)
            observed = self.observe()
            m.require(
                observed["deployment_root_absent"], "Prior Colab deployment requires reconciliation"
            )
            m.require(
                not list((ROOT / "artifacts" / DEPLOY.name).glob("*-intent.json")),
                "Original Colab phase already attempted",
            )
            self.action("stage")
            self.action("prepare")
            self.action("adopt")
            self.action("compile")
            # Do not compete for the still-funded Newton C100 qualification.
            while not self.observe()["original_Newton_C100_submitted"]:
                self.sleep()
            self.action("push")
            while True:
                actual = self.action("status")
                m.require(
                    actual.get("status") not in b.FAILURES, "Actual Colab qualification failed"
                )
                if actual.get("status") == "COMPLETE":
                    self.qualification = actual["receipt"]
                    b.once(JOURNAL / "actual-trained-qualification.json", self.qualification)
                    break
                self.sleep()
            b.once(
                JOURNAL / "actual-runtime-decision.json",
                b.runtime_plan(decision, self.qualification, "C", self.policy, 1),
            )
            self.action("compile", "C")
            self.action("push", "C")
            while True:
                actual = self.action("status", "C")
                m.require(
                    actual.get("status") not in b.FAILURES, "Actual Colab public execution failed"
                )
                if actual.get("status") == "COMPLETE":
                    self.remaining_public = 0
                    break
                self.sleep()
            entry = self.action("submit", "C")
            b.once(JOURNAL / "actual-entry.json", entry)
            while True:
                actual = self.action("outcome", "C")
                if actual["outcome"] in ("scored", "failed"):
                    b.once(JOURNAL / "complete.json", dict(entry=entry, outcome=actual))
                    return
                self.sleep(b.FINISH_END)

    o = Owner()
    if args.preflight_only:
        observed = o.observe()
        funding = o.config(observed, "qualification")["funding"]
        print(
            json.dumps(
                dict(
                    status="actual_readonly_Colab_handoff_preflight",
                    observed=observed,
                    funding=funding,
                    new_GPU_jobs=0,
                    new_account_mutations=0,
                )
            )
        )
        return
    JOURNAL.mkdir(parents=True, mode=0o700, exist_ok=True)
    b.once(
        JOURNAL / "owner-intent.json",
        dict(
            pid=os.getpid(),
            source_git=args.source_git,
            original_Colab_seed=20261021,
            deadline_UTC="2026-10-22T12:00:00Z",
            automatic_mutation_retry=False,
            evidence_source="exact completed notebook output or small browser download; no weights on Mac",
        ),
    )
    try:
        o.run()
    except BaseException as error:
        b.once(
            JOURNAL / "stopped.json",
            dict(
                error_type=type(error).__name__,
                message=str(error),
                uncertain_mutations_must_be_reconciled_readonly=True,
            ),
        )
        raise


if __name__ == "__main__":
    os.umask(0o077)
    main()
