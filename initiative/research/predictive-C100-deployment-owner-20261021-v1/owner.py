"""Finite handoff from the original C100 fit to its canonical serving phases.

No new fitting, checkpoint publication or account action occurs on import.
Known no-mutation waits may repeat; uncertain writes stop permanently for review.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "c8b7c3de810f8f986692f9388935939d66d82e85"
ORIGINAL = ROOT / "initiative/research/predictive-final-deployment-owner-20261020-v1"
DEPLOY = ROOT / "initiative/research/predictive-density100-deployment-20261021-v1"
RECOVERY = ROOT / "initiative/research/predictive-C100-preservation-recovery-20261021-v1"
FINAL = ROOT / "artifacts" / RECOVERY.name / "continuation/owner-result.json"
SOURCE = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
BINDING = "f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd"
JOURNAL = ROOT / "artifacts" / HERE.name


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source(path):
    return subprocess.check_output(
        ["git", "show", PIN + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT
    ).decode()


def module(body, path, name):
    value = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    value.__file__ = str(path)
    exec(compile(body, str(path), "exec"), value.__dict__)
    return value


def replace(body, old, new):
    require(body.count(old) == 1, "Original orchestration anchor changed")
    return body.replace(old, new)


def original():
    body = source(ORIGINAL / "owner.py")
    body = replace(body, 'ARMS = ("B", "C", "C0")', 'ARMS = ("C",)')
    body = replace(
        body,
        "actual_three_predictive_final_exports_prepared_CPU_only",
        "actual_single_C100_final_export_prepared_CPU_only",
    )
    body = replace(
        body,
        'required = 3000 + runtime["other_reserved_VM_seconds"] + 3 * runtime["public_VM_startup_allowance_seconds"]',
        'required = runtime["qualification_cap_seconds"] + runtime["public_watchdog_seconds"] + runtime["other_reserved_VM_seconds"] + runtime["public_VM_startup_allowance_seconds"]',
    )
    # Same observer transport, now on the actual C100 owning node.
    body = replace(body, '"node": "evc33",', '"node": "evc27",')
    body = replace(body, '"--nodelist=evc33"', '"--nodelist=evc27"')
    value = module(body, ORIGINAL / "owner.py", "C100_original_finite_phases")
    value.DEPLOY = DEPLOY
    return value


def training():
    recovery = module(
        source(RECOVERY / "continue.py"), RECOVERY / "continue.py", "C100_existing_fit"
    )
    return recovery.original()


def final_evidence(value):
    require(
        value.get("full_scheduler", {}).get("job_id") == "995351",
        "Actual original C100 scheduler required",
    )
    s = value["full_scheduler"]
    require(
        s["state"] == "COMPLETED"
        and s["exit_code"] == "0:0"
        and s["queue_empty"] is True
        and s["node"] == "evc27",
        "Original fit did not complete",
    )
    terminal = value["terminal"]
    require(
        terminal["GPU_job"] == "995351"
        and terminal["source_git"] == SOURCE
        and terminal["binding_sha256"] == BINDING
        and terminal["branch"] == "C100"
        and terminal["epochs"] == 6
        and terminal["updates"] == 2916,
        "Only the actual fixed final is deployable",
    )
    publications = value["publications"]
    require(set(publications) == set(map(str, range(1, 7))), "All six original epochs required")
    ordered = [publications[str(i)] for i in range(1, 7)]
    require(
        all(
            p["epoch"] == i
            and p["GPU_job"] == "995351"
            and p["source_git"] == SOURCE
            and p["status"]
            == "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
            and p["objects"] == 5
            for i, p in enumerate(ordered, 1)
        ),
        "All original publications must already be fully verified",
    )
    return dict(epoch_publications=ordered, full_terminal=terminal, full_scheduler=s)


def options(value, frozen):
    text = json.dumps(value["pilot"], sort_keys=True) + "\n"
    require(
        sha(text.encode()) == value["pilot_sha256"], "Exact original pilot serialization required"
    )
    return dict(
        full_source_git=SOURCE,
        publisher_sha256=frozen[0]["code_sha256"]["publish_epoch.py"],
        pilot_receipt_text=text,
        pilot_receipt_sha256=value["pilot_sha256"],
        pilot_scheduler=value["pilot_scheduler"],
    )


def observer_body(frozen, jobs):
    placement = replace(
        frozen[0]["placement_source"],
        'job["node"] == ("evc27" if arm == "C0" else "evc33")',
        'job["node"] == "evc27"',
    )
    body = source(ORIGINAL / "observe.py.txt").replace('== "evc33"', '== "evc27"')
    additions = """
    import hashlib
    import shutil
    import subprocess
    binding_path = Path('/tmp/rsna-predictive-density100-full-20261020-v1/binding.private.json')
    raw = binding_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_BINDING
    bound = json.loads(raw)
    paid = _placement.paid_shared_logs(PRIMARY_JOBS)
    quota = subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'], text=True)
    row = next(r.split() for r in quota.splitlines() if r.strip().startswith('/home/sa305415'))
    result.update(paid_original_GPU_logs=paid, node_free_bytes=shutil.disk_usage('/tmp').free,
        shared_free_bytes=(int(row[2])-int(row[1]))*1024, shared_free_files=int(row[6])-int(row[5]),
        original_license=bound['license']['path'], original_parent_source=bound['parent_source']['path'])
"""
    body = replace(
        body, '    if value.get("dataset_slug"):', additions + '\n    if value.get("dataset_slug"):'
    )
    return (
        "import types\n_placement=types.ModuleType('original_paid_logs')\n"
        f"exec(compile({placement!r},'placement.py','exec'),_placement.__dict__)\n"
        f"PRIMARY_JOBS={jobs!r}\nEXPECTED_BINDING={BINDING!r}\n" + body
    )


def config(o, observed, mode, policy):
    require(
        -2 <= time.time() - observed["observed_at"] <= 180,
        "Fresh actual account/storage observation required",
    )
    paid = observed["paid_original_GPU_logs"]
    require(
        paid["files"] == 6 and len(set(paid["paths"])) == 6 and paid["originals_deleted"] is False,
        "Exact six already-paid original logs required",
    )
    funding = dict(
        status="predictive_final_deployment_incremental_storage_funded",
        checked_at=o.base.datetime.now(o.base.timezone.utc).isoformat(),
        new_node_bytes=2 * 2**30,
        new_shared_bytes=2**20,
        new_shared_files=8,
        existing_60GiB_or_574MiB_reused=False,
        other_node_unpaid_bytes=512 * 2**30,
        other_shared_unpaid_bytes=122 * 2**20,
        other_shared_unpaid_files=368 - paid["files"],
    )
    # Conservative512GiB retains all original optional/primary evc27 obligations;
    # no deletion/paid-byte credits. The same canonical claim serializes the
    # already funded8-file SQLite/claim envelope, never two concurrent scopes.
    policy.storage(
        funding,
        node_free=observed["node_free_bytes"],
        shared_free=observed["shared_free_bytes"],
        shared_files=observed["shared_free_files"],
    )
    return dict(
        root=o.decision["root"],
        owning_node="evc27",
        original_runs={
            "C": dict(
                run_root="/tmp/rsna-predictive-full-C100-20261020-v1",
                license_path=observed["original_license"],
            )
        },
        parent_source_path=observed["original_parent_source"],
        funding=funding,
        C100_contract_options=o.contract_options,
        C100_final_evidence=o.evidence,
        runtime_plan=o.base.runtime_plan(
            o.decision, o.qualification, mode, policy, o.remaining_public
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("owner.py", "decision.json"):
        p = HERE / name
        require(
            subprocess.check_output(
                ["git", "show", args.source_git + ":" + p.relative_to(ROOT).as_posix()], cwd=ROOT
            )
            == p.read_bytes(),
            "Execute exact committed owner/decision",
        )
    b = original()
    c = training()
    fit_args = SimpleNamespace(source_git=SOURCE, socket=args.socket)
    frozen = c.frozen(fit_args)
    decision = json.loads((HERE / "decision.json").read_bytes())
    require(
        decision["status"] == "accepted_finite_C100_final_handoff"
        and decision["fixed_epoch"] == 6
        and decision["original_GPU_job"] == "995351"
        and decision["runtime"]["selected_scenario"] == "mean_two_lane"
        and decision["runtime"]["other_reserved_VM_seconds"] == 0,
        "Exact separate final-handoff decision required",
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
                "C100_runtime_policy",
            )
            self.transport = frozen[2]
            self.launcher = DEPLOY / "run_once.py"
            self.observer = observer_body(frozen, c.original_jobs())
            self.qualification = None
            self.remaining_public, self.counter = 1, 0

        def config(self, observed, mode):
            return config(self, observed, mode, self.policy)

        def run(self):
            while not FINAL.is_file():
                require(
                    subprocess.run(
                        ["ps", "-p", "76338", "-o", "pid="], capture_output=True
                    ).returncode
                    == 0,
                    "Original preservation owner ended without a final result",
                )
                self.sleep()
            original_final = final_evidence(b.read(FINAL))
            fresh = c.observe(fit_args, frozen, b.LAUNCH_END)
            require(fresh is not None, "Actual original terminal unavailable")
            self.evidence = final_evidence(fresh)
            require(self.evidence == original_final, "Final original evidence changed")
            self.contract_options = options(fresh, frozen)
            b.once(JOURNAL / "actual-final-evidence.json", self.evidence)
            observed = self.observe()
            require(observed["deployment_root_absent"], "Original remote root already exists")
            require(
                not list((ROOT / "artifacts" / DEPLOY.name).glob("*-intent.json")),
                "Existing explicit C100 phase needs reconciliation",
            )
            self.action("stage")
            prepared = self.action("prepare")
            self.action("dataset")
            while True:
                observed = self.observe(prepared["dataset_slug"])
                if observed["dataset_ready"]:
                    break
                require(observed["dataset_pending"], "Dataset failed/changed; never recreate")
                self.sleep()
            self.action("dataset-reconcile")
            self.action("compile")
            self.action("push")
            while True:
                actual = self.action("status")
                require(
                    actual.get("status") not in b.FAILURES, "Actual trained qualification failed"
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
                require(actual.get("status") not in b.FAILURES, "Actual public execution failed")
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
        # No fabricated production evidence: check only live storage and account.
        o.contract_options, o.evidence = {}, {}
        funded = o.config(observed, "qualification")["funding"]
        print(
            json.dumps(
                dict(
                    status="actual_readonly_C100_handoff_preflight",
                    observed=observed,
                    funding=funded,
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
            original_GPU_job="995351",
            decision_sha256=sha((HERE / "decision.json").read_bytes()),
            new_training=False,
            automatic_mutation_retry=False,
            deadline_UTC="2026-10-22T12:00:00Z",
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
