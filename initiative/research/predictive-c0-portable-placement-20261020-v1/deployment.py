"""Thin optional deployment successor; original canonical account controls unchanged."""

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import owner as fitting
import placement as p

ROOT, HERE = fitting.ROOT, fitting.HERE
PARENT = "8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f"
PARENT_DIR = ROOT / "initiative/research/predictive-final-deployment-owner-20261020-v1"


def original_owner():
    return fitting.module(
        fitting.git_file(PARENT, PARENT_DIR.relative_to(ROOT).as_posix() + "/owner.py"),
        "frozen_all33_deployment_owner",
        PARENT_DIR / "owner.py",
    )


def stopped_before_phase(path):
    value = json.loads(path.read_text())
    p.require(
        value["status"] == "actual_idle_predictive_deployment_owner_stopped_before_phase"
        and value["pid"] == 77985
        and value["source_git"] == PARENT
        and value["phase_intents"] == []
        and value["children"] == []
        and value["owner_process_absent_after_stop"] is True,
        "Exact original waiting deployment owner must be stopped before any phase",
    )
    command = subprocess.run(
        ["ps", "-p", "77985", "-o", "command="], capture_output=True, text=True
    ).stdout
    p.require(
        "predictive-final-deployment-owner" not in command, "Prior deployment owner remains active"
    )
    journal = ROOT / "artifacts" / PARENT_DIR.name
    p.require(
        {file.name for file in journal.iterdir()}
        <= {
            "owner-intent.json",
            "launch-intent.json",
            "launcher.json",
            "stdout",
            "stderr",
            "frozen_phase_launcher.py",
            "frozen-phase-source.json",
        },
        "Original deployment owner advanced; reconcile existing account phases",
    )
    launch = json.loads((journal / "launch-intent.json").read_text())
    p.require(
        launch["source_git"] == PARENT and launch["automatic_retry"] is False,
        "Original immutable deployment launch identity changed",
    )
    return value


def portable_finals(original, directory=HERE):
    # Use the original complete final/publication contract, changing only the
    # expected physical node of C0. No score, failed arm or unverified epoch passes.
    source = fitting.git_file(
        PARENT, PARENT_DIR.relative_to(ROOT).as_posix() + "/owner.py"
    ).decode()
    old = 'actual["queue_empty"] is True and actual["node"] == "evc33"'
    p.require(source.count(old) == 1, "Original final contract changed")
    source = source.replace(
        old,
        'actual["queue_empty"] is True and actual["node"] == ("evc27" if arm == "C0" else "evc33")',
    )
    changed = fitting.module(source, "portable_final_reader", PARENT_DIR / "owner.py")
    value = changed.full_finals(directory)
    if value is None:
        return None
    returned_path = directory / "actual-owner-return-C0.json"
    if not returned_path.exists():
        return None
    returned = json.loads(returned_path.read_text())
    c0, publication = value["finals"]["C0"], value["publications"]["C0"]
    p.require(
        returned["status"] == "actual_C0_final_export_returned_to33_all_bytes_SHA_verified"
        and returned["source_node"] == "evc27"
        and returned["destination_node"] == "evc33"
        and returned["original_training_GPU_job"] == c0["GPU_job"]
        and returned["science_git"] == p.SCIENCE
        and returned["binding_sha256"] == c0["binding_sha256"]
        and returned["mirror_run_root"] == str(p.RETURN33)
        and returned["original_six_private_revisions"][-1] == publication["revision"]
        and len(returned["original_six_private_revisions"]) == 6
        and returned["originals_deleted"] is False
        and returned["new_publication"] is False,
        "Actual C0 final return must retain origin27 and original publication",
    )
    for name, expected in publication["files"].items():
        p.require(
            returned["objects"]["publication/epoch-006/" + name] == expected,
            "Returned final object does not match independently verified publication",
        )
    # These values are kept in the original mirror; never synthesize a node33
    # training terminal. The original CPU compiler re-reads/re-hashes every object.
    result = copy.deepcopy(value)
    result["C0_return"] = returned
    result["C0_return_receipt_sha256"] = p.file_sha(returned_path)
    return result


def observer_source(original_source, return_receipt_sha256, jobs, placement_source):
    needle = 'run = Path(f"/tmp/rsna-predictive-full-{arm}-20261020-v2")'
    p.require(original_source.count(needle) == 1, "Original observer path contract changed")
    result = original_source.replace(
        needle,
        "run = Path("
        + repr(str(p.RETURN33))
        + ' if arm == "C0" else f"/tmp/rsna-predictive-full-{arm}-20261020-v2")',
    )
    check = (
        "\n    import hashlib\n"
        f"    returned = Path({str(p.RETURN33 / 'return-verified.json')!r})\n"
        f"    assert hashlib.sha256(returned.read_bytes()).hexdigest() == {return_receipt_sha256!r}\n"
        "    proof = json.loads(returned.read_text())\n"
        "    assert proof['source_node'] == 'evc27' and proof['destination_node'] == 'evc33'\n"
    )
    p.require(result.count("    paid_runs = {}") == 1, "Original observer accounting changed")
    result = result.replace("    paid_runs = {}", check + "    paid_runs = {}")
    preamble = (
        "import types\n"
        "_placement = types.ModuleType('portable_paid_logs')\n"
        f"exec(compile({placement_source!r},'placement.py','exec'),_placement.__dict__)\n"
    )
    result = result.replace(
        "    paid_runs = {}",
        f"    paid_logs = _placement.paid_shared_logs({jobs!r})\n    paid_runs = {{}}",
    )
    result = result.replace(
        '"paid_full_run_bytes": paid_runs,',
        '"paid_full_run_bytes": paid_runs, "paid_original_GPU_logs": paid_logs,',
    )
    return preamble + result


def rebind_config(value, paid):
    value = copy.deepcopy(value)
    p.require(
        type(paid["files"]) is int
        and 0 <= paid["files"] <= 6
        and paid["files"] == len(set(paid["paths"]))
        and paid["originals_deleted"] is False,
        "Exact paid GPU log proof required",
    )
    p.require(
        value["funding"]["other_shared_unpaid_files"] >= 368,
        "Original unpaid file envelope required before one paid-log adjustment",
    )
    value["funding"]["other_shared_unpaid_files"] -= paid["files"]
    value["funding"]["paid_original_GPU_log_files"] = paid
    value["original_runs"]["C0"] = {
        "run_root": str(p.RETURN33),
        "license_path": str(p.RETURN33 / "publication/epoch-006/source-permission.txt"),
    }
    return value


def run(args):
    os.umask(0o077)
    for name in ("deployment.py", "placement.py", "owner.py"):
        path = HERE / name
        p.require(
            fitting.git_file(args.source_git, path.relative_to(ROOT).as_posix())
            == path.read_bytes(),
            "Exact committed deployment adapter required",
        )
    stop = stopped_before_phase(args.deployment_handoff)
    placement_handoff = json.loads(args.fitting_handoff.read_text())
    p.handoff(placement_handoff)
    original = original_owner()
    decision = json.loads(args.decision.read_text())
    original.validate_decision(decision)
    p.require(
        decision["root"] == "/tmp/rsna-predictive-final-deployment-20261020-v1-portable-C0"
        and decision["portable_placement_source_git"] == args.source_git
        and decision["placement_handoff_sha256"] == p.file_sha(args.fitting_handoff)
        and decision["deployment_handoff_sha256"] == p.file_sha(args.deployment_handoff)
        and decision["runtime"]["public_VM_startup_allowance_seconds"] >= 120,
        "Fresh distinct portable deployment decision and3360-second envelope required",
    )
    # Preserve the original official population evidence, never substitute3887.
    population = decision["official_population_receipt"]
    raw = fitting.git_file(PARENT, population["path"])
    p.require(
        p.sha(raw) == population["sha256"]
        and json.loads(raw)["test_studies"]["approximate_count"]
        == decision["runtime"]["planning_study_count"],
        "Original population proof changed",
    )
    journal = ROOT / "artifacts" / (HERE.name + "-deployment")
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    p.once(
        journal / "owner-intent.json",
        {
            "pid": os.getpid(),
            "placement_source_git": args.source_git,
            "canonical_account_control_source_git": PARENT,
            "decision_sha256": p.file_sha(args.decision),
            "deployment_handoff_sha256": p.file_sha(args.deployment_handoff),
            "original_owner": stop["pid"],
            "automatic_retry": False,
        },
    )
    original.full_finals = lambda: portable_finals(original)
    base = original.Owner

    class PortableDeployment(base):
        def __init__(self):
            super().__init__(PARENT, decision, args.socket, journal)

        def _observe(self, dataset_slug=None):
            finals = portable_finals(original)
            p.require(finals is not None, "No account observation before exact portable finals")
            source = fitting.git_file(
                PARENT, (PARENT_DIR / "observe.py.txt").relative_to(ROOT).as_posix()
            ).decode()
            jobs = {}
            for phase in ("dispatch-BC", "dispatch-C0"):
                receipt = json.loads((HERE / ("actual-owner-" + phase + ".json")).read_text())
                for arm, job in receipt["jobs"].items():
                    p.require(
                        job["GPU_job"] == finals["finals"][arm]["GPU_job"],
                        "Original dispatch/final job mismatch",
                    )
                    jobs[arm] = job
            p.require(set(jobs) == {"B", "C", "C0"}, "All exact original dispatched arms required")
            self.observer = observer_source(
                source,
                finals["C0_return_receipt_sha256"],
                jobs,
                fitting.git_file(
                    args.source_git, (HERE / "placement.py").relative_to(ROOT).as_posix()
                ).decode(),
            )
            return super()._observe(dataset_slug)

        def config(self, observed, mode):
            value = super().config(observed, mode)
            return rebind_config(value, observed["paid_original_GPU_logs"])

    p.require(time.time() < original.LAUNCH_END, "Final launch deadline passed")
    try:
        PortableDeployment().run()
    except BaseException as error:
        p.once(
            journal / "stopped.json",
            {
                "error_type": type(error).__name__,
                "automatic_retry": False,
                "preserve_original_phases_and_qualified_baseline": True,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--deployment-handoff", type=Path, required=True)
    parser.add_argument("--fitting-handoff", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    run(parser.parse_args())
