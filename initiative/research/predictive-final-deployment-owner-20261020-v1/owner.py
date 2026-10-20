"""Finite, fixed-final deployment orchestration; no work happens on import."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEPLOY = ROOT / "initiative/research/predictive-final-deployment-20261020-v1"
FIT = ROOT / "initiative/research/predictive-full-fitting-20261020-v2"
ARMS = ("B", "C", "C0")
SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
FINAL = "actual_fixed_final_full3887_predictive_arm_complete"
VERIFIED = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
WAITING = {
    "waiting_existing_canonical_claim_no_mutation",
    "waiting_live_capacity_budget_or_allowance_no_mutation",
}
FAILURES = {"ERROR", "FAILED", "CANCELLED", "CANCEL_ACKNOWLEDGED"}
MUTATIONS = {"stage", "prepare", "dataset", "compile", "push", "submit"}
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
LAUNCH_END = datetime(2026, 10, 22, 12, tzinfo=timezone.utc).timestamp()
FINISH_END = datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc).timestamp()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def once(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing-" + str(os.getpid()))
    with tmp.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def git_file(pin, path):
    assert re.fullmatch("[0-9a-f]{40}", pin), "Exact committed source required"
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT
    )


def load_frozen(pin, path, name):
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__file__ = str(path)
    exec(compile(git_file(pin, path), str(path), "exec"), module.__dict__)
    return module


def validate_decision(value):
    assert value["status"] == "accepted_finite_predictive_final_deployment_owner", (
        "Root must accept the exact source, storage and runtime decisions before launch"
    )
    assert value["owning_node"] == "evc33" and value["arms"] == list(ARMS)
    assert value["fixed_epoch"] == 6 and value["science_git"] == SCIENCE
    assert value["protected_baseline_entry"] == 57333150
    assert value["new_training_or_model_selection"] is False
    assert value["funding"]["explicit_incremental_storage_accepted"] is True
    assert value["funding"]["existing_shared_eight_file_envelope_reused"] is True
    assert len(value["funding"]["reuse_basis"]) >= 30
    assert value["funding"]["other_shared_unpaid_files"] >= 368
    assert value["funding"]["other_shared_unpaid_bytes"] >= 122 * 2**20
    assert value["funding"]["other_node_base_bytes"] >= 66 * 2**30 + 594 * 2**20
    plan = value["runtime"]
    assert plan["qualification_cap_seconds"] == 1200
    assert plan["public_watchdog_seconds"] == 600
    assert plan["predictive_VM_envelope_seconds"] == 3000
    assert type(plan["public_VM_startup_allowance_seconds"]) is int
    assert plan["public_VM_startup_allowance_seconds"] > 0
    assert plan["other_reserved_VM_seconds"] >= 0
    assert type(plan["planning_study_count"]) is int and plan["planning_study_count"] > 0
    assert len(plan["planning_count_basis"]) >= 30
    assert plan["selected_scenario"] in ("mean_two_lane", "max_case_two_lane", "tail_stress_two_lane")
    assert len(plan["entry_rationale"]) >= 30
    assert plan["hidden_runtime_proven"] is False
    assert plan["public_tail_extrapolation_risk_accepted"] is True
    for key in ("startup_allowance_seconds", "safety_margin_seconds", "tail_sensitivity_factor"):
        assert math.isfinite(plan[key]) and plan[key] >= (1 if key.startswith("tail") else 0)
    assert value["runtime_decision_accepted"] is True


def full_finals(directory=FIT):
    """Only original local proofs; no CPU allocation while fitting remains incomplete."""
    finals, publications = {}, {}
    for arm in ARMS:
        if (directory / ("actual-failure-" + arm + ".json")).exists():
            raise RuntimeError("Original training arm failed; no replacement or diagnostic fallback")
        final = directory / ("actual-fixed-final-" + arm + ".json")
        pub = directory / ("actual-owner-publish-" + arm + "-006.json")
        if not final.exists() or not pub.exists():
            return None
        f, p = read(final), read(pub)
        assert f["status"] == FINAL and f["arm"] == arm and f["source_git"] == SCIENCE
        assert (f["epochs"], f["updates"], f["study_exposures"]) == (6, 2916, 23322)
        assert f["all_original_fitting_labels_masks_unchanged"] is True
        assert f["protected_or_gold_cases_read"] is False
        actual = f["actual_Slurm_terminal"]
        assert actual["state"] == "COMPLETED" and actual["exit_code"] == "0:0"
        assert actual["queue_empty"] is True and actual["node"] == "evc33"
        assert p["status"] == VERIFIED and p["objects"] == 5
        assert (p["arm"], p["epoch"], p["source_git"], p["GPU_job"]) == (
            arm, 6, SCIENCE, f["GPU_job"]
        )
        assert p["canonical_shared_publication_lock_used"] is True
        assert re.fullmatch("[0-9a-f]{40}", p["revision"])
        assert p["prefix"] == f"checkpoints/predictive-full3887-{SCIENCE[:12]}/{arm}/epoch-006"
        assert set(p["files"]) == {
            "model.pt", "provenance.json", "own-source.json", "source-permission.txt", "manifest.json"
        }
        finals[arm], publications[arm] = f, p
    assert len({f["initial_shared_tensor_sha256"] for f in finals.values()}) == 1
    assert len({f["GPU_job"] for f in finals.values()}) == 3
    return {"finals": finals, "publications": publications}


def phase_result(phase, mode, actual):
    """Only explicit known no-mutation waits permit another invocation."""
    if actual.get("status") in WAITING:
        return "wait"
    expected = {
        "stage": ("status", "predictive_final_deployment_controls_staged"),
        "prepare": ("status", "actual_three_predictive_final_exports_prepared_CPU_only"),
        "dataset": ("status", "predictive_dataset_create_returned"),
        "dataset-reconcile": ("status", "all_payload_bytes_verified"),
        "compile": ("status", "predictive_final_offline_source_compiled_CPU_only"),
        "push": ("state", "predictive_final_pushed"),
        "submit": ("state", "submitted"),
    }
    if phase in expected:
        key, value = expected[phase]
        assert actual.get(key) == value, "Unexpected phase response; reconcile, never retry"
    if phase == "push":
        assert actual["version"] == 1 and type(actual["kernel_id"]) is int
    if phase == "submit":
        assert mode in ARMS and type(actual["submission_id"]) is int
    if phase == "dataset-reconcile":
        assert actual["version"] == 1 and actual["private"] is True
        assert actual["downloaded_all_bytes"] is True
    return "done"


def frozen_launcher(source):
    """Change only the CLI's self-read to its already bound immutable source."""
    expression = "Path(__file__).read_text()"
    assert source.count(expression) == 1
    body = source.replace(expression, "__frozen_runner_source__")
    return (
        "import json, subprocess, sys\n"
        + "__file__ = " + repr(str(DEPLOY / "run_once.py")) + "\n"
        + "__frozen_runner_source__ = " + repr(source) + "\n"
        + "try:\n exec(compile(" + repr(body) + ", __file__, 'exec'))\n"
        + "except subprocess.TimeoutExpired:\n"
        + " if len(sys.argv)>1 and sys.argv[1] in ('status','outcome'):\n"
        + "  print(json.dumps({'status':'frozen_original_poll_timeout','phase':sys.argv[1]}),flush=True)\n"
        + "  raise SystemExit(1)\n"
        + " raise\n"
    )


def retryable_poll_failure(phase, original_transport):
    if phase not in {"status", "outcome"}:
        return False
    if original_transport.get("returncode") == 255:
        return True
    try:
        reply = json.loads(original_transport.get("stdout", ""))
    except ValueError:
        return False
    if reply.get("status") == "frozen_original_poll_timeout":
        return reply.get("phase") == phase
    return reply.get("status") == "blocked" and reply.get("error_type") in (
        "RemoteError", "ConnectionError", "ReadTimeout", "ConnectTimeout"
    )


def runtime_plan(decision, qualification, mode, policy, remaining_public=3):
    base = decision["runtime"]
    plan = {
        key: base[key]
        for key in (
            "qualification_cap_seconds", "public_watchdog_seconds", "other_reserved_VM_seconds",
            "planning_study_count", "planning_count_basis", "startup_allowance_seconds",
            "tail_sensitivity_factor", "safety_margin_seconds", "hidden_runtime_proven",
            "public_tail_extrapolation_risk_accepted", "probe", "probe_receipt_sha256",
        )
    }
    plan.update(status="accepted_predictive_final_runtime_budget", checked_at=datetime.now(timezone.utc).isoformat())
    # The600-second watchdog starts inside Python. Retain explicit VM startup
    # outside it for each still-unaccepted public arm; this is not a hard VM cap.
    assert type(remaining_public) is int and 0 <= remaining_public <= 3
    plan["other_reserved_VM_seconds"] += remaining_public * base["public_VM_startup_allowance_seconds"]
    if mode in ARMS:
        assert qualification["status"] == "actual_trained_predictive_public_T4_qualification_accepted"
        values = [row for row in qualification["models"] if row["arm"] == mode]
        assert len(values) == 1
        per_study = {}
        for item in values[0]["measurements"]:
            seconds = item["decode_seconds"] + item["render_seconds"] + item["FP16_seconds"]
            assert math.isfinite(seconds) and seconds > 0
            per_study[item["study_index"]] = max(per_study.get(item["study_index"], 0), seconds)
        assert set(per_study) == {0, 1, 2}
        n, startup = plan["planning_study_count"], plan["startup_allowance_seconds"]
        seconds = list(per_study.values())
        scenarios = {
            "mean_two_lane": startup + n * sum(seconds) / 6,
            "max_case_two_lane": startup + n * max(seconds) / 2,
            "tail_stress_two_lane": startup + n * max(seconds) * plan["tail_sensitivity_factor"] / 2,
        }
        plan["entry_decision"] = {
            "accepted": True,
            "trained_qualification_sha256": sha(canonical(qualification)),
            "selected_scenario": base["selected_scenario"],
            "selected_estimate_seconds": scenarios[base["selected_scenario"]],
            "rationale": base["entry_rationale"],
        }
        policy.hidden_projection(
            plan, values[0]["measurements"], qualification_sha256=sha(canonical(qualification))
        )
    return plan


class Owner:
    def __init__(self, pin, decision, socket, journal):
        self.pin, self.decision, self.socket, self.journal = pin, decision, socket, journal
        self.policy = load_frozen(pin, DEPLOY / "policy.py", "owner_original_policy")
        self.runner = load_frozen(pin, DEPLOY / "run_once.py", "owner_original_phases")
        self.runner_source = git_file(pin, DEPLOY / "run_once.py").decode()
        self.launcher = journal / "frozen_phase_launcher.py"
        body = frozen_launcher(self.runner_source).encode()
        with self.launcher.open("xb") as handle:
            handle.write(body)
        once(journal / "frozen-phase-source.json", {
            "source_git": pin, "original_runner_sha256": sha(self.runner_source.encode()),
            "launcher_sha256": sha(body), "only_self_read_replaced": True,
        })
        self.transport = load_frozen(
            pin, ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py",
            "owner_original_stdin_transport",
        )
        self.observer = git_file(pin, HERE / "observe.py.txt").decode()
        self.qualification = None
        self.remaining_public = 3
        self.counter = 0

    def sleep(self, deadline=LAUNCH_END):
        assert time.time() < deadline, "Finite deadline reached; preserve originals"
        time.sleep(min(45, max(0, deadline - time.time())))

    def _observe(self, dataset_slug=None):
        # No token in argv or stored config. Read-only network/metadata CPU only.
        env = ROOT / ".env"
        assert env.stat().st_mode & 0o077 == 0
        assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0
        values = dict(
            (k.strip(), v.strip().strip("\"'"))
            for line in env.read_text().splitlines()
            for k, sep, v in [line.partition("=")] if sep and not line.lstrip().startswith("#")
        )
        payload = {
            "root": self.decision["root"], "node": "evc33",
            "dataset_slug": dataset_slug, "token": values["KAGGLE_API_TOKEN"],
        }
        command = [
            "srun", "--account=skattel", "--partition=normal", "--qos=normal",
            "--nodelist=evc33", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
            "--mem=1G", "--time=00:05:00", "--job-name=rsna-predictive-serving-observe",
            "env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1",
            MANAGEMENT, "-B", "-c", self.transport.COMMAND,
        ]
        result = subprocess.run(
            ["ssh", "-S", self.socket, "-o", "BatchMode=yes", "newton", shlex.join(command)],
            input=self.transport.envelope(self.observer, payload), text=True,
            capture_output=True, timeout=340,
        )
        if result.returncode == 255:
            raise ConnectionError("Read-only SSH observation lost")
        value = json.loads(result.stdout)
        if value.get("status") == "read_only_observation_unavailable":
            raise ConnectionError("Read-only API observation unavailable")
        assert result.returncode == 0, "Read-only observation binding failed"
        assert value["status"] == "actual_predictive_deployment_read_only"
        return value

    def observe(self, dataset_slug=None, deadline=LAUNCH_END):
        start = time.monotonic()
        while True:
            try:
                return self._observe(dataset_slug)
            except (ConnectionError, subprocess.TimeoutExpired):
                assert time.monotonic() - start <= 1800, "Read-only observation absent30minutes"
                self.sleep(deadline)

    def config(self, observed, mode):
        assert 0 <= time.time() - observed["observed_at"] <= 180
        funding = self.decision["funding"]
        cache = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v3"
        paid_cache = 0
        for role, cap in (("raw", 320 * 2**30), ("context", 96 * 2**30)):
            receipt = read(cache / ("actual-owner-" + role + "_collect.json"))
            assert receipt["status"] == "actual_original_full_cache_terminal_collected"
            assert receipt["studies"] == 3887 and receipt["target_terminal"] == "COMPLETED"
            assert receipt["exit_code"] == "0:0"
            assert 0 < receipt["existing_cache_bytes"] <= cap
            paid_cache += receipt["existing_cache_bytes"]
        # The66GiB base retains all36GiB model/publication scopes plus scratch.
        # Actual paid bytes reduce those original scopes once, without deletion.
        paid_runs = observed["paid_full_run_bytes"]
        assert set(paid_runs) == set(ARMS)
        assert all(type(x) is int and 0 <= x <= 12 * 2**30 for x in paid_runs.values())
        other_node = funding["other_node_base_bytes"] + 416 * 2**30 - paid_cache - sum(paid_runs.values())
        return {
            "root": self.decision["root"], "owning_node": "evc33",
            "original_runs": {
                arm: {
                    "run_root": f"/tmp/rsna-predictive-full-{arm}-20261020-v2",
                    "license_path": "/tmp/rsna-S6-target-execution-inputs-20261019-v1/source-permission.txt",
                } for arm in ARMS
            },
            "parent_source_path": "/tmp/rsna-S6-target-execution-inputs-20261019-v1/parent_source.private.py",
            "funding": {
                "status": "predictive_final_deployment_incremental_storage_funded",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "new_node_bytes": 2 * 2**30, "new_shared_bytes": 2**20, "new_shared_files": 8,
                "existing_60GiB_or_574MiB_reused": False,
                "other_node_unpaid_bytes": other_node,
                **{key: funding[key] for key in (
                    "other_shared_unpaid_bytes", "other_shared_unpaid_files"
                )},
            },
            "runtime_plan": runtime_plan(
                self.decision, self.qualification, mode, self.policy, self.remaining_public
            ),
        }

    def action(self, phase, mode="qualification"):
        stem = phase + "-" + mode
        intent = self.journal / (stem + "-intent.json")
        if phase in MUTATIONS:
            assert not intent.exists(), "Existing phase intent requires read-only reconciliation"
        polling_outage = None
        while True:
            deadline = FINISH_END if phase == "outcome" else LAUNCH_END
            assert time.time() < deadline, "No new launch after October22 noon UTC"
            observed = self.observe(deadline=deadline)
            if observed["canonical_claim_present"]:
                self.sleep(deadline)
                continue
            if phase == "submit" and observed["allowance"] <= 0:
                self.sleep(deadline)
                continue
            if phase == "push" and mode == "qualification":
                runtime = self.decision["runtime"]
                required = 3000 + runtime["other_reserved_VM_seconds"] + 3 * runtime["public_VM_startup_allowance_seconds"]
                if observed["free_VM_seconds"] < required:
                    (self.journal / "waiting-budget.json").write_bytes(canonical({
                        "status": "waiting_actual_predictive_VM_budget_no_mutation",
                        "free_VM_seconds": observed["free_VM_seconds"],
                        "required_VM_seconds": required,
                        "root_budget_review_needed": True,
                        "caps_unchanged": True,
                    }))
                    self.sleep(deadline)
                    continue
            config = self.config(observed, mode)
            self.counter += 1
            config_path = self.journal / f"phase-{self.counter:05d}.private.json"
            once(config_path, config)
            if phase in MUTATIONS and not intent.exists():
                once(intent, {
                    "phase": phase, "mode": mode, "source_git": self.pin,
                    "automatic_retry": False,
                })
            command = [
                sys.executable, str(self.launcher), phase, "--config", str(config_path),
                "--source-git", self.pin, "--ssh-socket", self.socket, "--mode", mode,
            ]
            originals = ROOT / "artifacts" / DEPLOY.name
            before = set(originals.glob("*-transport.json"))
            attempt_started = time.monotonic()
            try:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1950)
            except subprocess.TimeoutExpired:
                assert phase in {"status", "outcome"}, (
                    "Mutation transport timed out; no automatic retry"
                )
                polling_outage = polling_outage or attempt_started
                assert time.monotonic() - polling_outage <= 1800, "Original poll absent30minutes"
                self.sleep(deadline)
                continue
            once(self.journal / f"transport-{self.counter:05d}.json", {
                "phase": phase, "mode": mode, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
            })
            if result.returncode != 0:
                created = set(originals.glob("*-transport.json")) - before
                transport = read(created.pop()) if len(created) == 1 else {"stdout": result.stdout}
                assert retryable_poll_failure(phase, transport), (
                    "Uncertain original mutation or invalid binding; no automatic retry"
                )
                polling_outage = polling_outage or attempt_started
                assert time.monotonic() - polling_outage <= 1800, "Original poll absent30minutes"
                self.sleep(deadline)
                continue
            polling_outage = None
            actual = json.loads(result.stdout)
            state = phase_result(phase, mode, actual)
            once(self.journal / f"result-{self.counter:05d}.json", actual)
            if state != "wait":
                if phase in MUTATIONS:
                    once(self.journal / (stem + "-result.json"), actual)
                return actual
            # The phase explicitly proves it stopped before external mutation.
            # Never repeat a completed action, thrown exception or unknown response.
            self.sleep(deadline)

    def run(self):
        while full_finals() is None:
            self.sleep()
        once(self.journal / "actual-fixed-final-inputs.json", full_finals())
        observed = self.observe()
        assert observed["deployment_root_absent"], "Prior deployment requires read-only reconciliation"
        assert not list((ROOT / "artifacts" / DEPLOY.name).glob("*-intent.json")), (
            "An original deployment call already exists; do not start a second owner"
        )
        self.action("stage")
        prepared = self.action("prepare")
        self.action("dataset")
        while True:
            observed = self.observe(prepared["dataset_slug"])
            if observed["dataset_ready"]:
                break
            assert observed["dataset_pending"], "Dataset failed or changed; never recreate it"
            self.sleep()
        self.action("dataset-reconcile")
        self.action("compile")
        self.action("push")
        while True:
            result = self.action("status")
            assert result.get("status") not in FAILURES, "Actual qualification failed"
            if result.get("status") == "COMPLETE":
                self.qualification = result["receipt"]
                once(self.journal / "actual-trained-qualification.json", self.qualification)
                break
            self.sleep()
        # Decide all fixed arms from the same actual receipt before any public push.
        decisions = {
            arm: runtime_plan(self.decision, self.qualification, arm, self.policy)
            for arm in ARMS
        }
        once(self.journal / "actual-trained-runtime-decisions.json", decisions)
        entries = {}
        for arm in ARMS:
            self.action("compile", arm)
            self.action("push", arm)
            while True:
                result = self.action("status", arm)
                assert result.get("status") not in FAILURES, "Actual public execution failed"
                if result.get("status") == "COMPLETE":
                    self.remaining_public -= 1
                    break
                self.sleep()
            entries[arm] = self.action("submit", arm)
            once(self.journal / ("actual-entry-" + arm + ".json"), entries[arm])
        outcomes = {}
        while len(outcomes) < 3:
            for arm in ARMS:
                if arm not in outcomes:
                    actual = self.action("outcome", arm)
                    if actual["outcome"] in ("scored", "failed"):
                        outcomes[arm] = actual
                        once(self.journal / ("actual-hidden-outcome-" + arm + ".json"), actual)
            if len(outcomes) < 3:
                self.sleep(FINISH_END)
        once(self.journal / "complete.json", {"entries": entries, "outcomes": outcomes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    for name in ("owner.py", "observe.py.txt", "decision.json"):
        assert git_file(args.source_git, HERE / name) == (HERE / name).read_bytes()
    decision = read(HERE / "decision.json")
    validate_decision(decision)
    population = decision["official_population_receipt"]
    body = git_file(args.source_git, ROOT / population["path"])
    assert sha(body) == population["sha256"]
    observed = json.loads(body)
    assert observed["status"] == "official_rendered_competition_data_description_rechecked"
    assert observed["test_studies"]["approximate_count"] == decision["runtime"]["planning_study_count"]
    assert observed["test_studies"]["exact_count_or_upper_bound"] is False
    assert time.time() < LAUNCH_END
    journal = ROOT / "artifacts" / HERE.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    once(journal / "owner-intent.json", {
        "pid": os.getpid(), "source_git": args.source_git,
        "decision_sha256": sha(canonical(decision)), "automatic_retry": False,
        "launch_deadline_UTC": "2026-10-22T12:00:00Z",
        "finish_deadline_UTC": "2026-10-22T23:59:00Z",
    })
    try:
        Owner(args.source_git, decision, args.socket, journal).run()
    except BaseException as exc:
        once(journal / "stopped.json", {
            "error_type": type(exc).__name__, "automatic_retry": False,
            "preserve_original_phases_and_qualified_baseline": True,
        })
        raise


if __name__ == "__main__":
    main()
