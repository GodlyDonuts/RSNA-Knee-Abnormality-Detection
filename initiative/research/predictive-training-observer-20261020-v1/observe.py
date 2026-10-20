"""Finite read-only observer of the original portable fitting jobs.

Uses only the pinned portable mode=progress path: shared logs and scheduler reads.
Never calls owner.main/freeze, allocates a CPU/GPU, or reads credentials or pixels.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PIN = "926b657770564c79b23509592e2d6a95761abd70"
SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PORTABLE = "initiative/research/predictive-c0-portable-placement-20261020-v1"
BASE = "initiative/research/predictive-full-fitting-20261020-v2"
SUCCESSOR = "initiative/research/predictive-v4-owner-handoff-20261020-v1"
OWNER_JOURNAL = ROOT / "artifacts/predictive-v4-owner-handoff-20261020-v1/finite-owner"
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
LOG_ROOT = "/home/sa305415/rsna-knee/kaggle"
DEADLINE = "2026-10-22T09:00:00+00:00"
TELEMETRY = (
    "step",
    "epoch",
    "studies",
    "loss",
    "exam_loss",
    "density_loss",
    "encoder_gradient_norm",
    "head_gradient_norm",
    "gradient_norm",
    "LR",
    "encoder_updated",
    "elapsed_seconds",
    "seconds_since_previous_update",
    "studies_per_second_since_previous_update",
    "CPU_context_load_seconds",
    "study_exposures",
    "peak_GPU_bytes",
    "peak_host_RSS_bytes",
    "output_bytes",
    "output_files",
    "sealed_epoch_sidecars",
)
MAXIMA = (
    "loss",
    "exam_loss",
    "head_gradient_norm",
    "encoder_gradient_norm",
    "gradient_norm",
    "peak_GPU_bytes",
    "peak_host_RSS_bytes",
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    require(not path.is_symlink() and path.stat().st_size < 2**20, "Bounded original receipt")
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def save(path, value, *, replace=False):
    raw = canonical(value) + b"\n"
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(pending, path)
        else:
            os.link(pending, path)
    finally:
        if pending.exists():
            pending.unlink()


def git_file(commit, name):
    require(re.fullmatch("[0-9a-f]{40}", commit), "Exact source pin")
    return subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)


def frozen():
    """Reuse the successor's pure capsule builder, never its freeze/launch path."""
    owner, owner_sha = read(OWNER_JOURNAL / "owner-intent.json")
    pin = owner["source_git"]
    require(re.fullmatch("[0-9a-f]{40}", pin), "Exact actual successor pin")
    own = {name: git_file(pin, SUCCESSOR + "/" + name).decode()
           for name in ("contract.py", "remote_adapter.py", "launch.py")}
    require({name: sha(body.encode()) for name, body in own.items()} == owner["source_sha256"],
            "Original successor source manifest differs")
    def owned_reference(key):
        relative = Path(owner[key])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Original local reference must stay inside repository")
        return ROOT / relative
    decision_path, handoff_path = owned_reference("decision_path"), owned_reference("handoff_path")
    decision, decision_sha = read(decision_path)
    handoff, handoff_sha = read(handoff_path)
    prepared, prepared_sha = read(ROOT / BASE / "actual-cpu-prepared.json")
    require(owner["science_git"] == prepared["source_git"] == SCIENCE
            and owner["original_owner_source_git"] == PIN
            and owner["original_owner_pid"] == 80995
            and owner["decision_sha256"] == decision_sha
            and owner["handoff_sha256"] == handoff_sha
            and decision["placement_source_git"] == pin
            and decision["status"] == "accepted_v4_cache_owner_handoff"
            and prepared["CPU_job"] == "979667", "Exact successor ownership/preparation required")
    contract = types.ModuleType("observer_successor_contract")
    exec(compile(own["contract.py"], "frozen_successor_contract", "exec"), contract.__dict__)
    contract.handoff(handoff, "fitting")
    launcher = types.ModuleType("observer_successor_capsule_builder")
    launcher.__file__ = str(ROOT / SUCCESSOR / "launch.py")
    previous = sys.modules.get("contract")
    sys.modules["contract"] = contract
    try:
        exec(compile(own["launch.py"], "frozen_successor_launcher_library", "exec"), launcher.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = previous
    originals = {name: git_file(PIN, PORTABLE + "/" + name).decode()
                 for name in ("placement.py", "remote.py")}
    program = launcher.build_program(own, originals, readonly=True)
    transport = types.ModuleType("training_observer_frozen_transport")
    exec(compile(git_file(SCIENCE,
        "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"),
        "frozen_transport", "exec"), transport.__dict__)
    payload = {"mode": "progress", "prepared": prepared,
        "original_remote": git_file(SCIENCE, BASE + "/owner_remote.py.txt").decode(),
        "handoff": handoff, "handoff_text": handoff_path.read_text(),
        "handoff_sha256": handoff_sha, "placement_source_git": pin,
        "successor_code_sha256": owner["source_sha256"],
        "shared_bytes": decision["shared_future_bytes"],
        "shared_files": decision["shared_future_files"]}
    provenance = {"owner_intent_sha256": owner_sha, "decision_sha256": decision_sha,
        "handoff_sha256": handoff_sha, "prepared_sha256": prepared_sha,
        "placement_source_git": pin, "original_portable_source_git": PIN,
        "successor_source_sha256": owner["source_sha256"],
        "progress_program_sha256": sha(program.encode())}
    return transport, program, payload, provenance


def dispatches():
    """Wait for the sole owner's original paired local result receipts."""
    found, refs = {}, {}
    for phase, node, arms in (
        ("dispatch-BC", "evc33", {"B", "C"}),
        ("dispatch-C0", "evc27", {"C0"}),
    ):
        journal = OWNER_JOURNAL / (phase + "-result.json")
        audit = ROOT / SUCCESSOR / ("actual-owner-" + phase + ".json")
        if not journal.exists() or not audit.exists():
            continue
        value, digest = read(journal)
        actual, audit_digest = read(audit)
        require(value == actual and digest == audit_digest, "Dispatch receipt pair differs")
        require(
            value["status"] == "actual_predictive_placement_jobs_submitted_once"
            and value["submitted_arm_count"] == len(arms)
            and value["source_git"] == SCIENCE
            and set(value["jobs"]) == arms,
            "Exact original dispatch cohort required",
        )
        for arm, job in value["jobs"].items():
            identity = job["GPU_job"]
            name = "rsna-predictive-full2-" + arm
            require(
                isinstance(identity, str)
                and identity.isdecimal()
                and job["arm"] == arm
                and job["node"] == node
                and job["source_git"] == SCIENCE
                and job["job_name"] == name
                and re.fullmatch("[0-9a-f]{64}", job["binding_sha256"])
                and re.fullmatch("[0-9a-f]{64}", job["script_sha256"])
                and job["stdout"] == f"{LOG_ROOT}/{name}-{identity}.out"
                and job["stderr"] == f"{LOG_ROOT}/{name}-{identity}.err"
                and job["run_root"] == f"/tmp/rsna-predictive-full-{arm}-20261020-v2",
                "Unexpected original job/node/source/path binding",
            )
        found[node] = value
        refs[node] = {"path": str(journal.relative_to(ROOT)), "sha256": digest}
    ids = [j["GPU_job"] for v in found.values() for j in v["jobs"].values()]
    require(len(ids) == len(set(ids)), "GPU job reused across arms")
    return found, refs


def telemetry(value):
    if value is None:
        return None
    out = {key: value[key] for key in TELEMETRY}
    require(type(out["step"]) is int and 1 <= out["step"] <= 2916, "Unexpected update")
    for key, item in out.items():
        numbers = item if key == "LR" else [item]
        require(
            all(type(n) in (int, float, bool) and math.isfinite(n) for n in numbers),
            "Nonfinite telemetry",
        )
    return out


def sanitized(actual, dispatched, reference, provenance):
    require(
        actual["status"] == "actual_predictive_full_job_progress_observed", "Unexpected progress"
    )
    require(set(actual["arms"]) == set(dispatched["jobs"]), "Progress arm identity changed")
    arms = {}
    for arm, item in actual["arms"].items():
        job = dispatched["jobs"][arm]
        require(item["GPU_job"] == job["GPU_job"], "Progress job changed")
        latest = telemetry(item["latest"])
        require(
            item["observed_updates"] == (latest["step"] if latest else 0), "Wrong latest update"
        )
        state = {
            k: item["slurm"][k]
            for k in ("state", "queue_empty", "exit_code", "name", "node")
            if k in item["slurm"]
        }
        early = item["first200"]
        if early is not None:
            require(
                early["actual_updates"] == 200
                and item["observed_updates"] >= 200
                and re.fullmatch("[0-9a-f]{64}", early["history_sha256"]),
                "Wrong first200 evidence",
            )
            maxima = {k: early["maxima"][k] for k in MAXIMA}
            require(all(math.isfinite(v) for v in maxima.values()), "Nonfinite first200")
            last = telemetry(early["last"])
            require(last["step"] == 200, "Wrong first200 endpoint")
            early = {
                "actual_updates": 200,
                "history_sha256": early["history_sha256"],
                "maxima": maxima,
                "last": last,
                "clinical_accuracy_claim": False,
            }
        arms[arm] = {
            "arm": arm,
            "GPU_job": job["GPU_job"],
            "node": job["node"],
            "source_git": SCIENCE,
            "placement_source_git": provenance.get("placement_source_git", PIN),
            "binding_sha256": job["binding_sha256"],
            "script_sha256": job["script_sha256"],
            "stdout": job["stdout"],
            "stderr": job["stderr"],
            "dispatch_receipt": reference,
            "observer_provenance": provenance,
            "observed_updates": item["observed_updates"],
            "slurm": state,
            "latest": latest,
            "first200": early,
            "stderr_bytes": item["stderr_bytes"],
            "sealed_epochs": item["sealed_epochs"],
            "fixed_final_seen": item["terminal"] is not None,
            "clinical_accuracy_or_logit_saturation_claim": False,
        }
    return arms


class ObservationRejected(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Original read-only progress validation rejected; preserve jobs")


def query(args, transport, program, payload, node, dispatched, all_jobs, deadline):
    command = [
        "ssh",
        "-S",
        args.socket,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "newton",
        shlex.join([MANAGEMENT, "-B", "-c", transport.COMMAND]),
    ]
    value = {**payload, "node": node, "dispatched": dispatched, "all_dispatched_jobs": all_jobs}
    result = subprocess.run(
        command,
        input=transport.envelope(program, value),
        capture_output=True,
        text=True,
        timeout=max(1, min(90, deadline - time.time())),
    )
    if result.returncode == 255:
        raise ConnectionError("Read-only SSH observation lost")
    if result.returncode:
        diagnostic = {}
        for line in result.stderr.splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and type(row.get("site_line")) is int:
                    diagnostic = {
                        "error_type": row.get("error_type"),
                        "site_line": row["site_line"],
                    }
            except ValueError:
                pass
        raise ObservationRejected(
            {
                "node": node,
                "jobs": {a: j["GPU_job"] for a, j in dispatched["jobs"].items()},
                "returncode": result.returncode,
                "diagnostic": diagnostic,
                "stdout_sha256": sha(result.stdout.encode()),
                "stderr_sha256": sha(result.stderr.encode()),
                "nonfinite_or_other_original_validation_failure_not_retried": True,
            }
        )
    return json.loads(result.stdout)


def run(args, journal):
    transport, program, payload, provenance = frozen()
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
    save(
        journal / "observer-intent.json",
        {
            "status": "read_only_training_observer_started",
            "pid": os.getpid(),
            "observer_source_git": args.source_git,
            "placement_source_git": provenance.get("placement_source_git", PIN),
            "source_git": SCIENCE,
            "deadline_utc": args.deadline_utc,
            "provenance": provenance,
            "GPU_or_CPU_allocations": 0,
            "remote_mutations": 0,
        },
    )
    latest, saved, bound, outage = {}, set(), {}, None
    while time.time() < deadline:
        dispatched, refs = dispatches()
        for node, ref in refs.items():
            require(node not in bound or bound[node] == ref, "Original dispatch receipt changed")
            bound[node] = ref
        all_jobs = {a: j for v in dispatched.values() for a, j in v["jobs"].items()}
        try:
            for node, jobs in dispatched.items():
                actual = query(args, transport, program, payload, node, jobs, all_jobs, deadline)
                latest.update(sanitized(actual, jobs, refs[node], provenance))
            outage = None
        except (ConnectionError, subprocess.TimeoutExpired, OSError):
            outage = outage or time.monotonic()
            require(time.monotonic() - outage <= 1800, "Read-only observation absent30minutes")
            time.sleep(min(args.poll_seconds, max(0, deadline - time.time())))
            continue
        observed = utc()
        for arm, item in latest.items():
            if item["first200"] is not None and arm not in saved:
                receipt = {
                    **item,
                    "status": "actual_first200_production_updates_observed",
                    "observed_at": observed,
                }
                save(HERE / f"actual-first200-{arm}.json", receipt)
                saved.add(arm)
                print(
                    json.dumps(
                        {
                            "status": receipt["status"],
                            "arm": arm,
                            "GPU_job": item["GPU_job"],
                            "maxima": item["first200"]["maxima"],
                        }
                    ),
                    flush=True,
                )
        if latest:
            save(
                HERE / "actual-latest.json",
                {
                    "observed_at": observed,
                    "arms": latest,
                    "status": "actual_training_progress_read_only",
                },
                replace=True,
            )
        terminal = {
            "COMPLETED",
            "FAILED",
            "TIMEOUT",
            "CANCELLED",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
        }
        if len(latest) == 3 and all(
            str(v["slurm"].get("state", "")).split()[0].rstrip("+") in terminal
            and v["slurm"]["queue_empty"]
            for v in latest.values()
        ):
            save(
                journal / "observer-result.json",
                {
                    "status": "all_original_training_jobs_terminal",
                    "first200_preserved": sorted(saved),
                    "remote_mutations": 0,
                    "observed_at": observed,
                },
            )
            return
        time.sleep(min(args.poll_seconds, max(0, deadline - time.time())))
    save(
        journal / "observer-result.json",
        {
            "status": "finite_observation_deadline",
            "first200_preserved": sorted(saved),
            "remote_mutations": 0,
            "observed_at": utc(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-utc", default=DEADLINE)
    args = parser.parse_args()
    os.umask(0o077)
    require(
        git_file(args.source_git, str(Path(__file__).resolve().relative_to(ROOT)))
        == Path(__file__).read_bytes(),
        "Observer must start from its exact committed source",
    )
    require(30 <= args.poll_seconds <= 300, "Bounded polling")
    require(
        datetime.now(timezone.utc)
        < datetime.fromisoformat(args.deadline_utc)
        <= datetime.fromisoformat(DEADLINE),
        "Bounded observation deadline",
    )
    journal = ROOT / "artifacts" / HERE.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not (journal / "observer-intent.json").exists(), "Original observer already exists")
    try:
        run(args, journal)
    except Exception as error:
        receipt = {
            "status": "read_only_training_observation_failed",
            "observed_at": utc(),
            "error_type": type(error).__name__,
            "remote_mutations": 0,
            "evidence": getattr(error, "evidence", {}),
            "clinical_accuracy_claim": False,
        }
        save(journal / "observer-failure.json", receipt)
        save(HERE / "actual-observation-failure.json", receipt)
        raise


if __name__ == "__main__":
    main()
