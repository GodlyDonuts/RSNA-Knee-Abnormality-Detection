"""CPU-only observations, singleton dispatch and sealed-epoch preservation for C100."""

import json
import os
import re
import runpy
import shlex
import subprocess
from datetime import datetime, timezone

import admission as a

VERIFIED = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
PLACEMENT_SHA = "e8c2a6698450fcfac51baeec6c8eb2ae19599100974a75dcb38b13ba9f9048bf"


def scheduler(job):
    a.require(re.fullmatch(r"\d+", job), "Exact original job ID required")
    raw = subprocess.check_output([
        "sacct", "-X", "-j", job, "-n", "-P",
        "--format=JobIDRaw,State,ExitCode,NodeList,Start,End",
    ], text=True, env={**os.environ, "TZ": "UTC"})
    rows = [s.split("|") for s in raw.splitlines() if s.startswith(job + "|")]
    a.require(len(rows) == 1, "Uncertain Slurm outcome requires read-only reconciliation")
    row = rows[0]
    queue = subprocess.check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i"], text=True)
    return {"GPU_job": job, "job_id": job, "state": row[1], "exit_code": row[2],
            "queue_empty": job not in {s.strip() for s in queue.splitlines()},
            "node": row[3], "start_UTC": row[4], "end_UTC": row[5]}


def status():
    result = {"pilot_scheduler": scheduler("981620"), "branch": "C100"}
    for name, path in (
        ("pilot", a.PILOT / "GPU-accepted.json"),
        ("portable", a.C0 / "portable-accepted.json"),
        ("admitted", a.ROOT / "admitted.json"),
        ("dispatch", a.ROOT / "GPU-submitted.json"),
        ("run_start", a.RUN / "run-start.json"),
        ("terminal", a.RUN / "terminal.json"),
    ):
        if path.exists():
            raw = a.read_ref(a.reference(path))
            result[name] = json.loads(raw)
            result[name + "_sha256"] = a.sha(raw)
    if "dispatch" in result:
        result["full_scheduler"] = scheduler(result["dispatch"]["GPU_job"])
    result["ready_epochs"] = [
        e for e in range(1, 7) if (a.RUN / f"epoch-{e:03d}.json").is_file()
    ]
    result["publications"] = {}
    for e in range(1, 7):
        root = a.RUN / "publication" / f"epoch-{e:03d}"
        if root.exists():
            verified = root / "verified.json"
            result["publications"][str(e)] = (
                json.loads(a.read_ref(a.reference(verified))) if verified.exists()
                else {"status": "uncertain_existing_publication_requires_read_only_reconciliation"}
            )
    return result


def load_binding(source_git):
    prepared = json.loads(a.read_ref(a.reference(a.ROOT / "admitted.json")))
    ref = {"path": str(a.ROOT / "binding.private.json"), "sha256": prepared["binding_sha256"]}
    binding = json.loads(a.read_ref(ref))
    a.require(binding["source_git"] == prepared["source_git"] == source_git
              and binding["branch"] == "C100" and binding["arm"] == "C", "Distinct original source")
    for name, h in binding["code_sha256"].items():
        a.read_ref({"path": str(a.ROOT / "code" / name), "sha256": h})
    pilot = json.loads(a.read_ref(binding["density100_pilot"]))
    a.validate_pilot(pilot, scheduler("981620"))
    return prepared, binding, ref


def original_jobs(v):
    """Only exact original dispatch receipts can retire their charged log inodes."""
    jobs = a.validate_primary_jobs(dict(v.get("original_dispatched_jobs", {})))
    for root in (a.C0,):
        path = root / "dispatch/submitted.json"
        if path.is_file():
            receipt = json.loads(a.read_ref(a.reference(path)))
            intent = json.loads(a.read_ref(a.reference(root / "dispatch/intent.json")))
            expected = {"B", "C", "C0"}
            a.require(intent["owner_git"] == a.PRIMARY_SOURCE and intent["source_git"] == a.BASE
                      and set(intent["arms"]) == expected
                      and receipt["status"] == "actual_predictive_placement_jobs_submitted_once"
                      and receipt["source_git"] == a.BASE,
                      "Stale primary dispatch cannot supply paid-log credit")
            a.validate_primary_jobs(receipt["jobs"], expected)
            for arm, job in receipt["jobs"].items():
                a.require(arm not in jobs or jobs[arm] == job, "Original job ownership differs")
                jobs[arm] = job
    return a.validate_primary_jobs(jobs)


def paid_primary_logs(source, jobs, root=None):
    ns = {}
    a.require(a.sha(source.encode()) == PLACEMENT_SHA,
              "Pinned original log-accounting source required")
    exec(compile(source, "original_placement_file_accounting", "exec"), ns)
    a.validate_primary_jobs(jobs)
    compatible = {arm: dict(job, node="evc27" if arm == "C0" else "evc33")
                  for arm, job in jobs.items()}
    # Reuse the pinned exact job/path/inode checker; never store this historical
    # node-field projection as an actual dispatch or input proof.
    return ns["paid_shared_logs"](compatible, root) if root is not None else ns["paid_shared_logs"](compatible)


def shared(v):
    a.require(v["placement_sha256"] == PLACEMENT_SHA, "Original placement source binding changed")
    paid = paid_primary_logs(v["placement_source"], original_jobs(v))
    quota = a.shared_budget(subprocess.check_output(
        ["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True,
    ), paid["files"])
    return {**quota, "paid_original_logs": paid}


def script(binding_sha, primary_jobs):
    a.require(len(primary_jobs) == 3 and len(set(primary_jobs)) == 3
              and all(re.fullmatch(r"\d+", job) for job in primary_jobs),
              "All three original primary job dependencies required")
    command = [
        a.PYTHON, "-B", str(a.ROOT / "code/worker.py"), "--binding",
        str(a.ROOT / "binding.private.json"), "--binding-sha256", binding_sha,
    ]
    return "\n".join([
        "#!/bin/bash", "#SBATCH --account=skattel", "#SBATCH --partition=normal",
        "#SBATCH --qos=normal", "#SBATCH --nodelist=evc27", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=2", "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
        "#SBATCH --mem=32G", "#SBATCH --time=08:00:00",
        "#SBATCH --deadline=2026-10-22T04:00:00",
        "#SBATCH --dependency=after:" + ":".join(primary_jobs),
        "#SBATCH --job-name=rsna-predictive-full-C100",
        "#SBATCH --chdir=" + str(a.ROOT),
        "#SBATCH --output=" + str(a.ROOT / "GPU-%j.out"),
        "#SBATCH --error=" + str(a.ROOT / "GPU-%j.err"),
        "set -euo pipefail", "umask 077",
        "export TZ=UTC PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 "
        "MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "exec " + shlex.join(command), "",
    ])


def dispatch(v):
    a.require(datetime.now(timezone.utc) < a.LAST_DISPATCH, "Too late for a separate full C100")
    prepared, binding, ref = load_binding(v["source_git"])
    a.require(not (a.ROOT / "GPU-intent.json").exists() and not a.RUN.exists(),
              "Existing C100 dispatch/output requires read-only reconciliation")
    jobs = original_jobs(v)
    a.require(set(jobs) == {"B", "C", "C0"}, "Keep all three primary dispatches ahead of C100")
    c0 = jobs["C0"]
    a.require(c0["source_git"] == a.BASE and c0["node"] == "evc27"
              and c0["arm"] == "C0" and c0["job_name"] == "rsna-predictive-full2-C0",
              "Actual original C0 dispatcher receipt required")
    for primary in jobs.values():
        state = scheduler(primary["GPU_job"])
        a.require(state["state"] in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"},
                  "Review any failed primary before reserving a secondary full run")
    node_text = subprocess.check_output(["scontrol", "show", "node", "evc27", "-o"], text=True)
    node = dict(t.split("=", 1) for t in node_text.split() if "=" in t)
    a.require(node["NodeName"] == "evc27" and "gpu:nvidia_h100_pcie:2" in node["Gres"]
              and re.search(r"(?:^|,)gres/gpu=2(?:,|$)", node["CfgTRES"]),
              "Only the admitted physical two-H100 node")
    portable = json.loads(a.read_ref(a.reference(a.C0 / "portable-accepted.json")))
    budget = {"node": a.fresh_node(portable), "shared": shared(v)}
    dependencies = [jobs[arm]["GPU_job"] for arm in ("C", "B", "C0")]
    body = script(ref["sha256"], dependencies)
    a.once(a.ROOT / "GPU-intent.json", {
        "source_git": v["source_git"], "binding_sha256": ref["sha256"],
        "script_sha256": a.sha(body.encode()), "fresh_capacity": budget,
        "original_C0_job": c0, "original_primary_jobs": jobs, "automatic_retry": False,
    })
    result = subprocess.run(["sbatch", "--parsable"], input=body, text=True,
                            capture_output=True, timeout=60, check=False,
                            env={**os.environ, "TZ": "UTC"})
    a.once(a.ROOT / "GPU-transport.private.json",
           {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    a.require(result.returncode == 0 and re.fullmatch(r"\d+(?:;[^\s]+)?\s*", result.stdout),
              "Uncertain singleton submission: reconcile read-only, never retry")
    receipt = {
        "status": "actual_single_full_C100_GPU_submitted", "branch": "C100", "arm": "C",
        "GPU_job": result.stdout.strip().split(";")[0], "source_git": v["source_git"],
        "node": "evc27", "binding_sha256": ref["sha256"],
        "density100_pilot_sha256": prepared["density100_pilot_sha256"],
        "script_sha256": a.sha(body.encode()), "original_C0_job": c0["GPU_job"],
        "original_primary_after_dependencies": dependencies,
        "runtime_cap_seconds": 28800, "new_shared_files": 0,
        "clinical_or_Kaggle_result": False,
    }
    a.once(a.ROOT / "GPU-submitted.json", receipt)
    return receipt


def publish(v):
    _, binding, ref = load_binding(v["source_git"])
    epoch = v["epoch"]
    a.require(type(epoch) is int and 1 <= epoch <= 6, "Original immutable epoch required")
    dispatched = json.loads(a.read_ref(a.reference(a.ROOT / "GPU-submitted.json")))
    side = json.loads(a.read_ref(a.reference(a.RUN / f"epoch-{epoch:03d}.json")))
    a.require(side["GPU_job"] == dispatched["GPU_job"], "Epoch belongs to another GPU job")
    publisher = a.ROOT / "code/publish_epoch.py"
    a.require(a.sha(publisher.read_bytes()) == binding["code_sha256"]["publish_epoch.py"],
              "Frozen C100 publisher required")
    ns = runpy.run_path(str(publisher), run_name="distinct_C100_epoch_publisher")
    payload = {"mode": "inspect", "binding": ref, "epoch": epoch, "source_git": v["source_git"]}
    state = ns["main"](payload)
    if state["status"] == VERIFIED:
        return state
    a.require(state["status"] == "sealed_epoch_ready_no_publication_intent",
              "Existing uncertain publication must not repeat")
    portable = json.loads(a.read_ref(a.reference(a.C0 / "portable-accepted.json")))
    node, quota = a.fresh_node(portable), shared(v)
    funding = {
        "status": "actual_predictive_epoch_publication_storage_funded",
        "CPU_job": os.environ["SLURM_JOB_ID"], "node": "evc27", "arm": "C", "branch": "C100",
        "run_root": str(a.RUN), "epoch": epoch, "arm_cap_bytes": a.FIT_CAP,
        "standing_free_reserve_bytes": 20 * a.GIB,
        "other_unpaid_node_liabilities_bytes": node["other_unpaid_node_liabilities_bytes"],
        "shared_unpaid_bytes": quota["unpaid_bytes"], "shared_unpaid_files": quota["unpaid_files"],
        "all_other_obligations_retained": True, "no_deletion_credit": True,
    }
    path = a.ROOT / f"publication-funding-{epoch:03d}.json"
    a.once(path, funding)
    payload.update(mode="publish", publication_funding=a.reference(path), token=v["token"])
    return ns["main"](payload)


def main(v):
    os.umask(0o077)
    a.require(os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
              and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "One-CPU zero-GPU operation")
    return {"status": lambda _: status(), "dispatch": dispatch, "publish": publish}[v["phase"]](v)
