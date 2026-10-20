"""One explicitly admitted paired pilot: inspect, prepare, submit; never retry."""

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL_SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


B = module("generative_worker_builder", HERE / "worker_builder.py")
P = module("generative_CPU_admission", HERE / "prepare.py")
PRIMARY_SOURCE, PRIMARY_OWNER_PID = P.PRIMARY_SOURCE, P.PRIMARY_OWNER_PID
PRIMARY_AUDIT = REPO / P.PRIMARY_AUDIT_REL
PRIMARY_JOURNAL = REPO / "artifacts" / PRIMARY_AUDIT.name / "finite-owner"


def frozen(pin, path):
    assert re.fullmatch("[0-9a-f]{40}", pin)
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=REPO)


def freeze(pin):
    """Pure local source assembly; useful to construct the later root decision."""
    own = {}
    for name in ("worker_builder.py", "prepare.py", "driver.py", "coordinate.py"):
        raw = frozen(pin, (HERE / name).relative_to(REPO).as_posix())
        assert raw == (HERE / name).read_bytes(), "Exact committed controls required"
        own[name] = raw.decode()
    names = subprocess.check_output([
        "git", "ls-tree", "-r", "--name-only", B.SCIENCE, "--", "rsna/competitive/predictive",
    ], cwd=REPO, text=True).splitlines()
    files = {name: frozen(B.SCIENCE, name).decode() for name in names if name.endswith(".py")}
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    routing = "rsna/competitive/predictive/generative_routing.py"
    files[routing] = frozen(pin, routing).decode()
    assert B.sha(files[routing].encode()) == B.ROUTING_SHA
    original = frozen(B.SCIENCE,
                      "initiative/research/predictive-stable-warmup-20261020-v1/worker.py.txt").decode()
    files.update({"worker.py": B.build(original), "worker_builder.py": own["worker_builder.py"],
                  "driver.py": own["driver.py"]})
    return dict(source_git=pin, files=files,
                sha256={n: B.sha(body.encode()) for n, body in files.items()},
                control_sha256={n: B.sha(body.encode()) for n, body in own.items()}), own


def original_jobs(audit=PRIMARY_AUDIT, journal=PRIMARY_JOURNAL, continuation=None):
    """Read exact all27 phase proofs through the explicit startup continuation."""
    continuation = REPO / P.CONTINUATION_REL if continuation is None else continuation
    raw = (journal / "owner-intent.json").read_bytes()
    P.validate_primary_ownership(raw, continuation.read_bytes())
    assert json.loads(raw)["science_git"] == FULL_SCIENCE
    path = audit / "actual-owner-dispatch-all27.json"
    if not path.exists():
        path = audit / "actual-known-dispatches-all27.json"
        if not path.exists():
            return {}
        receipt = json.loads(path.read_bytes())
        result = json.loads((journal / "known-dispatches-all27.json").read_bytes())
        recovered = json.loads((audit / "actual-dispatch-reconciled.json").read_bytes())
        assert recovered == json.loads((journal / "dispatch-reconciled.json").read_bytes())
        assert recovered["status"] == "actual_all27_original_dispatch_reconciled_readonly"
        assert recovered["jobs"] == receipt["jobs"] and recovered["new_GPU_jobs"] == 0
        assert isinstance(recovered["submission_completeness_known"], bool)
    else:
        receipt = json.loads(path.read_bytes())
        result = json.loads((journal / "dispatch-all27-result.json").read_bytes())
    intent = json.loads((journal / "dispatch-all27-intent.json").read_bytes())
    assert receipt == result and intent["mode"] == "dispatch" and intent["node"] == "evc27"
    assert intent["placement_source_git"] == PRIMARY_SOURCE
    assert re.fullmatch("[0-9a-f]{64}", intent["program_sha256"])
    assert receipt["status"] == "actual_predictive_placement_jobs_submitted_once"
    assert receipt["source_git"] == FULL_SCIENCE and set(receipt["jobs"]) == {"B", "C", "C0"}
    assert len({j["GPU_job"] for j in receipt["jobs"].values()}) == 3
    assert all(j["node"] == "evc27" and j["arm"] == arm and j["source_git"] == FULL_SCIENCE
               for arm, j in receipt["jobs"].items())
    return receipt["jobs"]


def script(decision):
    finish = P.stamp(decision["finish_by_UTC"]).strftime("%Y-%m-%dT%H:%M:%S")
    return f"""#!/bin/bash
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodelist=evc27
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --deadline={finish}
#SBATCH --chdir={B.ROOT}
#SBATCH --job-name=rsna-predictive-G100-matched200
#SBATCH --output={B.ROOT}/GPU-%j.out
#SBATCH --error={B.ROOT}/GPU-%j.err
set -euo pipefail
umask 077
export TZ=UTC PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec {P.PORTABLE}/runtime/bin/python -B -c 'import sys;sys.path[:0]=["{B.ROOT}/code","{P.PORTABLE}/overlay"];exec(compile(open("{B.ROOT}/code/driver.py").read(),"matched_generative_driver","exec"))'
"""


def invoke(ssh, transport, source, payload, run=subprocess.run):
    command = ["srun", "--account=skattel", "--partition=normal", "--qos=normal",
               "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
               "--mem=1G", "--time=00:05:00", "--immediate=60",
               "--job-name=rsna-G100-matched-admit", "env", "CUDA_VISIBLE_DEVICES=",
               "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", MANAGEMENT,
               "-B", "-c", transport.COMMAND]
    return run([*ssh, shlex.join(command)], input=transport.envelope(source, payload),
               capture_output=True, text=True, timeout=420, check=False)


def execute(payload, own, decision, funding, journal, ssh, transport,
            *, run=subprocess.run, now=lambda: datetime.now(timezone.utc)):
    """One bounded operation; preflight is read-only and all uncertainty is retained."""
    assert not journal.exists(), "Existing coordinator journal requires read-only reconciliation"
    _, policy = P.controls(payload["files"], payload["sha256"])
    once = policy["once"]
    P.decision_time(decision, funding, now())
    preflight = invoke(ssh, transport, own["prepare.py"], {**payload, "mode": "inspect"}, run)
    assert preflight.returncode == 0, "Read-only preflight did not pass; no mutation admitted"
    observed = json.loads(preflight.stdout)
    assert observed["status"] == "actual_matched_generative_routing_readonly_preflight_passed"
    assert observed["source_git"] == payload["source_git"]
    assert 0 <= (now() - P.stamp(observed["observed_at_UTC"])).total_seconds() <= 600
    P.decision_time(decision, funding, now())
    journal.mkdir(mode=0o700, exist_ok=False)
    once(journal / "preflight.json", observed)
    once(journal / "intent.json", dict(source_git=payload["source_git"],
         payload_sha256=B.sha(json.dumps(payload, sort_keys=True).encode()),
         control_sha256=payload["control_sha256"], preflight=observed,
         maximum_GPU_jobs=1, actual_optimizer_updates=600, automatic_retry=False,
         new_shared_files=0, automatic_full_training_promotion=False))
    admitted = invoke(ssh, transport, own["prepare.py"],
                      {**payload, "mode": "admit", "preflight": observed}, run)
    once(journal / "prepare-transport.private.json", dict(returncode=admitted.returncode,
         stdout=admitted.stdout, stderr=admitted.stderr))
    assert admitted.returncode == 0, "Uncertain CPU admission: reconcile, never retry"
    result = json.loads(admitted.stdout)
    assert result["status"] == "actual_matched_generative_routing_CPU_admitted"
    assert result["source_git"] == payload["source_git"] and re.fullmatch("[0-9a-f]{64}", result["binding_sha256"])
    once(journal / "prepare-result.json", result)
    P.decision_time(decision, funding, now())
    body = script(decision)
    once(journal / "GPU-intent.json", dict(source_git=payload["source_git"],
         binding_sha256=result["binding_sha256"], script_sha256=B.sha(body.encode()),
         maximum_GPU_seconds=5400, automatic_retry=False))
    submitted = run([*ssh, "TZ=UTC sbatch --parsable"], input=body, capture_output=True,
                    text=True, timeout=60, check=False)
    once(journal / "GPU-transport.private.json", dict(returncode=submitted.returncode,
         stdout=submitted.stdout, stderr=submitted.stderr))
    assert submitted.returncode == 0, "Uncertain GPU submission: reconcile, never retry"
    job = submitted.stdout.strip().split(";")[0]
    assert re.fullmatch("[0-9]+", job)
    receipt = dict(status="actual_single_matched_C100_G100_GPU_submitted", GPU_job=job,
                   source_git=payload["source_git"], CPU_admission_job=result["CPU_job"],
                   binding_sha256=result["binding_sha256"], script_sha256=B.sha(body.encode()),
                   node="evc27", actual_optimizer_updates=600, maximum_seconds=5400,
                   new_shared_files=0, automatic_full_training_promotion=False)
    once(journal / "GPU-result.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--funding", type=Path, required=True)
    parser.add_argument("--pilot-acceptance", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    payload, own = freeze(args.source_git)
    decision, funding = (json.loads(p.read_bytes()) for p in (args.decision, args.funding))
    assert funding.get("original_dispatched_jobs", {}) == original_jobs()
    payload.update(primary_control=P.primary_control(), decision_body=args.decision.read_text(), funding_body=args.funding.read_text(),
                   pilot_acceptance_sha256=B.sha(args.pilot_acceptance.read_bytes()))
    # The actual original acceptance is independently read and matched on evc27.
    transport_path = "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    transport = importlib.util.module_from_spec(importlib.util.spec_from_loader("matched_transport", loader=None))
    exec(compile(frozen(B.SCIENCE, transport_path), "frozen_stdin_transport", "exec"), transport.__dict__)
    receipt = execute(payload, own, decision, funding, REPO / "artifacts" / HERE.name,
                      ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton"], transport)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
