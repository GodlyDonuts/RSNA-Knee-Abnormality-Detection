"""One explicit pointwise deployment phase, frozen source and no uncertain retries."""

import argparse
import ast
import hashlib
import importlib.util
import json
import shlex
import subprocess
import time
from pathlib import Path

R = Path(__file__).resolve().parents[3]
D = Path(__file__).resolve().parent
L = R / "artifacts" / D.name
WAITING = {
    "waiting_existing_canonical_claim_no_mutation",
    "waiting_actual_GPU_capacity_or_budget_no_mutation",
    "waiting_authoritative_live_account_allowance_no_mutation",
    "waiting_original_dataset_ingestion_no_mutation",
}
MUTATIONS = {"stage", "dataset", "prepare", "push", "submit"}


def git_bytes(pin, path):
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(R).as_posix()], cwd=R
    )


def controls(pin):
    own = {}
    for name in (
        "adapter.py",
        "helpers.py",
        "policy_functions.py",
        "run_once.py",
        "follow_once.py",
        "engineering-decision.json",
        "config.json",
    ):
        path = D / name
        own[name] = git_bytes(pin, path)
        assert own[name] == path.read_bytes(), "Exact committed control source required"
    accepted_config = json.loads(own["config.json"])
    assert (
        accepted_config["runtime_plan"]["actual_same_graph_timing_receipt_sha256"]
        == hashlib.sha256(own["engineering-decision.json"]).hexdigest()
    )
    ns = {"__name__": "frozen_pointwise_adapter"}
    exec(compile(own["adapter.py"], "adapter", "exec"), ns)
    origin = R / "initiative/research/S6-Kaggle-production-route-20261019-v1"
    policy_functions = own["policy_functions.py"].decode()
    parts = {
        n.name: ast.get_source_segment(policy_functions, n)
        for n in ast.parse(policy_functions).body
        if isinstance(n, ast.FunctionDef)
    }
    sources = {
        n + ".py": git_bytes(pin, R / "scripts/kaggle" / (n + ".py")).decode()
        for n in ("workflow", "components", "gpu_capacity", "output_csv", "publication")
    }
    assert (
        hashlib.sha256(sources["workflow.py"].encode()).hexdigest()
        == "2d3074fbf677285687e80c0e6b58721407bc61ddea39a990797da0e1d4e3b900"
    )
    assert (
        hashlib.sha256(sources["gpu_capacity.py"].encode()).hexdigest()
        == "cd4dfb8261b6dba74e4e78ee6177a4027a19963560bc6fa3d244e6b878199092"
    )
    sources["policy.py"] = ns["policy"](
        git_bytes(pin, origin / "serving_policy.py.txt").decode(), parts
    )
    sources["program.py"] = ns["program"](
        git_bytes(pin, origin / "serving_cloud.py.txt").decode(), own["helpers.py"].decode()
    )
    return sources


def run(phase, mode, pin, config, socket):
    assert phase in {
        "inspect",
        "stage",
        "dataset",
        "dataset-reconcile",
        "prepare",
        "push",
        "status",
        "submit",
        "outcome",
    }
    assert mode in {"qualification", "competition"}
    sources = controls(pin)
    value = json.loads(Path(config).read_text())
    assert value == json.loads((D / "config.json").read_text()), (
        "Exact frozen engineering/runtime/funding decision required"
    )
    assert not any("token" in k.lower() or "password" in k.lower() for k in value)
    value.update(
        phase=phase,
        epoch=3,
        mode=mode,
        source_git=pin,
        source_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
        original_compiled=json.loads(
            (
                R
                / "initiative/research/teacher-pair-final-serving-20261020-v2/actual-pointwise-compile.json"
            ).read_text()
        ),
    )
    if phase == "stage":
        value["sources"] = sources
    elif phase != "inspect":
        env = R / ".env"
        assert env.stat().st_mode & 0o077 == 0
        entries = {
            k.strip(): z.strip().strip("\"'")
            for line in env.read_text().splitlines()
            for k, sep, z in [line.partition("=")]
            if sep and not line.lstrip().startswith("#")
        }
        value["token"] = entries["KAGGLE_API_TOKEN"]
    L.mkdir(parents=True, mode=0o700, exist_ok=True)
    stem = phase + "-" + mode
    guard, result = L / (stem + "-intent.json"), L / (stem + "-result.json")
    if phase in MUTATIONS:
        if guard.exists():
            assert result.exists() and json.loads(result.read_text())["status"] in WAITING, (
                "Prior uncertain or completed phase cannot repeat"
            )
            result.unlink()
        else:
            with guard.open("x") as handle:
                json.dump({"source_git": pin, "phase": phase, "automatic_retry": False}, handle)
    transport_path = (
        R / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    )
    transport = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("frozen_pointwise_transport", loader=None)
    )
    exec(compile(git_bytes(pin, transport_path), "transport", "exec"), transport.__dict__)
    command = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc48",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=2G",
        "--time=00:30:00",
        "--job-name=rsna-pointwise-serving-" + phase,
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        transport.COMMAND,
    ]
    returned = subprocess.run(
        ["ssh", "-S", socket, "-o", "BatchMode=yes", "newton", shlex.join(command)],
        input=transport.envelope(sources["program.py"], value),
        capture_output=True,
        text=True,
        timeout=1900,
    )
    (L / (stem + "-" + str(time.time_ns()) + ".json")).write_text(
        json.dumps(
            dict(returncode=returned.returncode, stdout=returned.stdout, stderr=returned.stderr),
            indent=2,
        )
        + "\n"
    )
    assert returned.returncode == 0, "Original phase requires read-only reconciliation; never retry"
    actual = json.loads(returned.stdout)
    result.write_text(json.dumps(actual, indent=2) + "\n")
    if phase == "stage" and actual["status"] == "actual_twoT4_epoch_serving_account_control_staged":
        with (D / "actual-staged.json").open("x") as handle:
            json.dump(actual, handle, indent=2)
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase")
    parser.add_argument("--mode", default="qualification")
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ssh-socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    print(
        json.dumps(run(args.phase, args.mode, args.source_git, args.config, args.ssh_socket)),
        flush=True,
    )


if __name__ == "__main__":
    main()
