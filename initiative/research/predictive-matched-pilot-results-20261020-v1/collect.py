"""Collect original terminal pilot receipts once; no trainer or publication action."""

import argparse
import hashlib
import inspect
import json
import math
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASES = {
    "contrast": dict(
        job="984784",
        root="/tmp/rsna-predictive-contrast-learning-20261020-v1",
        source="a9670168fda9ffe5709216e70216f368ac438553",
        binding="60553b87a178866626157378fa2059e36318c91b6b8b4ec464f65d0a05696e98",
        terminal="GPU-completed.json",
        status="actual_matched_visible_contrast_learning_complete",
        updates=400,
        roles=["control", "normalized"],
    ),
    "generative": dict(
        job="986011",
        root="/tmp/rsna-predictive-generative-routing-20261020-v1",
        source="473a9a94d90890e8a97e7ed46d157dcf417ad173",
        binding="41b7f830956bc3c7b47f6b90453cfe7fe00a2399510c5939379872719a49e4c6",
        terminal="GPU-accepted.json",
        status="actual_matched_C100_G100_through_peak_engineering_complete",
        updates=600,
        roles=["C100", "G100"],
    ),
}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def scheduler(text, job):
    rows = [s.split("|") for s in text.splitlines() if s.split("|")[0] == job]
    if len(rows) != 1:
        raise ValueError("One original scheduler record required")
    identity, state, code, node = rows[0]
    return dict(GPU_job=identity, state=state, exit_code=code, node=node)


def remote_read(case):
    """Runs on the owning node; export only original aggregate terminal bytes."""
    root = Path(case["root"])
    assert os.uname().nodename == "evc27" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def read(name):
        path = root / name
        s = path.stat()
        assert not path.is_symlink() and path.is_file() and s.st_uid == os.getuid()
        assert not s.st_mode & 0o077 and s.st_size <= 2 * 2**20
        raw = path.read_bytes()
        t = path.stat()
        assert (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) == (
            t.st_ino,
            t.st_size,
            t.st_mtime_ns,
            t.st_ctime_ns,
        )
        return raw

    env = {**os.environ, "TZ": "UTC"}
    accounting = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", case["job"], "--format=JobIDRaw,State,ExitCode,NodeList"],
        text=True,
        env=env,
    )
    state = scheduler(accounting, case["job"])
    assert (state["state"], state["exit_code"], state["node"]) == ("COMPLETED", "0:0", "evc27")
    queue = subprocess.check_output(
        ["squeue", "-h", "-j", case["job"], "-o", "%i|%T"], text=True, env=env
    )
    assert not queue.strip()
    binding = read("binding.private.json")
    assert sha(binding) == case["binding"]
    start = json.loads(read("GPU-run-start.json"))
    body = read(case["terminal"])
    value = json.loads(body)
    assert (
        value["status"] == case["status"] and value["actual_optimizer_updates"] == case["updates"]
    )
    for item in (start, value):
        assert item["GPU_job"] == case["job"] and item["source_git"] == case["source"]
        assert item["binding_sha256"] == case["binding"]
    return dict(
        CPU_job=os.environ["SLURM_JOB_ID"],
        scheduler=state,
        accounting_sha256=sha(accounting.encode()),
        queue_empty=True,
        acceptance_sha256=sha(body),
        acceptance_bytes=len(body),
        acceptance_body=body.decode(),
        original_files_written=0,
        weights_or_predictions_collected=False,
    )


def summarize(value, case):
    assert (
        value["status"] == case["status"] and value["actual_optimizer_updates"] == case["updates"]
    )
    if case["job"] == "984784":
        assert set(value["roles"]) == set(case["roles"])
        assert value["actual_normalized_stop1_resume199_completed"] is True
        assert value["new_GPU_full_versus_resume_equality_test_performed"] is False
        assert value["clinical_accuracy_or_automatic_full_promotion"] is False
        histories = {r: value["roles"][r]["history"] for r in case["roles"]}
    else:
        assert [v["role"] for v in value["roles"]] == case["roles"]
        assert value["fresh_initialization_equal"] is True
        assert value["G100_full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
        assert value["clinical_or_held_out_accuracy_claim"] is False
        histories = {v["role"]: v["real_optimizer_history"] for v in value["roles"]}
    out = {}
    fields = (
        "exam_loss",
        "density_loss",
        "gradient_norm",
        "encoder_gradient_norm",
        "head_gradient_norm",
    )
    for role, rows in histories.items():
        assert [r["step"] for r in rows] == list(range(1, 201))
        assert [r["studies"] for r in rows] == [8, 4] * 100
        assert [r["encoder_updated"] for r in rows] == [False] * 3 + [True] * 197
        assert all(math.isfinite(r[k]) for r in rows for k in fields)
        windows = {}
        for name, selected in (
            ("first20", rows[:20]),
            ("peak126to150", rows[125:150]),
            ("last20", rows[-20:]),
        ):
            windows[name] = {
                "update_means": {k: sum(r[k] for r in selected) / len(selected) for k in fields},
                "study_weighted_means": {
                    k: sum(r[k] * r["studies"] for r in selected)
                    / sum(r["studies"] for r in selected)
                    for k in fields
                },
                "clipped_fraction": sum(r["gradient_norm"] > 1 for r in selected) / len(selected),
            }
        out[role] = dict(
            updates=200,
            all_finite=True,
            windows=windows,
            maximum_LR=max(lr for r in rows for lr in r["LR"]),
        )
    return dict(
        GPU_job=case["job"],
        actual_optimizer_updates=case["updates"],
        roles=out,
        clinical_accuracy_claim=False,
        automatic_promotion=False,
        engineering_predictions_used_for_selection=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=CASES)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    case = CASES[args.role]
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton"]
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        case["job"],
        "--format=JobIDRaw,State,ExitCode,NodeList",
    ]
    raw = subprocess.check_output(ssh + [shlex.join(command)], text=True, timeout=30)
    state = scheduler(raw, case["job"])
    if state["state"] in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
        print(json.dumps(dict(status="original_job_not_terminal", **state)))
        return
    assert (state["state"], state["exit_code"], state["node"]) == ("COMPLETED", "0:0", "evc27"), (
        state
    )
    journal = REPO / "artifacts" / (HERE.name + "-" + args.role)
    journal.mkdir(mode=0o700, exist_ok=False)

    def once(path, value):
        with path.open("x") as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)
            f.write("\n")

    source = "import os,json,hashlib,subprocess\nfrom pathlib import Path\n"
    source += "\n".join(inspect.getsource(f) for f in (sha, scheduler, remote_read))
    source += f"\nprint(json.dumps(remote_read({case!r}),sort_keys=True))\n"
    once(
        journal / "intent.json",
        dict(
            case=case,
            program_sha256=sha(source.encode()),
            local_collector_sha256=sha(Path(__file__).read_bytes()),
            automatic_retry=False,
        ),
    )
    command = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc27",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=512M",
        "--time=00:03:00",
        "--immediate=60",
        "--job-name=rsna-" + args.role + "-terminal-read",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        "import sys;exec(compile(sys.stdin.read(),'original_terminal_read','exec'))",
    ]
    returned = subprocess.run(
        ssh + [shlex.join(command)], input=source, text=True, capture_output=True, timeout=240
    )
    once(
        journal / "returned.private.json",
        dict(returncode=returned.returncode, stdout=returned.stdout, stderr=returned.stderr),
    )
    assert returned.returncode == 0, (
        "Reconcile original collection; never repeat an uncertain action"
    )
    collected = json.loads(returned.stdout)
    body = collected.pop("acceptance_body").encode()
    assert (
        sha(body) == collected["acceptance_sha256"] and len(body) == collected["acceptance_bytes"]
    )
    analysis = summarize(json.loads(body), case)
    with (HERE / ("actual-" + args.role + "-accepted.json")).open("xb") as f:
        f.write(body)
    once(HERE / ("actual-" + args.role + "-terminal.json"), collected)
    once(HERE / ("actual-" + args.role + "-analysis.json"), analysis)
    once(journal / "result.json", collected)
    print(json.dumps(analysis, sort_keys=True))


if __name__ == "__main__":
    main()
