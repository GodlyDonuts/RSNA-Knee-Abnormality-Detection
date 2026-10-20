"""One zero-GPU owning-node terminal read; writes no remote files."""

import json
import os
import stat
import subprocess
from pathlib import Path

import observer_contract as c


def read(path, maximum=2**20):
    before = path.lstat()
    c.require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
              and not before.st_mode & 0o077 and before.st_size <= maximum,
              "Bounded owner-only original receipt required")
    body = path.read_bytes()
    after = path.lstat()
    def signature(s):
        return s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    c.require(signature(before) == signature(after), "Original receipt changed while reading")
    return body


def main():
    c.require(os.uname().nodename == c.NODE and not os.environ.get("SLURM_JOB_GPUS")
              and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
              and os.environ["SLURM_CPUS_PER_TASK"] == "1", "One owning-node CPU and zero GPUs")
    accounting = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", c.JOB, "--format=JobIDRaw,State,ExitCode,NodeList"],
        text=True)
    queue = subprocess.check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"], text=True)
    terminal = c.scheduler(accounting, queue)
    c.require(terminal["state"] in c.TERMINAL and terminal["live_queue_empty"],
              "Actual terminal and empty live queue required")
    root = Path(c.ROOT)
    binding = read(root / "binding.private.json")
    c.require(c.sha(binding) == c.BINDING, "Original private binding changed")
    prepared = json.loads(read(root / "prepared.json"))
    c.require(prepared["binding_sha256"] == c.BINDING and prepared["source_git"] == c.SOURCE
              and prepared["CPU_job"] == "981619"
              and prepared["status"] == "actual_C100_original_contexts_CPU_admitted",
              "Original CPU admission differs")
    out = {
        "status": "actual_C100_original_terminal_read_only_collected",
        "CPU_job": os.environ["SLURM_JOB_ID"], "GPU_job": c.JOB, "node": c.NODE,
        "source_git": c.SOURCE, "binding_sha256": c.BINDING, "scheduler": terminal,
        "original_files_written": 0, "GPU_allocated": False, "automatic_promotion": False,
    }
    started = root / "GPU-run-start.json"
    out["original_run_started"] = started.exists()
    if started.exists():
        body = read(started)
        value = json.loads(body)
        c.require(value["GPU_job"] == c.JOB and value["source_git"] == c.SOURCE
                  and value["branch"] == "C100", "Original run marker differs")
        out["original_run_start_sha256"] = c.sha(body)
    accepted = root / "GPU-accepted.json"
    if terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0":
        c.require(out["original_run_started"], "Original successful run marker required")
        body = read(accepted)
        value = json.loads(body)
        c.acceptance(value)
        original_terminal = read(root / "engineering/C/full/terminal.json")
        c.require(json.loads(original_terminal) == value["terminal"],
                  "Outer acceptance and original terminal differ")
        c.require(c.sha(read(root / "scientific-decision.json")) ==
                  value["scientific_decision_sha256"], "Original scientific decision differs")
        out.update(
            acceptance_sha256=c.sha(body), acceptance_bytes=len(body),
            acceptance_body=body.decode(), terminal_sha256=c.sha(original_terminal),
        )
    else:
        # Even a late accepted file cannot override a failed scheduler outcome.
        out["original_acceptance_present"] = accepted.exists()
        out["failure_logs"] = {}
        for suffix in ("out", "err"):
            path = root / f"GPU-{c.JOB}.{suffix}"
            if path.exists():
                body = read(path, 4 * 2**20)
                out["failure_logs"][suffix] = {"bytes": len(body), "sha256": c.sha(body)}
    print(json.dumps(out, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
