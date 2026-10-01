"""Run one immutable, source-bound full-cohort sampling arm on node-local scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

SOURCE_GIT = "7ac7a9779d3c8bfd756a44eceb6414e9bb763b85"
BASE = Path("/home/sa305415/rsna-knee")
PREPARATION = BASE / "stratified-production-7ac7a97-preparation-v1"
PILOT = BASE / "stratified-pilot-7ac7a97-v1"
CHECKPOINT_CAP = 128 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stat(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular immutable asset is required")
    s = path.stat()
    return dict(
        dev=s.st_dev, ino=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns
    )


class StopForwarder:
    """Forward scheduler requests to the supervisor, which checkpoints its workers."""

    def __init__(self):
        self.requested = False
        self.child = None
        self.sent = False

    def request(self, _signal, _frame):
        self.requested = True
        self.forward()

    def forward(self):
        if (
            self.requested
            and self.child is not None
            and self.child.poll() is None
            and not self.sent
        ):
            try:
                os.kill(self.child.pid, signal.SIGUSR1)
                self.sent = True
            except ProcessLookupError:
                pass

    def install(self):
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request)

    def supervise(self, command, cwd, output):
        self.child = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        try:
            while self.child.poll() is None:
                # Never treat an incomplete temporary checkpoint as an accepted epoch.
                paths = [
                    output / "checkpoint.pt",
                    *sorted((output / "epoch-checkpoints").glob("*.pt")),
                ]
                if any(p.exists() and p.stat().st_size >= CHECKPOINT_CAP for p in paths):
                    self.requested = True
                self.forward()
                time.sleep(0.2)
            return self.child.wait()
        finally:
            if self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.child = None


def terminal(job):
    return subprocess.check_output(
        ["sacct", "-X", "--noheader", "-j", job, "--format=JobIDRaw,State,ExitCode", "--parsable2"],
        text=True,
    ).strip()


def validate_prerequisites(control):
    manifest_path = control / "files.json"
    manifest = json.loads(manifest_path.read_text())
    for name, row in manifest["files"].items():
        p = control / name
        if stat(p)["bytes"] != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("Pinned production payload changed")
    cpu_path = PREPARATION / "cpu-prerequisites-verified.json"
    proof = json.loads(cpu_path.read_text())
    pilot = json.loads((PILOT / "terminal-verified.json").read_text())
    if (
        terminal("866694") != "866694|COMPLETED|0:0"
        or terminal("866624") != "866624|COMPLETED|0:0"
        or proof["status"] != "full_cohort_source_assets_and_sampling_CPU_preparation_verified"
        or proof["source_git"] != SOURCE_GIT
        or proof["data"]["full_cohort_metadata"]["studies"] != 3403
        or pilot["status"]
        != "sampling_GPU_pilot_initialization_and_two_epoch_exact_recovery_passed"
    ):
        raise ValueError("Actual full CPU acceptance and GPU recovery pilot are required")
    if (
        sha(cpu_path) != manifest["CPU_proof_sha256"]
        or sha(PILOT / "terminal-verified.json") != manifest["pilot_proof_sha256"]
    ):
        raise ValueError("Prerequisite receipts changed")
    source = json.loads((control / "source.json").read_text())
    if source["source_git"] != SOURCE_GIT or source["files"] != proof["source_files"]:
        raise ValueError("Scientific source does not match the actual full CPU proof")
    if sha(PILOT / "source.pack") != source["pack_sha256"]:
        raise ValueError("Scientific source pack changed")
    for path, row in proof["assets"].items():
        if stat(Path(path)) != row["stat"]:
            raise ValueError("CPU accepted asset was replaced or modified")
    return manifest, source, proof


def run(control, policy):
    if policy not in ("uniform", "stratified"):
        raise ValueError("Unknown fixed sampling role")
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdecimal()
        or not os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
    ):
        raise ValueError("One H100/two CPU Slurm allocation required")
    manifest, source, proof = validate_prerequisites(control)
    output = Path(manifest["runs"][policy])
    if output.exists():
        raise FileExistsError("Existing production output requires exact read-only recovery review")
    stopper = StopForwarder()
    stopper.install()
    code = environment_root = cache_root = None
    try:
        code = Path(tempfile.mkdtemp(prefix="rsna-sampling-production-" + job + "-", dir="/tmp"))
        if (
            code.stat().st_dev == Path.home().stat().st_dev
            or shutil.disk_usage(code).free < 120 * 1024**3
        ):
            raise ValueError("Node-local scratch with the accepted free-space floor is required")
        subprocess.run(["git", "init", "-q", str(code)], check=True)
        subprocess.run(
            ["git", "-C", str(code), "unpack-objects", "-q"],
            input=(PILOT / "source.pack").read_bytes(),
            check=True,
        )
        (code / ".git/objects/info/alternates").write_text(source["readonly_alternate"] + "\n")
        (code / ".git/shallow").write_text(SOURCE_GIT + "\n")
        subprocess.run(
            ["git", "-C", str(code), "config", "core.sparseCheckout", "true"], check=True
        )
        (code / ".git/info/sparse-checkout").write_text(
            "\n".join("/" + n for n in source["files"]) + "\n"
        )
        subprocess.run(
            ["git", "-C", str(code), "checkout", "-q", "--detach", SOURCE_GIT], check=True
        )
        for name, row in source["files"].items():
            if stat(code / name)["bytes"] != row["bytes"] or sha(code / name) != row["sha256"]:
                raise ValueError("Materialized source differs from the frozen scientific files")
        if stopper.requested:
            return 75
        env_archive = (
            BASE
            / "kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar"
        )
        cache_archive = BASE / "cache/cache256-v3-official-recovery-v1.tar"

        def extract(helper, archive, count, unpacked, extra=()):
            return json.loads(
                subprocess.check_output(
                    [
                        "/usr/bin/python3.12",
                        "-B",
                        str(code / ("scripts/newton/" + helper)),
                        "extract",
                        "--archive",
                        str(archive),
                        "--sha256",
                        proof["assets"][str(archive)]["sha256"],
                        "--expected-files",
                        str(count),
                        "--expected-bytes",
                        str(unpacked),
                        "--scratch-base",
                        "/tmp",
                        "--min-free-gib",
                        "20",
                        *extra,
                    ],
                    text=True,
                )
            )

        env = extract(
            "runtime_environment.py",
            env_archive,
            40403,
            6322691335,
            ["--shared-python", "/usr/bin/python3.12"],
        )
        environment_root = Path(env["runtime_root"])
        python = str(environment_root / "environment/bin/python")
        if stopper.requested:
            return 75
        cached = extract("runtime_archive.py", cache_archive, 48772, 82794366615)
        cache_root = Path(cached["runtime_root"])
        if stopper.requested:
            return 75
        os.environ.update(PYTHONPATH=str(code), TRITON_CACHE_DIR=str(cache_root / "triton-cache"))
        subprocess.run(
            [
                python,
                "-B",
                "-c",
                'import torch;assert torch.cuda.device_count()==1;torch.cuda.set_device(0);assert torch.ones(8,device="cuda").sum().item()==8;torch.cuda.synchronize()',
            ],
            check=True,
        )
        inp = BASE / "code-stopfix-1f486765"
        studies = inp / "initiative/releases/report-supervision-v1/studies.parquet"
        series = (
            inp / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
        )
        wrapper = str(code / "scripts/newton/enforce_deterministic_runtime.py")
        init_path = control / (policy + "-initialization-" + job + ".json")
        subprocess.run(
            [
                python,
                "-B",
                wrapper,
                "--script",
                str(control / "verify_initialization.py"),
                "--control",
                str(control / "uniform-config.json"),
                "--candidate",
                str(control / "stratified-config.json"),
                "--studies",
                str(studies),
                "--series",
                str(series),
                "--cache",
                str(cache_root / "cache"),
                "--outer-predictions",
                str(PILOT / "uniform-outer_predictions.csv"),
                "--output",
                str(init_path),
            ],
            cwd=code,
            check=True,
        )
        initial = json.loads(init_path.read_text())
        pilot_init = json.loads((PILOT / "initialization-verified.json").read_text())
        if (
            initial["initial_state"] != pilot_init["initial_state"]
            or initial["initial_logits_sha256"] != pilot_init["initial_logits_sha256"]
        ):
            raise ValueError("Full-run initialization differs from the accepted pilot")
        if stopper.requested:
            return 75
        output.mkdir(mode=0o700)
        save(
            output / "sampling-input-provenance.json",
            {
                "status": "full_sampling_inputs_and_actual_initialization_accepted",
                "source_git": SOURCE_GIT,
                "policy": policy,
                "job_id": job,
                "node": os.uname().nodename,
                "source_files": source["files"],
                "source_manifest_sha256": sha(control / "source.json"),
                "payload_manifest_sha256": sha(control / "files.json"),
                "CPU_proof_sha256": sha(PREPARATION / "cpu-prerequisites-verified.json"),
                "pilot_proof_sha256": sha(PILOT / "terminal-verified.json"),
                "assets": proof["assets"],
                "initialization": initial,
                "config_sha256": sha(control / (policy + "-config.json")),
                "strict_determinism": True,
                "base_weights_reused_not_diagnostic_checkpoint": True,
                "cache_side": 256,
                "cache_members": 48772,
                "training_studies": 2247,
                "outer_studies": 1156,
                "checkpoint_cap_bytes": CHECKPOINT_CAP,
                "environment": env,
                "cache": cached,
                "evaluation_policy": "uniform; fixed_final_epoch; no_outer_epoch_selection",
                "accepted_data_masks_or_folds_changed": False,
                "GPU_header_receipt_sha256": sha(
                    control / ("node-header-verified-" + job + ".json")
                ),
            },
        )
        command = [
            python,
            "-B",
            str(code / "scripts/newton/supervise.py"),
            "--grace-seconds",
            "240",
            "--checkpoint",
            str(output / "checkpoint.pt"),
            "--",
            python,
            "-B",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            wrapper,
            "--module",
            "rsna.competitive.train",
            "--config",
            str(control / (policy + "-config.json")),
            "--studies",
            str(studies),
            "--series",
            str(series),
            "--cache",
            str(cache_root / "cache"),
            "--folds",
            str(inp / "initiative/experiments/competitive-v1/folds"),
            "--artifact-root",
            str(inp),
            "--output",
            str(output),
            "--device",
            "cuda",
        ]
        code_return = stopper.supervise(command, code, output)
        save(
            control / (policy + "-terminal-" + job + ".json"),
            {
                "job_id": job,
                "policy": policy,
                "child_exit": code_return,
                "stop_requested": stopper.requested,
                "sampling_input_provenance_sha256": sha(output / "sampling-input-provenance.json"),
            },
        )
        return code_return
    finally:
        if cache_root is not None:
            subprocess.run(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(code / "scripts/newton/runtime_archive.py"),
                    "cleanup",
                    "--root",
                    str(cache_root),
                ],
                check=True,
            )
        if environment_root is not None:
            subprocess.run(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(code / "scripts/newton/runtime_environment.py"),
                    "cleanup",
                    "--root",
                    str(environment_root),
                ],
                check=True,
            )
        if code is not None:
            if (
                code.is_symlink()
                or code.stat().st_uid != os.getuid()
                or not code.name.startswith("rsna-sampling-production-" + job + "-")
            ):
                raise ValueError("Unowned source scratch is never removed")
            shutil.rmtree(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("uniform", "stratified"))
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.control, args.policy))


if __name__ == "__main__":
    main()
