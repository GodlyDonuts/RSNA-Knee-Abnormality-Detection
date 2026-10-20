"""Optional finite placement owner. Requires explicit actual idle-owner handoffs.

Prepared only: no defaults authorize a launch, transfer, GPU or account mutation.
This is a distinct owner, never a restart or hotpatch of owner68792.
"""

import argparse
import importlib.util
import json
import os
import re
import select
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import placement as p

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
VERIFIED = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"


def git_file(commit, name):
    p.require(re.fullmatch("[0-9a-f]{40}", commit), "Exact committed source required")
    return subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)


def module(source, name, file):
    spec = importlib.util.spec_from_loader(name, loader=None)
    result = importlib.util.module_from_spec(spec)
    result.__file__ = str(file)
    exec(compile(source, str(file), "exec"), result.__dict__)
    return result


def freeze(pin, decision_path, handoff_path, deployment_stop_path):
    files = {}
    for name in ("owner.py", "placement.py", "remote.py"):
        path = HERE / name
        raw = git_file(pin, path.relative_to(ROOT).as_posix())
        p.require(raw == path.read_bytes(), "Invoke the exact frozen placement controls")
        files[name] = raw.decode()
    decision = json.loads(decision_path.read_text())
    p.require(
        decision["status"] == "accepted_optional_C0_evc27_placement"
        and decision["placement_source_git"] == pin
        and decision["no_paid_credits"] is True
        and decision["shared_C0_log_envelope_reassigned_not_added"] is True,
        "Fresh explicit placement/funding acceptance required",
    )
    handoff_raw = handoff_path.read_bytes()
    stopped = json.loads(handoff_raw)
    p.handoff(stopped)
    deployment = json.loads(deployment_stop_path.read_text())
    p.require(
        deployment["status"] == "actual_idle_predictive_deployment_owner_stopped_before_phase"
        and deployment["pid"] == 77985
        and deployment["source_git"] == "8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f"
        and deployment["phase_intents"] == []
        and deployment["children"] == []
        and deployment["owner_process_absent_after_stop"] is True,
        "Original waiting deployment owner needs an exclusive pre-phase handoff",
    )
    for pid in (68792, deployment["pid"]):
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        ).stdout
        p.require(
            "predictive-full-fitting" not in command
            and "predictive-final-deployment-owner" not in command,
            "Prior finite owner is still active",
        )
    journal = ROOT / "artifacts/predictive-full-fitting-20261020-v2/finite-owner"
    p.require(
        {q.name for q in journal.iterdir()}
        <= {"launcher.json", "owner-intent.json", "stdout", "stderr"},
        "Original owner advanced beyond idle waiting; reconcile before any placement",
    )
    old = module(
        git_file(p.SCIENCE, p.BASE + "/owner.py"),
        "frozen_original_owner",
        ROOT / p.BASE / "owner.py",
    )
    old.SCIENCE = p.SCIENCE
    prepared = json.loads((ROOT / p.BASE / "actual-cpu-prepared.json").read_text())
    p.require(
        prepared["source_git"] == p.SCIENCE and prepared["CPU_job"] == "979667",
        "Exact actual CPU preparation required",
    )
    original_remote = git_file(p.SCIENCE, p.BASE + "/owner_remote.py.txt").decode()
    transport_name = "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    transport = module(
        git_file(p.SCIENCE, transport_name), "frozen_transport", ROOT / transport_name
    )
    program = (
        "import sys,json,types\n"
        "m=types.ModuleType('placement');sys.modules['placement']=m\n"
        f"exec(compile({files['placement.py']!r},'placement.py','exec'),m.__dict__)\n"
        "n={'__name__':'frozen_portable_operation'}\n"
        f"exec(compile({files['remote.py']!r},'remote.py','exec'),n)\n"
        "try:\n print(json.dumps(n['main'](json.load(sys.stdin)),allow_nan=False),flush=True)\n"
        "except BaseException as e:\n"
        " t=e.__traceback__\n"
        " while t.tb_next:t=t.tb_next\n"
        " print(json.dumps({'error_type':type(e).__name__,'site_line':t.tb_lineno}),file=sys.stderr)\n"
        " raise SystemExit(1)\n"
    )
    payload = {
        "prepared": prepared,
        "original_remote": original_remote,
        "handoff": stopped,
        "handoff_text": handoff_raw.decode(),
        "handoff_sha256": p.sha(handoff_raw),
        "placement_source_git": pin,
        "owner_git": pin,
        "placement_code_sha256": {n: p.sha(s.encode()) for n, s in files.items()},
        "shared_bytes": decision["shared_future_bytes"],
        "shared_files": decision["shared_future_files"],
        "verified_status": VERIFIED,
    }
    # Validate liability inputs before writing an owner intent.
    p.derived_remote(original_remote, "evc27", payload["shared_bytes"], payload["shared_files"])
    return old, transport, program, payload


class Owner:
    def __init__(self, args, original, transport, program, payload, journal):
        self.args, self.original, self.transport = args, original, transport
        self.program, self.payload, self.journal = program, payload, journal
        self.deadline = datetime.fromisoformat(args.deadline_utc).timestamp()

    def command(self, node, mode):
        if mode == "progress":
            remote = [MANAGEMENT, "-B", "-c", self.transport.COMMAND]
        else:
            duration = "02:30:00" if mode in ("receive", "send") else "00:45:00"
            remote = [
                "srun",
                "--account=skattel",
                "--partition=normal",
                "--qos=normal",
                "--nodelist=" + node,
                "--ntasks=1",
                "--cpus-per-task=1",
                "--gres=none",
                "--mem=8G",
                "--time=" + duration,
                "--job-name=rsna-C0-placement-" + mode,
                "env",
                "CUDA_VISIBLE_DEVICES=",
                "PYTHONNOUSERSITE=1",
                "PYTHONDONTWRITEBYTECODE=1",
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                p.PY27 if node == "evc27" else p.PY33,
                "-B",
                "-c",
                self.transport.COMMAND,
            ]
        return [
            "ssh",
            "-S",
            self.args.socket,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "newton",
            shlex.join(remote),
        ]

    def invoke(self, node, mode, phase=None, extra=None):
        # Only observations may be repeated, for at most30minutes. Every mutation
        # remains single-attempt even if SSH disappears after the intent is saved.
        if mode != "progress":
            return self._invoke_once(node, mode, phase, extra)
        p.require(phase is None, "Progress observation cannot own a mutation intent")
        started = time.monotonic()
        while True:
            try:
                return self._invoke_once(node, mode, None, extra)
            except (ConnectionError, subprocess.TimeoutExpired, OSError):
                p.require(
                    time.monotonic() - started < 1800, "Read-only observation absent30minutes"
                )
                p.require(time.time() < self.deadline, "Finite observation deadline")
                time.sleep(min(30, max(0, self.deadline - time.time())))

    def _invoke_once(self, node, mode, phase=None, extra=None):
        value = {**self.payload, **(extra or {}), "mode": mode, "node": node}
        if mode == "publish":
            value["token"] = self.original.token()
        if phase:
            p.once(
                self.journal / (phase + "-intent.json"),
                {
                    "mode": mode,
                    "node": node,
                    "placement_source_git": self.args.source_git,
                    "program_sha256": p.sha(self.program.encode()),
                    "automatic_retry": False,
                },
            )
        try:
            result = subprocess.run(
                self.command(node, mode),
                input=self.transport.envelope(self.program, value),
                capture_output=True,
                text=True,
                timeout=min(
                    90 if mode == "progress" else (9000 if mode == "send" else 3600),
                    max(1, self.deadline - time.time()),
                ),
            )
        finally:
            value.pop("token", None)
        if phase:
            p.once(
                self.journal / (phase + "-transport.json"),
                {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
            )
        if mode == "progress" and result.returncode == 255:
            raise ConnectionError("Read-only SSH observation lost")
        p.require(result.returncode == 0, "Original operation failed/uncertain; no automatic retry")
        actual = json.loads(result.stdout)
        if phase:
            p.once(self.journal / (phase + "-result.json"), actual)
            p.once(HERE / ("actual-owner-" + phase + ".json"), actual)
        return actual

    def transfer(self, returning=False):
        receive = "return_receive" if returning else "receive"
        send = "return_send" if returning else "send"
        destination, source = ("evc33", "evc27") if returning else ("evc27", "evc33")
        phase = "return-C0" if returning else "contexts-C0"
        p.once(
            self.journal / (phase + "-intent.json"),
            {
                "source": source,
                "destination": destination,
                "automatic_retry": False,
                "placement_source_git": self.args.source_git,
            },
        )
        value = {**self.payload, "mode": receive, "node": destination}
        error_path = self.journal / (phase + "-receiver-stderr")
        # Endpoint secret is consumed in RAM and never copied into a receipt/log.
        with error_path.open("xb") as error:
            process = subprocess.Popen(
                self.command(destination, receive),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=error,
                text=True,
            )
            process.stdin.write(self.transport.envelope(self.program, value))
            process.stdin.close()
            process.stdin = None
            while time.time() < self.deadline:
                ready, _, _ = select.select([process.stdout], [], [], 30)
                if ready:
                    break
            else:
                raise RuntimeError("Receiver observation deadline; preserve original allocation")
            endpoint = json.loads(process.stdout.readline())
            p.require(
                endpoint["host"] == destination and len(endpoint["token"]) == 64,
                "Original private endpoint required",
            )
            actual_sender = self.invoke(source, send, phase + "-sender", {"endpoint": endpoint})
            stdout, _ = process.communicate(timeout=max(1, min(9000, self.deadline - time.time())))
        p.require(process.returncode == 0, "Receiver uncertain; never repeat transfer")
        actual = json.loads(stdout)
        p.require(actual == actual_sender, "Independent sender/receiver terminal differs")
        p.once(self.journal / (phase + "-result.json"), actual)
        p.once(HERE / ("actual-owner-" + phase + ".json"), actual)
        return actual

    def run(self):
        while time.time() < self.deadline:
            receipts = self.original.cache_receipts()
            if receipts is not None:
                self.payload.update(
                    raw_collected=receipts["raw"], context_collected=receipts["context"]
                )
                break
            time.sleep(self.args.poll_seconds)
        else:
            raise RuntimeError("No complete cache before finite deadline; no GPU dispatched")
        admitted = self.invoke("evc33", "admit", "admission-BC")
        p.require(admitted["arms"] == ["B", "C"], "B/C-only original admission required")
        dispatched33 = self.invoke("evc33", "dispatch", "dispatch-BC", {"admitted": admitted})
        p.require(set(dispatched33["jobs"]) == {"B", "C"}, "Original C0 must not be submitted")
        self.payload["all_dispatched_jobs"] = dict(dispatched33["jobs"])
        self.payload["portable"] = self.transfer()
        dispatched27 = self.invoke("evc27", "dispatch", "dispatch-C0")
        p.require(set(dispatched27["jobs"]) == {"C0"}, "Only portable C0")
        self.payload["all_dispatched_jobs"].update(dispatched27["jobs"])
        dispatched = {"evc33": dispatched33, "evc27": dispatched27}
        published, complete, failed = set(), {}, {}
        while time.time() < self.deadline:
            arms = {}
            for node in ("evc33", "evc27"):
                progress = self.invoke(node, "progress", extra={"dispatched": dispatched[node]})
                arms.update(progress["arms"])
            for arm, item in arms.items():
                if (
                    self.original.completed(item["slurm"])
                    and item["terminal"]
                    and arm not in complete
                ):
                    complete[arm] = item
                    p.once(
                        HERE / ("actual-fixed-final-" + arm + ".json"),
                        {
                            **item["terminal"],
                            "actual_Slurm_terminal": item["slurm"],
                            "stderr_bytes": item["stderr_bytes"],
                            "placement_source_git": self.args.source_git,
                        },
                    )
                elif (
                    self.original.terminal(item["slurm"])
                    and not self.original.completed(item["slurm"])
                    and arm not in failed
                ):
                    failed[arm] = item
                    p.once(HERE / ("actual-failure-" + arm + ".json"), item)
            candidate = self.original.ready_epoch({"arms": arms}, published)
            if candidate:
                arm, epoch = candidate
                node = "evc27" if arm == "C0" else "evc33"
                actual = self.invoke(
                    node,
                    "publish",
                    f"publish-{arm}-{epoch:03d}",
                    {"arm": arm, "epoch": epoch, "dispatched": dispatched[node]},
                )
                p.require(
                    actual["status"] == VERIFIED
                    and actual["objects"] == 5
                    and (actual["arm"], actual["epoch"], actual["source_git"])
                    == (arm, epoch, p.SCIENCE),
                    "Exact original epoch preservation required",
                )
                published.add((arm, epoch))
                continue
            if len(complete) + len(failed) == 3:
                returned = None
                if "C0" in complete:
                    p.require(
                        all(("C0", e) in published for e in range(1, 7)),
                        "All six C0 publication proofs required before return",
                    )
                    returned = self.transfer(returning=True)
                result = {
                    "status": "actual_portable_predictive_owner_finished",
                    "science_git": p.SCIENCE,
                    "placement_source_git": self.args.source_git,
                    "jobs": {**dispatched33["jobs"], **dispatched27["jobs"]},
                    "complete": sorted(complete),
                    "failed": sorted(failed),
                    "privately_verified_epochs": sorted(published),
                    "C0_return": returned,
                    "automatic_restart": False,
                    "clinical_accuracy_or_score": False,
                }
                p.once(self.journal / "owner-result.json", result)
                p.once(HERE / "actual-owner-finished.json", result)
                return
            time.sleep(self.args.poll_seconds)
        raise RuntimeError("Finite owner deadline; preserve every original job and receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--deployment-handoff", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--deadline-utc", default="2026-10-22T09:00:00+00:00")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    os.umask(0o077)
    p.require(30 <= args.poll_seconds <= 300, "Bounded polling")
    deadline = datetime.fromisoformat(args.deadline_utc)
    p.require(
        datetime.now(timezone.utc) < deadline <= datetime(2026, 10, 22, 12, tzinfo=timezone.utc),
        "Finite owner deadline",
    )
    original, transport, program, payload = freeze(
        args.source_git, args.decision, args.handoff, args.deployment_handoff
    )
    journal = ROOT / "artifacts" / HERE.name / "finite-owner"
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    p.once(
        journal / "owner-intent.json",
        {
            "pid": os.getpid(),
            "placement_source_git": args.source_git,
            "science_git": p.SCIENCE,
            "handoff_sha256": payload["handoff_sha256"],
            "automatic_retry": False,
            "deployment_handoff_sha256": p.file_sha(args.deployment_handoff),
            "decision_sha256": p.file_sha(args.decision),
            "deadline": args.deadline_utc,
        },
    )
    try:
        Owner(args, original, transport, program, payload, journal).run()
    except BaseException as error:
        p.once(
            journal / "owner-ended.json",
            {
                "error_type": type(error).__name__,
                "status": "requires_read_only_reconciliation",
                "remote_jobs_cancelled": [],
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
