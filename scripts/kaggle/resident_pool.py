"""Two device workers reuse models while preserving original chunk receipts."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

if __package__:
    from .resident_worker import parse_command
else:
    from resident_worker import parse_command

from rsna.infer import write_submission
from rsna.schema import TARGETS


class ResidentPool:
    def __init__(
        self,
        *,
        workers=2,
        timeout_seconds=32400,
        process_factory=subprocess.Popen,
        performance_fp16_probe=False,
    ):
        if type(performance_fp16_probe) is not bool:
            raise ValueError("Explicit performance-probe flag required")
        self.performance_fp16_probe = performance_fp16_probe
        if workers not in (1, 2) or timeout_seconds <= 0:
            raise ValueError("Bounded one/two-device workers required")
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        self.children = []
        self.requests = 0
        self.deadline = time.monotonic() + timeout_seconds
        self.selector = selectors.DefaultSelector()
        try:
            for index in range(workers):
                p = process_factory(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("resident_worker.py")),
                        "--device-index",
                        str(index),
                    ]
                    + (["--performance-fp16-probe"] if performance_fp16_probe else []),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                self.children.append(p)
                self.selector.register(p.stdout, selectors.EVENT_READ, index)
        except BaseException:
            self.close()
            raise

    def exchange(self, commands):
        self.requests += 1
        pending = {}
        responses = {}
        for index, command in enumerate(commands):
            request = f"{self.requests}:{index}"
            child = self.children[index]
            child.stdin.write(json.dumps({"command": command, "request_id": request}) + "\n")
            child.stdin.flush()
            pending[index] = request
        while pending:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Resident total wall-time budget exhausted")
            for key, _ in self.selector.select(min(remaining, 30)):
                index = key.data
                if index not in pending:
                    continue
                line = key.fileobj.readline()
                if not line:
                    raise RuntimeError("Resident worker exited; no automatic restart")
                result = json.loads(line)
                if result.get("request_id") != pending[index]:
                    raise ValueError("Resident response identity mismatch")
                responses[index] = result
                self.selector.unregister(key.fileobj)
                del pending[index]
        for index in range(len(commands)):
            self.selector.register(self.children[index].stdout, selectors.EVENT_READ, index)
        return [responses[i] for i in range(len(commands))]

    def invoke(self, command, *, check, capture_output=True, text=True):
        args = parse_command(command, allow_fp16_probe=self.performance_fp16_probe)
        template = pd.read_csv(args.template)
        series = pd.read_csv(args.series)
        if not check or not capture_output or not text:
            raise ValueError("Resident call needs exact error capture")
        root = args.output.parent / "resident-shards"
        root.mkdir(mode=0o700)
        commands = []
        outputs = []
        count = min(len(self.children), len(template))
        started = time.monotonic()
        if count < 1:
            raise ValueError("Empty resident study request")
        for index in range(count):
            shard = root / f"worker-{index}"
            shard.mkdir(mode=0o700)
            part = template.iloc[index::count].copy()
            t = shard / "template.csv"
            s = shard / "series.csv"
            out = shard / "submission.csv"
            part.to_csv(t, index=False)
            series.loc[series.StudyInstanceUID.isin(part.StudyInstanceUID)].to_csv(s, index=False)
            selected = list(command)
            for name, path in [("--template", t), ("--series", s), ("--output", out)]:
                selected[selected.index(name) + 1] = str(path)
            commands.append(selected)
            outputs.append(out)
        replies = self.exchange(commands)
        for reply in replies:
            if reply.get("status") != "passed":
                raise subprocess.CalledProcessError(
                    1, command, output="", stderr=reply.get("stderr", "Resident worker failed")
                )
        frames = []
        records = [r["result"] for r in replies]
        common = {
            k: records[0][k]
            for k in (
                "training_release_binding",
                "cache_version",
                "cache_side",
                "encoder",
                "training_precision",
                "inference_precision",
                "device_type",
                "checkpoint_epoch",
                "checkpoint_sha256",
                "diagnostic_only",
                "scheduled_epoch_authorized",
                "initial_pretrained_weights_read",
                "cache_source_revision",
            )
        }
        for index, (out, record) in enumerate(zip(outputs, records)):
            if {k: record[k] for k in common} != common or record["resident_model_load_count"] != 1:
                raise ValueError("Resident shard source/model/runtime differs")
            frame = pd.read_csv(out)
            if (
                list(frame.columns) != ["StudyInstanceUID", *TARGETS]
                or frame.StudyInstanceUID.tolist()
                != template.iloc[index::count].StudyInstanceUID.tolist()
            ):
                raise ValueError("Resident shard identities or column order differ")
            frames.append(frame)
        merged = pd.concat(frames, ignore_index=True)
        if merged.StudyInstanceUID.duplicated().any() or set(merged.StudyInstanceUID) != set(
            template.StudyInstanceUID
        ):
            raise ValueError("Resident union does not match original roster")
        merged = merged.set_index("StudyInstanceUID").loc[template.StudyInstanceUID].reset_index()
        write_submission(merged, template, args.output)
        from rsna.competitive.checkpoint import file_sha256

        metrics = {
            **common,
            "status": "passed",
            "studies": len(template),
            "elapsed_seconds": time.monotonic() - started,
            "submission_sha256": file_sha256(args.output),
            "test_inputs": {
                "series_sha256": file_sha256(args.series),
                "template_sha256": file_sha256(args.template),
                "metadata_binding": "separate_test_inputs_not_checkpoint_bound",
            },
            "resident_device_workers": count,
            "resident_shards": records,
            "hidden_test_runtime_validated": False,
        }
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, sort_keys=True, indent=2) + "\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metrics), stderr="")

    def close(self):
        for p in self.children:
            if p.poll() is None:
                try:
                    p.stdin.write('{"action":"close"}\n')
                    p.stdin.flush()
                    p.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    p.terminate()
                    p.wait(timeout=10)
            if p.stdin:
                p.stdin.close()
            if p.stdout:
                p.stdout.close()
        self.selector.close()
