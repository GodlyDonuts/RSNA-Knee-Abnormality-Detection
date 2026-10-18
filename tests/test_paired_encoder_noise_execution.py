"""Exercise pre-admission failures without a remote machine, token or GPU."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROGRAM = Path(
    "initiative/research/paired-encoder-gradient-noise-20261018-v1/execution/program.py.txt"
)


def execute_before_admission(tmp_path, monkeypatch, *, failed=False, corrupt=False):
    root = tmp_path / "private"
    target = root / "paired-encoder-noise-v1"
    target.mkdir(parents=True)
    raw = b"synthetic source; not a real notebook"
    digest = hashlib.sha256(raw).hexdigest()
    metadata = {"binding": {"source_sha256": digest}}
    (target / "source.private.py").write_bytes(raw + (b"corruption" if corrupt else b""))
    (target / "staged.json").write_text(json.dumps(metadata))
    family = tmp_path / "family"
    family.mkdir()
    retirement = b"synthetic paid-liability proof"
    (family / "completed-fixed3-unspent512MiB-retirement-v1.json").write_bytes(retirement)
    if failed:
        old = root / "paired-logit-real-encoder-v1"
        old.mkdir()
        (old / "actual-failed.json").write_text('{"status":"ERROR"}')
    source = PROGRAM.read_text()
    for old, new in (
        ("/tmp/rsna-licensed-importance-encoder-20261017-v1", str(root)),
        ("/tmp/rsna-criterion-N3-features-20261017-v1", str(tmp_path / "reader")),
        ("/tmp/rsna-family-label-epoch-preservation-20261018-v1", str(family)),
        ("78c4b58bad6bb2ba58bebedc259b19b85b006c32813d69a63e1b1c5212ea6dfb", digest),
        (
            "f210549eb1d2aa76a62140ce8bbc9ae03164f1d68542f191357710ee0a00c54a",
            hashlib.sha256(retirement).hexdigest(),
        ),
    ):
        source = source.replace(old, new)
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename="evc33"))
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-CPU")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10**15))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: "/home/sa305415 0 1000000000 0 0 0 1000000\n",
    )
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"phase": "push", "metadata": metadata}))
    )
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    try:
        exec(compile(source, "new-probe-admission-simulation", "exec"), {})
    finally:
        after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        assert before == after, (
            "Pre-admission wait/failure must never write an intent or alter source"
        )


def test_missing_actual_recovery_waits_without_credentials_or_api(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        execute_before_admission(tmp_path, monkeypatch)
    assert error.value.code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "waiting_original_actual_encoder_recovery_no_GPU_mutation"
    assert out["new_GPU_pushes"] == 0


def test_failed_actual_recovery_stops_before_gpu_admission(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="Original actual encoder recovery failed"):
        execute_before_admission(tmp_path, monkeypatch, failed=True)


def test_changed_notebook_stops_before_any_intent(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        execute_before_admission(tmp_path, monkeypatch, corrupt=True)
