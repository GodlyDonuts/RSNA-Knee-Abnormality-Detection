"""Mixed-generation no-write reconciliation and bounded operational recovery."""

import ast
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-cache-capacity-recovery-20261020-v1"
OLD = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v3"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suffix = load("capacity_suffix_tests", CONTROL / "suffix.py")
owner = load("capacity_owner_tests", CONTROL / "owner.py")
recon = load("capacity_original_recon", OLD / "reconcile.py")
fixture = load(
    "capacity_original_fixture", ROOT / "tests/test_predictive_full_cache_continuation.py"
)


def successor(tmp_path):
    result = tmp_path / "successor"
    result.mkdir(mode=0o700)
    for name in ("cache", "scratch", "proofs"):
        (result / name).mkdir(mode=0o700)
    return result


def complete(root, record, path, i):
    intent = {
        "source_ordinal": i,
        "source_bytes": sum(r["bytes"] for r in record["source_files"]),
        "original_source_table_sha256": recon.digest(recon.json_bytes(record["source_files"])),
    }
    (root / "proofs" / f"case-{i:04d}.intent.json").write_text(json.dumps(intent))
    source = {
        "files": [
            {"source_path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]}
            for r in record["source_files"]
        ],
        "source_bytes": intent["source_bytes"],
        "study_uid_sha256": recon.digest(record["uid"].encode()),
        "series_csv_sha256": recon.digest(record["series_csv"].encode()),
    }
    compressed = gzip.compress(json.dumps(source).encode())
    (root / "proofs" / f"case-{i:04d}.source-proof.private.json.gz").write_bytes(compressed)
    row = {
        "uid": record["uid"],
        "path": str(path),
        "proof": {"stat": recon.signature(path)},
        "source_proof_sha256": recon.digest(compressed),
        "original16_reused": False,
    }
    (root / "proofs" / f"case-{i:04d}.verified.private.json").write_text(json.dumps(row))
    return row


def test_later_proof_honestly_admits_old_orphan_without_requiring_missing_buffer(tmp_path):
    original, records, prior, _ = fixture.case.__wrapped__(tmp_path)
    later = successor(tmp_path)
    old_orphan = original / "cache/case-0002.private.npz"
    expected = complete(later, records[2], old_orphan, 2)
    before = {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    value = suffix.raw_inventory(recon, [original, later], records, prior, maximum_buffers=6)
    assert set(value["complete"]) == {0, 1, 2}
    assert value["complete"][2] == expected and value["proof_roots"][2] == later
    assert value["pending"] == [] and value["orphan_caches"] == {} and value["buffers"] == {}
    assert len(value["all_buffers"]) == 1  # Retained old buffer is still physically charged.
    assert before == {
        str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_pending_fitting_resume_buffer_and_crlf_preserved(tmp_path):
    original, records, prior, data = fixture.case.__wrapped__(tmp_path)
    later = successor(tmp_path)
    moved = later / "scratch/fitting-resume-original"
    data.parent.rename(moved)
    (later / "proofs/case-0002.intent.json").write_bytes(
        (original / "proofs/case-0002.intent.json").read_bytes()
    )
    records[2]["series_csv"] = records[2]["series_csv"].replace("\n", "\r\n")
    (moved / "data/train_series.csv").write_bytes(records[2]["series_csv"].encode())
    value = suffix.raw_inventory(recon, [original, later], records, prior, maximum_buffers=6)
    assert value["pending"] == [2] and value["buffers"][2] == moved / "data"
    assert value["buffer_owners"][str(moved / "data")] == later
    assert (moved / "data/train_series.csv").read_bytes().endswith(b"\r\n")


@pytest.mark.parametrize("change", ["row_path", "stat", "source", "metadata"])
def test_retained_original_mismatch_never_becomes_refetch(tmp_path, change):
    original, records, prior, data = fixture.case.__wrapped__(tmp_path)
    later = successor(tmp_path)
    if change == "row_path":
        path = original / "proofs/case-0001.verified.private.json"
        value = json.loads(path.read_text())
        value["path"] = str(tmp_path / "unbound.npz")
        path.write_text(json.dumps(value))
    elif change == "stat":
        (original / "cache/case-0001.private.npz").write_bytes(b"changed")
    elif change == "source":
        (original / "proofs/case-0001.source-proof.private.json.gz").write_bytes(b"changed")
    else:
        (data / "train_series.csv").write_bytes(b"changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        suffix.raw_inventory(recon, [original, later], records, prior)


def test_partial_record_and_orphan_preserved_until_original_pixel_comparison(tmp_path):
    original, records, prior, _ = fixture.case.__wrapped__(tmp_path)
    later = successor(tmp_path)
    (original / "proofs/case-0002.verified.private.json").write_text('{"uid":')
    value = suffix.raw_inventory(recon, [original, later], records, prior)
    assert value["pending"] == [2] and 2 in value["orphan_caches"]
    assert 2 in value["partial_verified_records"]


def test_context_union_keeps_original_renderer_and_later_proof_old_archive(tmp_path):
    roots = [tmp_path / "oldctx", tmp_path / "newctx"]
    for root in roots:
        root.mkdir(mode=0o700)
    raw = {0: {"proof": {"sha256": "raw0"}}, 1: {"proof": {"sha256": "raw1"}}}

    def provenance(row, original=False):
        return {"raw": row["proof"]["sha256"], "renderer": "old" if original else "corrected"}

    rows = []
    for i in range(2):
        p = roots[0] / f"context-{i:04d}.private.npz"
        p.write_bytes(b"complete" + bytes([i]))
        row = {
            "uid": str(i),
            "path": str(p),
            "proof": {"stat": recon.signature(p)},
            "provenance": provenance(raw[i], original=i == 0),
        }
        (roots[i] / f"context-{i:04d}.verified.private.json").write_text(json.dumps(row))
        rows.append(row)
    value = suffix.context_inventory(recon, roots, [], raw, ["0", "1"], provenance)
    assert list(value["complete"].values()) == rows and not value["orphan_caches"]


def test_capacity_exact_reserve_threshold_and_no_deletion_credit():
    other = 62 * 2**30 + 594 * 2**20
    sample = suffix.capacity_snapshot(
        56_893_537_879, 16_043_022_828, 483_983_949_824, other, scratch_cap=6 * 2**30
    )
    assert sample["accepted"] and sample["margin_bytes"] > 14 * 2**30
    at = suffix.capacity_snapshot(
        sample["raw_paid_bytes"],
        sample["context_paid_bytes"],
        sample["required_bytes"],
        other,
        scratch_cap=6 * 2**30,
    )
    assert not at["accepted"] and at["margin_bytes"] == 0
    with pytest.raises(ValueError, match="scratch"):
        suffix.capacity_snapshot(0, 0, 2**50, other, scratch_cap=4 * 2**30)


def test_transient_capacity_wait_logs_numeric_failure_then_continues_once():
    clock, observations = [0], []

    def snapshot():
        return {"accepted": clock[0] >= 60, "free_bytes": clock[0], "required_bytes": 59}

    def sleep(seconds):
        clock[0] += seconds

    result = suffix.wait_capacity(
        snapshot, emit=observations.append, wait_seconds=90, monotonic=lambda: clock[0], sleep=sleep
    )
    assert result["accepted"] and [r["wait_elapsed_seconds"] for r in observations] == [0, 30]


def test_wait_is_finite_and_never_writes_or_retries_source():
    clock, observations = [0], []
    with pytest.raises(RuntimeError, match="preserve every original"):
        suffix.wait_capacity(
            lambda: {"accepted": False},
            emit=observations.append,
            wait_seconds=60,
            monotonic=lambda: clock[0],
            sleep=lambda n: clock.__setitem__(0, clock[0] + n),
        )
    assert len(observations) == 3 and clock[0] == 60


def test_transition_requires_both_terminal_no_duplicate_phase():
    jobs = {
        job: {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}
        for job in ("979664", "979665")
    }
    assert owner.next_action(jobs, {}) == ("dispatch", "raw")
    phase = {"raw": {"job_id": "999001", "returncode": None, "ready": True}}
    jobs["999001"] = {"state": "RUNNING", "queue_empty": False, "exit_code": "0:0"}
    assert owner.next_action(jobs, phase) == ("dispatch", "context")
    phase["context"] = {"job_id": "999002", "returncode": None}
    assert owner.next_action(jobs, phase) == ("wait",)
    jobs["999001"] = {"state": "FAILED", "queue_empty": True, "exit_code": "1:0"}
    with pytest.raises(AssertionError, match="separate review"):
        owner.next_action(jobs, phase)


def test_raw_requests_four_lanes_eight_cpu_no_gpu_and_context_one_cpu():
    class Transport:
        COMMAND = "stdin-only"

    raw = owner.command("raw", Transport)
    context = owner.command("context", Transport)
    assert "--cpus-per-task=8" in raw and "--mem=96G" in raw and "--gres=none" in raw
    assert "--cpus-per-task=1" in context and "--mem=12G" in context
    assert "--nodelist=evc33" in raw and "--time=20:00:00" in raw


def test_case_numerics_only_resource_gate_and_output_location_can_change():
    original = (
        ROOT / "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt"
    ).read_text()
    node = next(
        n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "one"
    )
    old = ast.get_source_segment(original, node)
    # Execute the actual prepared text transformation from the wrapper.
    source = (CONTROL / "program.py.txt").read_text()
    a = source.index('tree = ast.parse(v["original_program"])')
    b = source.index("start = time.monotonic()", a)
    ns = {"ast": ast, "v": {"original_program": original}}
    exec(source[a:b], ns)
    updated = ns["one_text"]
    assert (
        old[old.index('    with (D / "proofs"') :].replace(
            'p = D / "cache" / ("case-" + str(i).zfill(4) + ".private.npz")', "p = output_path(i)"
        )
        == updated[updated.index('    with (D / "proofs"') :]
    )
    compile(updated, "exact_case_numerics", "exec")
    for name in ("program.py.txt", "context.py.txt", "admit.py.txt", "collect.py.txt"):
        compile((CONTROL / name).read_text(), name, "exec")


def receipt_arguments():
    folder = ROOT / "initiative/research/predictive-alias-failure-20261020-v1"
    values = []
    for name in (
        "actual-regression.json",
        "actual-first-recovery-preintent-failure.json",
        "actual-prefix-byte-admission.json",
    ):
        body = (folder / name).read_bytes()
        values.extend((body, owner.sha(body)))
    return values


def test_frozen_payload_reuses_actual_original_sources_and_separately_binds_new_controls(
    monkeypatch,
):
    real = owner.git_file
    monkeypatch.setattr(
        owner,
        "git_file",
        lambda commit, path: (
            (ROOT / path).read_text() if commit == "f" * 40 else real(commit, path)
        ),
    )
    decision = (
        ROOT
        / "initiative/research/parallel-serving-account-20261020-v1/capacity-recovery-decision.accepted.json"
    ).read_bytes()
    bundle = owner.freeze(
        "f" * 40, *receipt_arguments(), capacity_body=decision, capacity_sha256=owner.sha(decision)
    )
    raw, context = bundle["payloads"]["raw"], bundle["payloads"]["context"]
    assert raw["reconcile"] == (OLD / "reconcile.py").read_text()
    assert raw["alias_contract"] == (OLD / "contract.py").read_text()
    assert raw["original_program"] == real(
        owner.base_module().RAW_SOURCE,
        "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt",
    )
    assert raw["source_git"] == context["source_git"] == "f" * 40
    assert raw["lane_count"] == 4 and raw["scratch_cap_bytes"] == 6 * 2**30
    assert raw["context_module_sha256"] == context["module_sha256"]
    assert raw["mixed_prefix_admission"] is None  # Actual readonly admission comes first.
    assert set(bundle["programs"]) == {"raw", "context", "collect", "admit"}
    assert (
        raw["files"]["rsna/competitive/predictive/cases.py"]
        == context["files"]["rsna/competitive/predictive/cases.py"]
    )


def test_finite_owner_fake_full_flow_dispatches_each_once_and_collects_without_remote(
    tmp_path, monkeypatch
):
    import sys

    base = owner.base_module()
    base.ROOT = tmp_path
    (tmp_path / "artifacts").mkdir()
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setattr(owner, "CONTROL", control)
    monkeypatch.setattr(owner, "base_module", lambda: base)
    bundle = {
        "commit": "f" * 40,
        "sha256": {},
        "transport": object(),
        "payloads": {"raw": {}, "context": {}},
    }
    monkeypatch.setattr(owner, "freeze", lambda *a, **kw: bundle)
    calls = []
    phases = {}

    def observe(_socket, ids, _transport, quota=False):
        jobs = {
            job: {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}
            for job in ("979664", "979665", "979573", "979604")
        }
        for phase in phases.values():
            jobs[phase["job_id"]] = {"state": "COMPLETED", "exit_code": "0:0", "queue_empty": True}
        return jobs, "/home/sa305415 100 40000000 0 - 100 6000 0"

    def launch(name, role, payload, _bundle, _journal, _socket, _quota):
        assert name not in phases
        calls.append((name, role, payload.copy()))
        phase = {
            "job_id": str(990000 + len(calls)),
            "returncode": 0,
            "ready": True,
            "accepted": role == "collect",
        }
        phases[name] = phase
        return phase

    monkeypatch.setattr(base, "observe", observe)
    monkeypatch.setattr(base, "launch", launch)
    monkeypatch.setattr(base, "refresh", lambda *args: None)
    monkeypatch.setattr(base, "credential", lambda: "dummy-credential-through-stdin-only")
    monkeypatch.setattr(base.time, "sleep", lambda _seconds: None)
    p = tmp_path / "receipt.json"
    p.write_text("{}")
    argv = ["owner.py", "--source-git", "f" * 40]
    for receipt in ("regression", "prior-attempt", "prefix-admission"):
        argv += ["--" + receipt + "-receipt", str(p), "--" + receipt + "-sha256", "0" * 64]
    for receipt in ("capacity-decision", "mixed-prefix-admission"):
        argv += ["--" + receipt, str(p), "--" + receipt + "-sha256", "0" * 64]
    monkeypatch.setattr(sys, "argv", argv)
    owner.main()
    assert [c[0] for c in calls] == ["raw", "raw_collect", "context", "context_collect"]
    assert calls[1][2]["root"] == owner.ROOTS["raw"]
    assert calls[3][2]["root"] == owner.ROOTS["context"]
    result = json.loads(
        (
            tmp_path / "artifacts/predictive-cache-capacity-recovery-20261020-v1/owner-result.json"
        ).read_text()
    )
    assert result["new_GPU_models_or_competition_entries"] == 0
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        owner.main()  # Existing intent is never resumed as a second owner.
    assert len(calls) == 4
