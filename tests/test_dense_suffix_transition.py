"""Explicit checkpoint-continuous allocation transition of the dense-suffix production."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_dense_suffix_production import ENVELOPE, LIABILITIES, Trainer, binding, go, write
from test_dense_suffix_production import env as env  # noqa: F401  (fixture)

from scripts.newton import run_dense_suffix_production as launch
from scripts.newton import transition_dense_suffix_allocation as tr

REPO = Path(__file__).resolve().parents[1]
START = REPO / "initiative/audits/dense-suffix-full-production-start-20261014-v1"
H100 = "NVIDIA H100 PCIe"


def facts(**edit):
    def scheduler(job, node):
        value = {
            "job_state": {"RUNNING"},
            "job_nodes": node,
            "end_time": time.time() + 48 * 3600,
            "node_state": {"MIXED"},
            "gpu_nodes": {"evc40"},
        }
        return dict(value, **edit)

    return scheduler


class Snapshots(Trainer):
    """Canonical trainer contract incl. os.replace checkpoints and bound epoch snapshots."""

    def __call__(self, config, out, extra, logs, seconds):
        (out / "checkpoint.pt").unlink(missing_ok=True)
        code = super().__call__(config, out, extra, logs, seconds)
        if code == 0:
            bound = launch.canonical(binding(self.env, out.name))
            for e in range(1, launch.EPOCHS + 1):
                pt = out / "epoch-checkpoints" / f"epoch-{e:03d}.pt"
                pt.write_bytes(f"{out.name}-{e}".encode())
                meta = pt.with_suffix(".json")
                meta.unlink()
                write(meta, {"epoch": e, "optimizer_steps": launch.UPDATES * e,
                             "checkpoint_sha256": launch.sha(pt), "binding_sha256": bound})  # fmt: skip
        return code


def complete_state(terminal, **edit):
    value = {
        "schema_version": 2,
        "binding_sha256": terminal["binding_sha256"],
        "optimizer_boundary": True,
        "optimizer_state_entries": 12,
        "learning_rates": [1e-4, 2e-5],
        "amp_scaler": True,
        "rng_streams": 1,
        "epoch": 0,
        "next_update": 175,
        "step": 175,
    }
    return dict(value, **edit)


@pytest.fixture
def t(env, tmp_path, monkeypatch):  # noqa: F811
    runtime = tmp_path / "runtime"
    python = runtime / "environment/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!runtime\n")
    python.chmod(0o755)
    inputs = env.cpu["GPU_wrapper_arguments"]
    cache = Path(inputs["cache"])
    (cache / "shard-000").mkdir(parents=True)
    (cache / "shard-000" / "study.npy").write_bytes(b"pixels")
    (cache / "index.json").write_text("{}")
    art = Path(inputs["artifact-root"])
    write(art / "initiative/releases/report-supervision-v1/studies.parquet", {"studies": 1})
    series = "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    write(art / series, {"series": 1})
    base = tmp_path / "shared-base"
    write(base / "model.safetensors", {"weights": 1})
    write(base / "config.json", {"config": 1})
    real = tr.member_roots
    monkeypatch.setattr(
        tr,
        "member_roots",
        lambda *a: dict(
            real(*a),
            shared_base_weights=base / "model.safetensors",
            shared_base_config=base / "config.json",
        ),
    )
    monkeypatch.setattr(tr.sys, "executable", str(python))
    coordinator = env.root / tr.TRANSITIONS / "coordinator.py"
    coordinator.parent.mkdir(parents=True)
    coordinator.write_bytes(Path(tr.__file__).read_bytes())
    original = dict(
        tr.ORIGINAL,
        launcher_sha256=launch.sha(env.root / "launch.py"),
        publisher_sha256=launch.sha(env.root / "publish.py"),
        funding_sha256=launch.sha(env.root / "production-funding.json"),
        runtime_root=str(runtime),
    )
    return SimpleNamespace(
        env=env, root=env.root, original=original, coordinator=coordinator, python=python,
        cache=cache, placed=[], dir=tr.tdir(env.root),
    )  # fmt: skip


def stop_original(t):
    assert go(t.env, Trainer(t.env, {"dense": ["stop"]})) == 75
    return launch.load(t.root / "journal/dense/invocation-01-terminal.json")


def original_state(t):
    return complete_state(launch.load(t.root / "journal/dense/invocation-01-terminal.json"))


def record(t, state=None, **kwargs):
    return tr.record(
        t.root,
        t.env.adm,
        original=t.original,
        pins=t.env.pins,
        placement=lambda node, job: t.placed.append(("record", node, job)),
        inspect=lambda path: state or original_state(t),
        coordinator=t.coordinator,
        **kwargs,
    )


def fund(t, node="evc40", receipt=None, **edit):
    value = launch.load(t.dir / "record.json")
    record_sha = launch.sha(t.dir / "record.json")
    relocation = node != "evc31"
    for name in ("transport-receipt.json", "replacement-funding.json"):
        (t.dir / name).unlink(missing_ok=True)
    receipt_sha = None
    if relocation:
        transport = {
            "executed_by": "root",
            "transition_record_sha256": record_sha,
            "manifest_sha256": value["manifest_sha256"],
            "source_node": "evc31",
            "destination_node": node,
            "members": value["transport_members"],
            "bytes": value["transport_bytes"],
            "original_state_deleted": False,
            "automatic_retry": False,
        }
        receipt_sha = write(t.dir / "transport-receipt.json", dict(transport, **(receipt or {})))
    funding = {
        "funded_by": "root",
        "replacement_funded": True,
        "schema": tr.SCHEMA,
        "transition_record_sha256": record_sha,
        "coordinator_sha256": launch.sha(t.coordinator),
        "original_funding_sha256": t.original["funding_sha256"],
        "original_production_admission_sha256": value["original"]["production_admission_sha256"],
        "CPU_receipt_sha256": value["original"]["CPU_receipt_sha256"],
        "predecessor_job": "899766",
        "node": node,
        "parent_job": "912345",
        "assigned_GPU": "0",
        "CPUs": 2,
        "relocation": relocation,
        "transport_receipt_sha256": receipt_sha,
        "runtime_python": str(t.python),
        "max_slice_seconds": 6 * 3600,
        "max_invocations_per_arm": 6,
        "destination_envelope_bytes": ENVELOPE,
        "existing_liabilities_bytes": LIABILITIES,
        "reserve_bytes": launch.RESERVE,
        "required_free_bytes": LIABILITIES + ENVELOPE + launch.RESERVE,
        "reserve_files": 16,
        "shared_future_bytes": 0,
    }
    write(t.dir / "replacement-funding.json", dict(funding, **edit))


def admit(t):
    return tr.admit(
        t.root,
        original=t.original,
        placement=lambda node, job: t.placed.append(("admit", node, job)),
        coordinator=t.coordinator,
    )


def resume(t, trainer, scheduler=None, placement=None, state=None, slice_seconds=4 * 3600.0):
    return tr.run(
        t.root,
        original=t.original,
        slice_seconds=slice_seconds,
        execute=trainer,
        placement=placement or (lambda funding: H100),
        scheduler=scheduler or facts(),
        inspect=lambda path: state or original_state(t),
        coordinator=t.coordinator,
    )


def ready(t, node="evc40"):
    stop_original(t)
    record(t)
    fund(t, node)
    admit(t)


def tree(path):
    return {str(p): launch.sha(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def intents(t):
    return sorted((t.dir / "journal").rglob("invocation-*-intent.json"))


def test_pins_match_the_actual_original_production_start():
    intent = json.loads((START / "dispatch-intent.json").read_text())
    assert intent["funding_sha256"] == tr.ORIGINAL["funding_sha256"]
    assert intent["launcher_sha256"] == tr.ORIGINAL["launcher_sha256"]
    assert intent["publisher_sha256"] == tr.ORIGINAL["publisher_sha256"]
    assert (str(intent["original_job"]), intent["node"]) == (
        tr.ORIGINAL["parent_job"],
        tr.ORIGINAL["node"],
    )
    assert (
        launch.sha(REPO / "scripts/newton/run_dense_suffix_production.py")
        == (tr.ORIGINAL["launcher_sha256"])
    )
    assert (
        launch.sha(REPO / "scripts/newton/dense_suffix_epoch_publisher.py")
        == (tr.ORIGINAL["publisher_sha256"])
    )
    assert tr.ROOT == launch.ROOT and tr.ADMISSION == launch.ADMISSION
    assert {"MAINT", "MAINTENANCE", "RESERVED"} <= tr.FORBIDDEN_NODE_STATES
    assert tr.MAX_GPU_NODES == 2


def test_relocation_resumes_only_the_exact_exit75_checkpoint_and_finishes(t):
    terminal = stop_original(t)
    original_journal = tree(t.root / "journal")
    checkpoint = launch.sha(t.root / "production/dense/checkpoint.pt")
    summary = record(t)
    assert t.placed == [("record", "evc31", "899766")]
    assert summary["checkpoint_sha256"] == checkpoint == terminal["checkpoint_sha256"]
    assert summary["GPU_intent"] is False and summary["original_state_deleted"] is False
    roots = summary["member_roots"]
    assert roots["public_cache"] == str(t.cache)
    assert summary["transport_members"] < summary["members"]
    fund(t)
    assert admit(t)["relocation"] is True
    trainer = Snapshots(t.env)
    assert resume(t, trainer) == 0
    dense = str(t.root / "production/dense/checkpoint.pt")
    assert trainer.calls == [("dense", ["--resume", dense]), ("suffix_r32", [])]
    assert tree(t.root / "journal") == original_journal
    intent = launch.load(t.dir / "journal/dense/invocation-02-intent.json")
    assert intent["mode"] == "resume" and intent["resume_checkpoint_sha256"] == checkpoint
    assert intent["job_id"] == "912345" and intent["predecessor_job"] == "899766"
    assert launch.sha(t.dir / "exit75-checkpoint-01.pt") == checkpoint
    assert launch.load(t.dir / "journal/suffix_r32/invocation-01-intent.json")["mode"] == "fresh"
    final = launch.load(t.dir / "terminal.json")
    assert final["status"] == tr.TERMINAL_STATUS and final["accuracy_claimed"] is False
    assert not (t.root / "terminal.json").exists()
    lineage = tr.verify(t.root, original=t.original)["epochs"]
    assert set(lineage) == set(launch.ARMS) and list(lineage["dense"]) == list(range(1, 11))
    assert resume(t, trainer) == 0 and len(trainer.calls) == 2


def test_replacement_stop_continues_from_the_transition_journal(t):
    ready(t)
    trainer = Snapshots(t.env, {"dense": ["stop", "complete"]})
    assert resume(t, trainer) == 75
    assert resume(t, trainer) == 0
    assert [c[0] for c in trainer.calls] == ["dense", "dense", "suffix_r32"]
    assert launch.load(t.dir / "journal/dense/invocation-03-intent.json")["mode"] == "resume"


def test_owning_node_replacement_reuses_inputs_but_never_bypasses_maintenance(t):
    ready(t, node="evc31")
    assert not (t.dir / "transport-receipt.json").exists()
    trainer = Snapshots(t.env)
    for state in ({"MAINT", "RESERVED"}, {"MIXED", "RESERVED"}, {"IDLE", "DRAIN"}):
        with pytest.raises(ValueError, match="never bypassed"):
            resume(t, trainer, facts(node_state=state))
    assert trainer.calls == [] and intents(t) == []
    assert resume(t, trainer, facts(gpu_nodes={"evc40", "evc31"})) == 0


def test_absent_original_terminal_is_never_transitioned(t):
    with pytest.raises(ValueError, match="absent terminal"):
        record(t)
    assert not (t.dir / "record.json").exists()


def test_uncertain_original_intent_is_never_transitioned(t):
    stop_original(t)
    (t.root / "journal/dense/invocation-01-terminal.json").unlink()
    with pytest.raises(ValueError, match="uncertain replay"):
        record(t)


@pytest.mark.parametrize("plan", [["fail"], ["complete"]])
def test_nonresumable_original_terminal_is_never_transitioned(t, plan):
    trainer = Trainer(t.env, {"dense": plan, "suffix_r32": ["fail"]})
    with pytest.raises(ValueError, match="Trainer failed"):
        go(t.env, trainer)
    with pytest.raises(ValueError, match="exit-75"):
        record(t)


def test_original_exit75_with_unverifiable_state_is_not_resumable(t):
    stop_original(t)
    path = t.root / "journal/dense/invocation-01-terminal.json"
    value = dict(launch.load(path), resumable=False)
    path.unlink()
    write(path, value)
    with pytest.raises(ValueError, match="exit-75"):
        record(t)


@pytest.mark.parametrize(
    "edit",
    [
        {"step": 0, "epoch": 0, "next_update": 0},
        {"step": 176},
        {"schema_version": 1},
        {"binding_sha256": "0" * 64},
        {"optimizer_boundary": False},
        {"optimizer_state_entries": 0},
        {"learning_rates": [float("nan")]},
        {"learning_rates": []},
        {"amp_scaler": False},
        {"rng_streams": 0},
        {"step": 2810, "epoch": 10, "next_update": 0},
    ],
)
def test_initial_or_incomplete_checkpoint_state_is_refused(t, edit):
    terminal = stop_original(t)
    with pytest.raises(ValueError, match="Initial-state|AdamW/LR/counter"):
        record(t, complete_state(terminal, **edit))
    assert not (t.dir / "record.json").exists()


def test_real_complete_state_summary_reads_the_canonical_checkpoint(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    parameter.sum().backward()
    optimizer.step()
    bound = {"config": {"checkpoint_format": "frozen_base_delta"}}
    torch.save(
        {
            "schema_version": 2,
            "binding": bound,
            "optimizer_boundary": True,
            "optimizer": optimizer.state_dict(),
            "amp_scaler": {"scale": 128.0},
            "epoch": 0,
            "next_update": 175,
            "step": 175,
            "rng_by_rank": [{"torch": torch.get_rng_state()}],
        },
        tmp_path / "checkpoint.pt",
    )
    state = tr.checkpoint_state(tmp_path / "checkpoint.pt")
    assert state["binding_sha256"] == launch.canonical(bound)
    tr.validate_checkpoint_state(
        launch, state, launch.canonical(bound), tr.checkpoint_schema(bound)
    )
    assert state["learning_rates"] == [1e-4] and state["optimizer_state_entries"] == 1


def rewrite(path, edit):
    value = edit(launch.load(path))
    path.unlink()
    write(path, value)


@pytest.mark.parametrize(
    "member, edit",
    [
        ("checkpoint.pt", None),
        ("run.json", lambda v: dict(v, binding=dict(v["binding"], world_size=2))),
        ("invocation.json", lambda v: dict(v, checkpoint_sha256="0" * 64)),
        ("invocation.json", lambda v: dict(v, status="completed")),
    ],
)
def test_changed_predecessor_bytes_before_record_are_refused(t, member, edit):
    stop_original(t)
    path = t.root / "production/dense" / member
    if edit is None:
        path.write_bytes(path.read_bytes() + b" ")
    else:
        rewrite(path, edit)
    with pytest.raises(ValueError, match="exact resume refused"):
        record(t)
    assert not (t.dir / "record.json").exists()


@pytest.mark.parametrize(
    "member",
    [
        "production/dense/checkpoint.pt",
        "production/dense/invocation.json",
        "journal/dense/invocation-01-terminal.json",
        "journal/dense/invocation-01-intent.json",
        "production-funding.json",
        "production-admission.json",
        "dense-config.json",
    ],
)
def test_changed_predecessor_bytes_after_record_refuse_resume(t, member):
    ready(t)
    path = t.root / member
    path.write_bytes(path.read_bytes() + b" ")
    trainer = Snapshots(t.env)
    with pytest.raises(ValueError):
        resume(t, trainer)
    assert trainer.calls == [] and intents(t) == []


def test_changed_or_extra_destination_members_refuse_admission(t):
    stop_original(t)
    record(t)
    fund(t)
    (t.cache / "shard-000" / "study.npy").write_bytes(b"pixelz")
    with pytest.raises(ValueError, match="differs from original bytes"):
        admit(t)
    (t.cache / "shard-000" / "study.npy").write_bytes(b"pixels")
    (t.cache / "shard-000" / "extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="members differ"):
        admit(t)
    (t.cache / "shard-000" / "extra.npy").unlink()
    (t.python).write_text("#!other runtime\n")
    with pytest.raises(ValueError, match="differs from original bytes"):
        admit(t)
    assert not (t.dir / "destination-admission.json").exists()


@pytest.mark.parametrize(
    "receipt",
    [
        {"manifest_sha256": "0" * 64},
        {"bytes": 1},
        {"members": 1},
        {"destination_node": "evc34"},
        {"original_state_deleted": True},
        {"executed_by": "devin"},
        {"transition_record_sha256": "0" * 64},
    ],
)
def test_relocation_requires_exact_explicit_transport(t, receipt):
    stop_original(t)
    record(t)
    fund(t, receipt=receipt)
    with pytest.raises(ValueError, match="Explicit exact transport"):
        admit(t)


def test_relocation_without_transport_or_with_silent_rebinding_is_refused(t):
    stop_original(t)
    record(t)
    fund(t, transport_receipt_sha256=None)
    with pytest.raises(ValueError, match="rebound"):
        admit(t)
    fund(t, runtime_python="/usr/bin/python3")
    with pytest.raises(ValueError, match="rebound"):
        admit(t)
    fund(t, node="evc31", relocation=True)
    with pytest.raises(ValueError, match="rebound"):
        admit(t)


@pytest.mark.parametrize(
    "edit",
    [
        {"transition_record_sha256": "0" * 64},
        {"coordinator_sha256": "0" * 64},
        {"original_funding_sha256": "0" * 64},
        {"original_production_admission_sha256": "0" * 64},
        {"CPU_receipt_sha256": "0" * 64},
        {"predecessor_job": "899765"},
        {"parent_job": "899766"},
        {"funded_by": "devin"},
        {"replacement_funded": False},
        {"reserve_bytes": 0},
        {"required_free_bytes": 1},
        {"shared_future_bytes": 1},
        {"reserve_files": 0},
        {"max_invocations_per_arm": 1},
        {"max_slice_seconds": 60},
    ],
)
def test_stale_or_incomplete_replacement_funding_is_refused(t, edit):
    stop_original(t)
    record(t)
    fund(t, **edit)
    with pytest.raises(ValueError):
        admit(t)
    assert t.placed == [("record", "evc31", "899766")]


def test_funding_changed_after_admission_refuses_resume(t):
    ready(t)
    fund(t, CPUs=1)
    trainer = Snapshots(t.env)
    with pytest.raises(ValueError, match="Exact destination admission"):
        resume(t, trainer)
    assert trainer.calls == []


@pytest.mark.parametrize(
    "edit",
    [
        {"job_state": {"PENDING"}},
        {"job_state": {"COMPLETING"}},
        {"job_nodes": "evc34"},
        {"end_time": time.time() + 600},
        {"node_state": {"MIXED", "MAINT"}},
        {"gpu_nodes": {"evc31", "evc34"}},
    ],
)
def test_stale_scheduler_placement_refuses_any_GPU_intent(t, edit):
    ready(t)
    trainer = Snapshots(t.env)
    with pytest.raises(ValueError):
        resume(t, trainer, facts(**edit))
    assert trainer.calls == [] and intents(t) == []


def test_unassigned_device_or_reserve_refuses_any_GPU_intent(t, monkeypatch):
    ready(t)
    trainer = Snapshots(t.env)

    def wrong(funding):
        raise ValueError("Funded replacement step with exactly one assigned H100 required")

    with pytest.raises(ValueError, match="assigned H100"):
        resume(t, trainer, placement=wrong)
    monkeypatch.setattr(
        launch.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**9, total=0, used=0)
    )
    with pytest.raises(ValueError, match="reserve"):
        resume(t, trainer)
    assert trainer.calls == [] and intents(t) == []


def test_duplicate_transition_record_and_admission_are_refused(t):
    ready(t)
    with pytest.raises(ValueError, match="duplicate transition"):
        record(t)
    with pytest.raises(ValueError, match="never repeated"):
        admit(t)
    (t.root / tr.TRANSITIONS / "002").mkdir()
    with pytest.raises(ValueError, match="further one"):
        record(t)


def test_failed_or_uncertain_replacement_invocation_is_never_retried(t):
    ready(t)
    trainer = Snapshots(t.env, {"dense": ["fail"]})
    with pytest.raises(ValueError, match="Trainer failed"):
        resume(t, trainer)
    with pytest.raises(ValueError, match="no automatic retry"):
        resume(t, trainer)
    (t.dir / "journal/dense/invocation-02-terminal.json").unlink()
    with pytest.raises(ValueError, match="uncertain replay"):
        resume(t, trainer)
    assert len(trainer.calls) == 1


def test_resume_slice_must_be_bounded_by_replacement_funding(t):
    ready(t)
    trainer = Snapshots(t.env)
    for seconds in (float("inf"), 600.0, 7 * 3600.0):
        with pytest.raises(ValueError, match="slice"):
            resume(t, trainer, slice_seconds=seconds)
    assert trainer.calls == []


def test_source_bound_epoch_lineage_is_verified(t):
    ready(t)
    assert resume(t, Snapshots(t.env)) == 0
    pt = t.root / "production/dense/epoch-checkpoints/epoch-003.pt"
    pt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source/run"):
        tr.verify(t.root, original=t.original)


def test_publication_uses_the_original_publisher_with_replacement_placement(t, monkeypatch):
    ready(t)
    calls = []
    real = tr._load

    def load(path, name, expected):
        module = real(path, name, expected)
        if name == "original_dense_suffix_publisher":
            module.publish = lambda *a, placement, **k: calls.append((a, placement))
        return module

    monkeypatch.setattr(tr, "_load", load)
    with pytest.raises(ValueError, match="immutable source-bound"):
        tr.publish("dense", 1, t.root, t.env.adm, original=t.original, coordinator=t.coordinator)
    assert resume(t, Snapshots(t.env)) == 0
    tr.publish("dense", 1, t.root, t.env.adm, original=t.original, coordinator=t.coordinator)
    ((args, placement),) = calls
    assert args == ("dense", 1, t.root, t.env.adm)
    with pytest.raises(ValueError, match="Replacement-node"):
        placement("evc31")


def test_scheduler_facts_parse_and_refuse_ambiguous_allocations(monkeypatch):
    job = {"jobs": [{"job_state": ["RUNNING"], "nodes": "evc40", "end_time": {"number": 5}}]}
    node = {"nodes": [{"state": ["MIXED", "RESERVED"]}]}
    rows = "912345|gres/gpu:h100:1|evc40\n908454|gres/gpu:h100:1|evc40\n1|N/A|evc12\n"

    def output(command, text):
        return {"job": json.dumps(job), "node": json.dumps(node)}.get(command[2], rows)

    monkeypatch.setattr(tr.subprocess, "check_output", output)
    value = tr.scheduler_facts("912345", "evc40")
    assert value["node_state"] == {"MIXED", "RESERVED"} and value["gpu_nodes"] == {"evc40"}
    assert value["end_time"] == 5.0
    rows = "1|gres/gpu:2|evc[31,40]\n"
    with pytest.raises(ValueError, match="multi-node"):
        tr.scheduler_facts("912345", "evc40")
