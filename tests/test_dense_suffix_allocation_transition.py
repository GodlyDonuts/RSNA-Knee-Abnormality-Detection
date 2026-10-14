"""Checkpoint-continuous allocation transition for dense-suffix production: fail closed."""

import json
import shutil
import time
from pathlib import Path

import pytest
import test_dense_suffix_production as production
from test_dense_suffix_production import Trainer, go, update, write

from scripts.newton import dense_suffix_allocation_transition as tr
from scripts.newton import run_dense_suffix_production as launch

REPO = Path(__file__).resolve().parents[1]
env = production.env
REPLACEMENT = "912345"
STEPS = 175


class Scheduler:
    def __init__(self, node="evc31"):
        self.jobs = {
            "899766": {"job_id": 899766, "job_state": ["TIMEOUT"]},
            REPLACEMENT: self.running(REPLACEMENT, node),
        }
        self.states = {"evc31": ["MIXED"], "evc34": ["IDLE"], "evc40": ["MIXED"]}
        self.gpu_nodes = {"evc40", node}

    @staticmethod
    def running(job, node):
        return {
            "job_id": int(job),
            "job_state": ["RUNNING"],
            "nodes": node,
            "reservation": "",
            "tres_alloc_str": "cpu=2,mem=32G,node=1,billing=2,gres/gpu=1",
            "end_time": {"number": time.time() + 48 * 3600},
        }

    def job(self, job_id):
        return self.jobs.get(str(job_id))

    def node(self, name):
        return {"name": name, "state": self.states[name]}

    def running_gpu_nodes(self):
        return set(self.gpu_nodes)


def stopping(trainer, steps=STEPS):
    """Trainer stand-in whose exit-75 record carries the real trainer's update history."""

    def execute(config, out, extra, logs, seconds):
        code = trainer(config, out, extra, logs, seconds)
        if code == 75:
            path = out / "invocation.json"
            value = json.loads(path.read_text())
            path.unlink()
            write(path, dict(value, updates=[update(s, 1.0 / s) for s in range(1, steps + 1)]))
        return code

    return execute


def original(env):
    return dict(
        tr.ORIGINAL,
        funding_sha256=launch.sha(env.root / "production-funding.json"),
        launcher_sha256=env.funding["launcher_sha256"],
        publisher_sha256=env.funding["publisher_sha256"],
    )


def funding(env, k=1, **edit):
    base = env.root / tr.TRANSITIONS
    base.mkdir(exist_ok=True)
    if not (base / tr.COORDINATOR).exists():
        shutil.copy(REPO / "scripts/newton/dense_suffix_allocation_transition.py",
                    base / tr.COORDINATOR)  # fmt: skip
    value = {
        "funded_by": "root",
        "transition_version": tr.VERSION,
        "transition": k,
        "prior_funding_sha256": launch.sha(env.root / "production-funding.json"),
        "prior_admission_sha256": launch.sha(env.root / "production-admission.json"),
        "prior_transition_sha256": None
        if k == 1
        else launch.sha(tr.tdir(env.root, k - 1) / tr.RECORD),
        "supersedes_unused_transition": None,
        "coordinator_sha256": launch.sha(base / tr.COORDINATOR),
        "placement": "owning_node",
        "parent_job": REPLACEMENT,
        "node": "evc31",
        "assigned_GPU": "0",
        "CPUs": 2,
        "max_slice_seconds": 4 * 3600,
        "node_local_envelope_bytes": 200 * 2**30,
        "reserve_bytes": launch.RESERVE,
        "required_free_bytes": 100 * 2**30,
        "required_free_files": 1,
        "shared_future_bytes": 0,
        "additional_GPU_funded_by_issue": False,
    }
    value.update(edit)
    path = tr.tdir(env.root, k) / tr.FUNDING
    path.unlink(missing_ok=True)
    write(path, value)
    return value


@pytest.fixture
def stopped(env):
    """Actual original-parent state: dense exit 75 after 175 finite updates."""
    assert go(env, stopping(Trainer(env, {"dense": ["stop"]}))) == 75
    env.original = original(env)
    env.sched = Scheduler()
    return env


def admit(env, host="evc31"):
    return tr.admit(
        env.root,
        env.adm,
        pins=env.pins,
        original=env.original,
        hostname=lambda: host,
        scheduler=env.sched,
    )


def fit(env, trainer, slice_seconds=4 * 3600.0):
    return tr.run(
        env.root,
        env.adm,
        execute=trainer,
        device=lambda funding: "NVIDIA H100 80GB HBM3",
        scheduler=env.sched,
        pins=env.pins,
        original=env.original,
        slice_seconds=slice_seconds,
    )


def snapshot(env):
    names = ["production-funding.json", "production-admission.json", "launch.py", "publish.py"]
    files = {n: launch.sha(env.root / n) for n in names}
    files.update(tr.journal_hashes(launch, env.root / "journal"))
    return files


def test_exit75_transition_resumes_exact_checkpoint_then_successor(stopped):
    env = stopped
    before = snapshot(env)
    checkpoint = launch.sha(env.root / "production" / "dense" / "checkpoint.pt")
    funding(env)
    record = admit(env)
    assert record["arm"] == "dense" and record["complete_state"]["optimizer_steps"] == STEPS
    assert record["complete_state"]["checkpoint_sha256"] == checkpoint
    assert record["predecessor"]["parent_job"] == "899766"
    assert record["original"]["funding_sha256"] == env.original["funding_sha256"]
    trainer = Trainer(env)
    assert fit(env, trainer) == 0
    out = env.root / "production" / "dense"
    assert trainer.calls == [
        ("dense", ["--resume", str(out / "checkpoint.pt")]),
        ("suffix_r32", []),
    ]
    journal = tr.tdir(env.root, 1) / "journal"
    intent = json.loads((journal / "dense" / "invocation-01-intent.json").read_text())
    assert intent["mode"] == "resume" and intent["resume_checkpoint_sha256"] == checkpoint
    assert intent["job_id"] == REPLACEMENT and intent["transition"] == 1
    r32 = json.loads((journal / "suffix_r32" / "invocation-01-intent.json").read_text())
    assert r32["mode"] == "fresh" and r32["resume_checkpoint_sha256"] is None
    assert snapshot(env) == before
    assert not (env.root / "__pycache__").exists()
    assert not (env.root / "terminal.json").exists()
    terminal = json.loads((tr.tdir(env.root, 1) / "terminal.json").read_text())
    assert terminal["accuracy_claimed"] is False and terminal["transition"] == 1
    assert fit(env, trainer) == 0 and len(trainer.calls) == 2
    with pytest.raises(ValueError):
        go(env, Trainer(env))


def _terminal(env):
    path = env.root / "journal" / "dense" / "invocation-01-terminal.json"
    path.write_text(path.read_text() + " ")


def _intent(env):
    path = env.root / "journal" / "dense" / "invocation-01-intent.json"
    path.write_text(path.read_text() + " ")


def _log(env):
    path = env.root / "journal" / "dense" / "invocation-01.stdout"
    path.write_text("edited\n")


def _extra_original_invocation(env):
    write(env.root / "journal" / "dense" / "invocation-02-intent.json", {"arm": "dense"})


def _checkpoint(env):
    (env.root / "production" / "dense" / "checkpoint.pt").write_bytes(b"other")


def _run_record(env):
    path = env.root / "production" / "dense" / "run.json"
    path.write_text(path.read_text() + " ")


def _source(env):
    (env.code / "rsna/x.py").write_text("x = 2\n")


def _launcher(env):
    (env.root / "launch.py").write_text((env.root / "launch.py").read_text() + "\n")


def _admission(env):
    path = env.root / "production-admission.json"
    value = json.loads(path.read_text())
    path.unlink()
    write(path, dict(value, epochs=9))


def _config(env):
    path = env.root / "dense-config.json"
    value = json.loads(path.read_text())
    path.unlink()
    write(path, dict(value, epochs=9))


def _roster(env):
    env.fold.write_text("study_uid,fold\na,1\n")


PREDECESSOR = [
    _terminal,
    _intent,
    _log,
    _extra_original_invocation,
    _checkpoint,
    _run_record,
    _source,
    _launcher,
    _admission,
    _config,
    _roster,
]


@pytest.mark.parametrize("edit", [e for e in PREDECESSOR if e not in (_log, _run_record)])
def test_changed_predecessor_bytes_refused_before_admission(stopped, edit):
    funding(stopped)
    edit(stopped)
    with pytest.raises(ValueError):
        admit(stopped)
    assert not (tr.tdir(stopped.root, 1) / tr.RECORD).exists()


@pytest.mark.parametrize("edit", PREDECESSOR)
def test_changed_predecessor_bytes_refused_before_resumed_intent(stopped, edit):
    funding(stopped)
    admit(stopped)
    edit(stopped)
    trainer = Trainer(stopped)
    with pytest.raises(ValueError):
        fit(stopped, trainer)
    assert trainer.calls == []
    assert not (tr.tdir(stopped.root, 1) / "journal").exists()


def test_absent_terminal_is_never_admitted(stopped):
    (stopped.root / "journal" / "dense" / "invocation-01-terminal.json").unlink()
    funding(stopped)
    with pytest.raises(ValueError, match="uncertain replay"):
        admit(stopped)


def _rewrite_terminal(env, **edit):
    path = env.root / "journal" / "dense" / "invocation-01-terminal.json"
    value = dict(json.loads(path.read_text()), **edit)
    path.unlink()
    launch.save(path, value)


@pytest.mark.parametrize(
    "edit",
    [
        {"resumable": False},
        {"resumable": None},
        {"returncode": None},
        {"returncode": 143},
        {"checkpoint_sha256": "0" * 64},
        {"binding_sha256": "0" * 64},
    ],
)
def test_uncertain_terminal_is_never_admitted(stopped, edit):
    _rewrite_terminal(stopped, **edit)
    funding(stopped)
    with pytest.raises(ValueError):
        admit(stopped)
    assert not (tr.tdir(stopped.root, 1) / tr.RECORD).exists()


def test_never_started_original_is_never_admitted(env):
    env.original, env.sched = original(env), Scheduler()
    with pytest.raises((ValueError, OSError)):
        admit(env)


@pytest.mark.parametrize("code", [1, -9])
def test_nonresumable_terminal_is_never_admitted(env, code):
    trainer = Trainer(env, {"dense": ["stop"]})

    def stopped(config, out, extra, logs, seconds):
        trainer(config, out, extra, logs, seconds)
        return code

    with pytest.raises(ValueError, match="originals preserved"):
        go(env, stopped)
    env.original, env.sched = original(env), Scheduler()
    funding(env)
    with pytest.raises(ValueError, match="no automatic retry"):
        admit(env)


def test_step0_exit75_record_is_never_admitted(env):
    assert go(env, stopping(Trainer(env, {"dense": ["stop"]}), steps=0)) == 75
    env.original, env.sched = original(env), Scheduler()
    funding(env)
    with pytest.raises(ValueError, match="step-0"):
        admit(env)


def _stale(**edit):
    return edit


STALE = [
    _stale(prior_funding_sha256="0" * 64),
    _stale(prior_admission_sha256="0" * 64),
    _stale(prior_transition_sha256="0" * 64),
    _stale(coordinator_sha256="0" * 64),
    _stale(transition=2),
    _stale(transition_version="v0"),
    _stale(funded_by="issue"),
    _stale(additional_GPU_funded_by_issue=True),
    _stale(parent_job="899766"),
    _stale(node="evc40"),
    _stale(placement="relocation"),
    _stale(placement="silent_rebind"),
    _stale(shared_future_bytes=1),
    _stale(reserve_bytes=0),
    _stale(max_slice_seconds=60),
    _stale(node_local_envelope_bytes=1),
    _stale(supersedes_unused_transition=0),
    _stale(unexpected=True),
]


@pytest.mark.parametrize("edit", STALE)
def test_stale_placement_or_funding_refused(stopped, edit):
    funding(stopped, **edit)
    with pytest.raises(ValueError):
        admit(stopped)
    assert not (tr.tdir(stopped.root, 1) / tr.RECORD).exists()


def test_funding_changed_after_admission_refused(stopped):
    funding(stopped)
    admit(stopped)
    funding(stopped, assigned_GPU="1")
    trainer = Trainer(stopped)
    with pytest.raises(ValueError, match="changed"):
        fit(stopped, trainer)
    assert trainer.calls == []


def test_admission_on_non_owning_node_refused(stopped):
    funding(stopped)
    with pytest.raises(ValueError, match="owning node"):
        admit(stopped, host="evc40")


def _pending(s):
    s.jobs[REPLACEMENT]["job_state"] = ["PENDING"]


def _wrong_node(s):
    s.jobs[REPLACEMENT]["nodes"] = "evc34"


def _reservation(s):
    s.jobs[REPLACEMENT]["reservation"] = "maint"


def _two_gpus(s):
    s.jobs[REPLACEMENT]["tres_alloc_str"] = "cpu=2,node=1,gres/gpu=2"


def _absent(s):
    del s.jobs[REPLACEMENT]


def _maintenance(s):
    s.states["evc31"] = ["IDLE", "MAINTENANCE"]


def _reserved(s):
    s.states["evc31"] = ["MIXED", "RESERVED"]


def _drain(s):
    s.states["evc31"] = "MIXED+DRAIN"


def _original_live(s):
    s.jobs["899766"]["job_state"] = ["RUNNING"]


def _three_nodes(s):
    s.gpu_nodes |= {"evc34"}


def _not_listed(s):
    s.gpu_nodes = {"evc40"}


@pytest.mark.parametrize(
    "edit",
    [
        _pending,
        _wrong_node,
        _reservation,
        _two_gpus,
        _absent,
        _maintenance,
        _reserved,
        _drain,
        _original_live,
        _three_nodes,
        _not_listed,
    ],
)
def test_scheduler_allocation_rechecked_before_gpu_intent(stopped, edit):
    funding(stopped)
    admit(stopped)
    edit(stopped.sched)
    trainer = Trainer(stopped)
    with pytest.raises(ValueError):
        fit(stopped, trainer)
    assert trainer.calls == []
    assert not (tr.tdir(stopped.root, 1) / "journal").exists()


def test_predecessor_still_live_is_not_admitted(stopped):
    _original_live(stopped.sched)
    funding(stopped)
    with pytest.raises(ValueError, match="still live"):
        admit(stopped)


def test_byte_or_file_reserve_rechecked(stopped):
    funding(stopped, required_free_files=2**62)
    admit(stopped)
    trainer = Trainer(stopped)
    with pytest.raises(ValueError, match="reserve"):
        fit(stopped, trainer)
    assert trainer.calls == []
    assert (stopped.root / "production" / "dense" / "checkpoint.pt").exists()


def test_slice_bounded_by_replacement_funding(stopped):
    funding(stopped)
    admit(stopped)
    for seconds in (4 * 3600 + 1, 3599, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="slice"):
            fit(stopped, Trainer(stopped), slice_seconds=seconds)


def test_duplicate_transition_refused(stopped):
    funding(stopped)
    admit(stopped)
    with pytest.raises(ValueError, match="Duplicate"):
        admit(stopped)
    stopped.sched.jobs[REPLACEMENT]["job_state"] = ["CANCELLED"]
    stopped.sched.jobs["912346"] = Scheduler.running("912346", "evc31")
    funding(stopped, k=2, parent_job="912346")
    with pytest.raises(ValueError, match="Duplicate transition"):
        admit(stopped)
    funding(stopped, k=2, parent_job="912346", supersedes_unused_transition=1)
    record = admit(stopped)
    assert record["predecessor"]["segment"] == 0
    assert record["predecessor"]["superseded_unused_transitions"] == [1]


def test_initial_state_restart_refused(stopped):
    funding(stopped)
    admit(stopped)
    dense = stopped.root / "production" / "dense"
    dense.rename(dense.with_name("dense-moved"))
    trainer = Trainer(stopped)
    with pytest.raises((ValueError, OSError)):
        fit(stopped, trainer)
    assert trainer.calls == []


def test_second_transition_chains_from_the_replacement_exit75(stopped):
    env = stopped
    funding(env)
    admit(env)
    trainer = stopping(Trainer(env, {"dense": ["stop", "complete"]}))
    assert fit(env, trainer) == 75
    first = tr.tdir(env.root, 1) / "journal" / "dense"
    assert json.loads((first / "invocation-01-terminal.json").read_text())["resumable"] is True
    with pytest.raises(ValueError, match="still live"):
        funding(env, k=2, parent_job="912346")
        admit(env)
    env.sched.jobs[REPLACEMENT]["job_state"] = ["TIMEOUT"]
    env.sched.jobs["912346"] = Scheduler.running("912346", "evc31")
    record = admit(env)
    assert record["predecessor"]["segment"] == 1
    assert record["predecessor"]["parent_job"] == REPLACEMENT
    calls = []

    def recorded(config, out, extra, logs, seconds):
        calls.append((out.name, list(extra)))
        return trainer(config, out, extra, logs, seconds)

    assert fit(env, recorded) == 0
    assert calls[0] == (
        "dense",
        ["--resume", str(env.root / "production" / "dense" / "checkpoint.pt")],
    )
    assert calls[1] == ("suffix_r32", [])


def relocated(env, node="evc34", **edit):
    env.sched = Scheduler(node)
    cache = Path(launch.admitted(env.root, env.adm, env.pins)["inputs"]["cache"])
    if not cache.exists():
        write(cache / "member-0000.pt.json", {"member": 0})
    value = funding(env, placement="relocation", node=node)
    state = launch.admitted(env.root, env.adm, env.pins)
    files = {}
    skip = tr.tdir(env.root, 1)
    for tree in tr.required_trees(env.root, env.adm, state):
        files.update(tr._listing(launch, tree, skip, True))
    transport = {"transition": 1, "source_node": "evc31", "destination_node": node,
                 "files": files}  # fmt: skip
    transport_sha = write(skip / tr.TRANSPORT, transport)
    destination = {
        "status": tr.DESTINATION_STATUS,
        "funded_by": "root",
        "transition": 1,
        "node": node,
        "parent_job": REPLACEMENT,
        "transport_sha256": transport_sha,
        "same_absolute_paths": True,
        "path_rebinding": False,
    }
    destination.update(edit)
    write(skip / tr.DESTINATION, destination)
    return value


def test_relocation_requires_exact_transport_and_destination_admission(stopped):
    relocated(stopped)
    record = admit(stopped, host="evc34")
    assert record["relocation"]["transport_sha256"]
    trainer = Trainer(stopped)
    assert fit(stopped, trainer) == 0
    assert trainer.calls[0][0] == "dense" and trainer.calls[0][1][0] == "--resume"


@pytest.mark.parametrize(
    "edit", [{"path_rebinding": True}, {"same_absolute_paths": False}, {"transport_sha256": "0"}]
)
def test_relocation_without_exact_destination_admission_refused(stopped, edit):
    relocated(stopped, **edit)
    with pytest.raises(ValueError, match="destination admission"):
        admit(stopped, host="evc34")


def test_relocation_transport_differing_bytes_refused(stopped):
    relocated(stopped)
    (stopped.root / "production" / "dense" / "checkpoint.pt").write_bytes(b"silently-rebound")
    with pytest.raises(ValueError):
        admit(stopped, host="evc34")


def test_relocation_missing_transport_refused(stopped):
    relocated(stopped)
    (tr.tdir(stopped.root, 1) / tr.TRANSPORT).unlink()
    with pytest.raises(ValueError):
        admit(stopped, host="evc34")


def test_relocation_rechecks_transport_before_first_resumed_intent(stopped):
    relocated(stopped)
    admit(stopped, host="evc34")
    (stopped.adm / "extra-untransported.json").write_text("{}")
    trainer = Trainer(stopped)
    with pytest.raises(ValueError, match="transport"):
        fit(stopped, trainer)
    assert trainer.calls == []


def test_pins_match_the_actual_original_production_dispatch():
    audit = REPO / "initiative/audits/dense-suffix-full-production-start-20261014-v1"
    intent = json.loads((audit / "dispatch-intent.json").read_text())
    assert intent["funding_sha256"] == tr.ORIGINAL["funding_sha256"]
    assert intent["launcher_sha256"] == tr.ORIGINAL["launcher_sha256"]
    assert intent["publisher_sha256"] == tr.ORIGINAL["publisher_sha256"]
    assert str(intent["original_job"]) == tr.ORIGINAL["parent_job"]
    assert intent["node"] == tr.ORIGINAL["node"]
    assert (
        launch.sha(REPO / "scripts/newton/run_dense_suffix_production.py")
        == (tr.ORIGINAL["launcher_sha256"])
    )
    assert (
        launch.sha(REPO / "scripts/newton/dense_suffix_epoch_publisher.py")
        == (tr.ORIGINAL["publisher_sha256"])
    )
