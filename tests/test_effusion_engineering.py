"""Recovery evidence must reject scientifically material state changes."""

import copy

import pytest
import torch

from scripts.newton.probe_effusion_engineering import compare_recovery


@pytest.mark.parametrize("mutation", ["valid", "trainer", "base", "extra", "binding"])
def test_recovery_source_preserves_original_numerical_modules(mutation):
    from scripts.newton.probe_effusion_engineering import accept_numerical_parent, digest

    original = {
        "files": {
            "rsna/competitive/train.py": {"sha256": "a", "bytes": 1},
            "rsna/competitive/backbones.py": {"sha256": "b", "bytes": 1},
            "scripts/newton/probe_effusion_engineering.py": {"sha256": "c", "bytes": 1},
            "scripts/newton/launch_effusion_engineering.py": {"sha256": "d", "bytes": 1},
        }
    }
    revised = copy.deepcopy(original)
    revised["original_input_source_sha256"] = digest(original)
    for name in ("probe_effusion_engineering.py", "launch_effusion_engineering.py"):
        revised["files"]["scripts/newton/" + name]["sha256"] = "changed"
    if mutation == "trainer":
        revised["files"]["rsna/competitive/train.py"]["sha256"] = "changed"
    elif mutation == "base":
        revised["files"]["rsna/competitive/backbones.py"]["sha256"] = "changed"
    elif mutation == "extra":
        revised["files"]["scripts/newton/runtime_environment.py"] = {"sha256": "new", "bytes": 1}
    elif mutation == "binding":
        revised["original_input_source_sha256"] = "unbound"
    if mutation == "valid":
        accept_numerical_parent(revised, original)
    else:
        with pytest.raises(ValueError):
            accept_numerical_parent(revised, original)


def test_initialization_uses_independent_child_per_arm(tmp_path, monkeypatch):
    import json

    from scripts.newton import probe_effusion_engineering as probe

    commands = []

    def child(command, *, stderr, timeout):
        commands.append(command)
        assert stderr.name.endswith("-initialization.stderr")
        assert timeout == 300
        return json.dumps({"complete_model_sha256": "same", "all_RNG_sha256": "same"}).encode()

    monkeypatch.setattr(probe.subprocess, "check_output", child)
    results = [
        probe.initialization_process(tmp_path, tmp_path, arm) for arm in ("control", "candidate")
    ]
    assert results[0] == results[1]
    assert len(commands) == 2
    assert [c[c.index("--arm") + 1] for c in commands] == ["control", "candidate"]
    assert all(c[c.index("--phase") + 1] == "initialization" for c in commands)
    with pytest.raises(FileExistsError):
        probe.initialization_process(tmp_path, tmp_path, "control")


def records():
    full = dict(
        schema_version=2,
        epoch=2,
        next_update=0,
        step=4,
        optimizer_boundary=True,
        model={"a": torch.tensor([1.0])},
        model_storage={"format": "frozen_base_delta_v1", "base": "original"},
        optimizer={"state": {0: {"exp_avg": torch.tensor([0.1])}}, "param_groups": [{"lr": 0.01}]},
        rng_by_rank=[{"torch": torch.tensor([1], dtype=torch.uint8)}],
        binding={"supervision": True},
    )
    partial = dict(epoch=0, next_update=1, step=1, binding=full["binding"])
    updates = [
        dict(epoch=e, update=u, optimizer_steps=i, loss=0.1, gradient_norm=0.2, observed_cells=2)
        for i, (e, u) in enumerate([(0, 1), (0, 2), (1, 1), (1, 2)], 1)
    ]
    invocations = [
        dict(status="completed", exit_code=0, updates=updates),
        dict(status="checkpointed_for_resume", exit_code=75, updates=updates[:1]),
        dict(status="completed", exit_code=0, updates=updates[1:]),
    ]
    return full, copy.deepcopy(full), partial, invocations, [b"same", b"same"]


def test_complete_recovery():
    assert len(compare_recovery(*records())) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "optimizer",
        "rng",
        "lr",
        "base",
        "counter",
        "binding",
        "predictions",
        "trajectory",
        "partial",
        "nonfinite",
    ],
)
def test_reject_changed_recovery(mutation):
    full, replay, partial, invocations, predictions = records()
    if mutation == "model":
        replay["model"]["a"][0] = 2
    elif mutation == "optimizer":
        replay["optimizer"]["state"][0]["exp_avg"][0] = 0.2
    elif mutation == "rng":
        replay["rng_by_rank"][0]["torch"][0] = 2
    elif mutation == "lr":
        replay["optimizer"]["param_groups"][0]["lr"] = 0.02
    elif mutation == "base":
        replay["model_storage"]["base"] = "changed"
    elif mutation == "counter":
        replay["step"] = 3
    elif mutation == "binding":
        replay["binding"]["supervision"] = False
    elif mutation == "predictions":
        predictions[1] = b"changed"
    elif mutation == "trajectory":
        invocations[2] = copy.deepcopy(invocations[2])
        invocations[2]["updates"][0]["loss"] = 0.3
    elif mutation == "partial":
        partial["step"] = 2
    else:
        replay["model"]["a"][0] = float("nan")
    with pytest.raises(ValueError):
        compare_recovery(full, replay, partial, invocations, predictions)


@pytest.mark.parametrize(
    "mutation", ["valid", "missing", "extra", "duplicate", "escape", "linked", "wrong_hash"]
)
def test_source_envelope(tmp_path, mutation):
    import hashlib
    import io
    import tarfile

    from scripts.newton.launch_effusion_engineering import unpack_source

    payload = b"immutable numerical source"
    archive = tmp_path / "source.tar"
    expected = {"rsna/a.py": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
    names = ["rsna/a.py"]
    if mutation == "missing":
        names = []
    elif mutation == "extra":
        names.append("extra.py")
    elif mutation == "duplicate":
        names *= 2
    elif mutation == "escape":
        names = ["../escape"]
    elif mutation == "wrong_hash":
        expected["rsna/a.py"]["sha256"] = "0" * 64
    with tarfile.open(archive, "w") as handle:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            if mutation == "linked":
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/elsewhere"
            handle.addfile(info, io.BytesIO(payload))
    root = tmp_path / "output"
    root.mkdir()
    if mutation == "valid":
        unpack_source(archive, root, expected)
        assert (root / "rsna/a.py").read_bytes() == payload
    else:
        with pytest.raises(ValueError):
            unpack_source(archive, root, expected)
