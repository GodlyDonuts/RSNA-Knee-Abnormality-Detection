"""Recovery evidence must reject scientifically material state changes."""

import copy

import pytest
import torch

from scripts.newton.probe_effusion_engineering import compare_recovery


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
