import copy
import hashlib
import io
import tarfile

import numpy as np
import pytest
import torch

from rsna.competitive.regional_context_train import (
    PARENT_SOURCE,
    REPRESENTATION,
    RegionalFeatures,
    feature_vector,
)
from rsna.competitive.released_context import ContextResidual, window_metadata
from rsna.competitive.released_context_train import PARENT_SHA, Recipe, Trainer
from scripts.newton.probe_regional_context_recovery import zero_start_teacher


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def dataset(path, role, *, change=None):
    path.mkdir(exist_ok=True)
    generator = np.random.default_rng(7)
    records = []
    c = torch.tensor([1, 25, 26, 47, 48, 94])
    slots, depth = window_metadata(c)
    _, mirror = window_metadata(c, mirrored=True)
    with tarfile.open(path / "features.tar", "w") as archive:
        for i in range(5):
            uid = f"synthetic-{i}"
            arrays = {
                "features": generator.normal(size=(2, 6, 8)).astype(np.float32),
                "regional": generator.normal(size=(2, 6, 7, 8)).astype(np.float16),
                "parent_logits": generator.normal(size=(2, 3)).astype(np.float32),
                "centers": c.numpy(),
                "slots": slots.numpy(),
                "depth": torch.stack((depth, mirror)).numpy(),
                "present": np.ones(6, dtype=bool),
            }
            if change:
                change(arrays)
            out = io.BytesIO()
            np.savez(out, **arrays)
            raw = out.getvalue()
            name = hashlib.sha256(uid.encode()).hexdigest() + ".npz"
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
            records.append(
                {
                    "study_uid": uid,
                    "file": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "archive": "features.tar",
                    "source_key": "self",
                }
            )
    labels = np.full((5, 3), 0.8, np.float32)
    labels[0, 1] = np.nan
    return RegionalFeatures(
        records,
        {"self": path},
        labels,
        np.isfinite(labels),
        {
            "representation": REPRESENTATION,
            "parent_weights_sha256": PARENT_SHA,
            "parent_source_sha256": PARENT_SOURCE,
        },
        role,
        channels=8,
        windows=6,
        targets=3,
    )


def test_control_and_candidate_have_same_metadata_labels_teacher_but_different_inputs(tmp_path):
    control = dataset(tmp_path / "control", "global_repeat")
    candidate = dataset(tmp_path / "candidate", "regional")
    a, b = control[0], candidate[0]
    assert a[0].shape == b[0].shape == (2, 6, 64)
    assert torch.equal(a[0][..., :8], b[0][..., :8])
    assert not torch.equal(a[0], b[0])
    for i in range(1, 6):
        assert torch.equal(a[i], b[i])
    assert torch.equal(a[6].nan_to_num(), b[6].nan_to_num())
    assert torch.equal(a[7], b[7])
    for part in range(8):
        assert torch.equal(a[0][..., :8], a[0][..., part * 8 : (part + 1) * 8])
    assert control.binding != candidate.binding


@pytest.mark.parametrize("role", ["global_repeat", "regional"])
def test_complete_unchanged_engine_recovery_and_predictions_for_both_inputs(tmp_path, role):
    data = dataset(tmp_path / role, role)
    recipe = Recipe(epochs=2, batch_size=2, embedding_dim=64, targets=3, width=8)
    full = Trainer(data, recipe)
    parent = np.stack([data[i][1].sigmoid().mean(0).numpy() for i in range(5)])
    assert np.array_equal(full.predictions(), parent)
    full.run(epoch_directory=tmp_path / "epochs")
    state, prediction = copy.deepcopy(full.state()), full.predictions()
    stopped = Trainer(data, recipe)
    stopped.run(stop_after=1)
    stopped.save(tmp_path / "stop.pt")
    resumed = Trainer(data, recipe)
    resumed.load(tmp_path / "stop.pt")
    resumed.run()
    equal(state, resumed.state())
    assert np.array_equal(prediction, resumed.predictions())
    assert [x["studies"] for x in resumed.history] == [2, 2, 1, 2, 2, 1]
    assert len(list((tmp_path / "epochs").glob("*.pt"))) == 2
    assert resumed.history[-1]["max_abs_correction"] > 0


def test_role_is_bound_to_resume_and_initial_parameters_match(tmp_path):
    c, r = dataset(tmp_path / "c", "global_repeat"), dataset(tmp_path / "r", "regional")
    recipe = Recipe(epochs=2, batch_size=2, embedding_dim=64, targets=3, width=8)
    control, regional = Trainer(c, recipe), Trainer(r, recipe)
    equal(control.model.state_dict(), regional.model.state_dict())
    control.run(stop_after=1)
    control.save(tmp_path / "control.pt")
    with pytest.raises(ValueError, match="changed"):
        regional.load(tmp_path / "control.pt")


def test_exact_production_parameter_count():
    a, b = ContextResidual(8192), ContextResidual(8192)
    assert sum(p.numel() for p in a.parameters()) == 1334796
    equal(a.state_dict(), b.state_dict())


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(regional=a["regional"].astype(np.float32)),
        lambda a: a.update(features=a["features"].astype(np.float16)),
        lambda a: a.update(parent_logits=a["parent_logits"].astype(np.float64)),
        lambda a: a.update(regional=np.full(a["regional"].shape, np.inf, np.float16)),
        lambda a: a.update(present=a["present"].astype(np.int64)),
        lambda a: a.update(depth=a["depth"] + np.float32(0.01)),
        lambda a: a.update(unexpected=np.array([1])),
    ],
)
def test_unaccepted_schema_precision_nonfinite_and_metadata_rejected(tmp_path, change):
    data = dataset(tmp_path, "regional", change=change)
    with pytest.raises(ValueError):
        data[0]


def test_archive_stat_and_member_hash_are_enforced(tmp_path):
    data = dataset(tmp_path / "data", "regional")
    data[0]
    with (tmp_path / "data/features.tar").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ValueError, match="changed"):
        data[0]
    other = dataset(tmp_path / "other", "regional")
    other.records[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ"):
        other[0]


def test_feature_vector_requires_exact_role_and_storage_precision():
    f, r = torch.ones(2, 6, 8), torch.ones(2, 6, 7, 8, dtype=torch.float16)
    with pytest.raises(ValueError):
        feature_vector(f, r, "ambiguous")
    with pytest.raises(ValueError):
        feature_vector(f, r.float(), "regional")


def test_zero_start_oracle_uses_the_exact_prediction_batch_and_tail_shape(tmp_path, monkeypatch):
    import scripts.newton.probe_regional_context_recovery as probe

    data = dataset(tmp_path / "data", "regional")
    recipe = Recipe(epochs=2, batch_size=2, embedding_dim=64, targets=3, width=8)
    shapes = []
    original = probe.tta_probabilities

    def observed(logits):
        shapes.append(tuple(logits.shape))
        return original(logits)

    monkeypatch.setattr(probe, "tta_probabilities", observed)
    teacher = zero_start_teacher(data, recipe)
    assert shapes == [(2, 2, 3), (2, 2, 3), (1, 2, 3)]
    assert np.array_equal(Trainer(data, recipe).predictions(), teacher)
