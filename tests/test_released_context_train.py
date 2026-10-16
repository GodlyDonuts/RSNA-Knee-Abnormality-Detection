import copy
import hashlib
import io
import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest
import torch

from rsna.competitive.released_context import window_metadata
from rsna.competitive.released_context_train import FrozenFeatures, Recipe, Trainer, learning_rate


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    else:
        assert a == b


def dataset(tmp_path, *, archive=False, count=5):
    tmp_path.mkdir(exist_ok=True)
    g = np.random.default_rng(4)
    records = []
    files = []
    for i in range(count):
        uid = str(i)
        c = torch.tensor([1, 25, 26, 47, 48, 94])
        slot, depth = window_metadata(c)
        _, mirror = window_metadata(c, mirrored=True)
        buf = io.BytesIO()
        np.savez(
            buf,
            features=g.normal(size=(2, 6, 8)).astype(np.float32),
            parent_logits=g.normal(size=(2, 3)).astype(np.float32),
            centers=c.numpy(),
            slots=slot.numpy(),
            depth=torch.stack([depth, mirror]).numpy(),
            present=np.ones(6, dtype=bool),
        )
        raw = buf.getvalue()
        name = hashlib.sha256(uid.encode()).hexdigest() + ".npz"
        row = {
            "study_uid": uid,
            "file": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "source_kernel": "self",
        }
        if archive:
            row["archive"] = "features.tar"
        else:
            (tmp_path / name).write_bytes(raw)
        files.append((name, raw))
        records.append(row)
    if archive:
        with tarfile.open(tmp_path / "features.tar", "w") as handle:
            for name, raw in files:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                handle.addfile(member, io.BytesIO(raw))
    y = np.full((count, 3), 0.8, np.float32)
    y[0, 1] = np.nan
    return FrozenFeatures(
        records,
        {"self": tmp_path},
        y,
        np.isfinite(y),
        {"sha": "original"},
        embedding_dim=8,
        targets=3,
    )


def recipe():
    return Recipe(epochs=2, batch_size=2, embedding_dim=8, targets=3, width=8)


@pytest.mark.parametrize("archived", [False, True])
def test_full_engine_stop_resume_complete_state_uneven_batch_and_predictions(tmp_path, archived):
    data = dataset(tmp_path / "data", archive=archived)
    full = Trainer(data, recipe())
    full.run(epoch_directory=tmp_path / "epochs")
    full_state = copy.deepcopy(full.state())
    full_prediction = full.predictions()
    stopped = Trainer(data, recipe())
    stopped.run(stop_after=1)
    stopped.save(tmp_path / "stop.pt")
    resumed = Trainer(data, recipe())
    resumed.load(tmp_path / "stop.pt")
    resumed.run()
    same(full_state, resumed.state())
    assert np.array_equal(full_prediction, resumed.predictions())
    assert [r["studies"] for r in resumed.history] == [2, 2, 1, 2, 2, 1]
    assert len(list((tmp_path / "epochs").glob("*.pt"))) == 2
    assert resumed.history[1]["max_abs_correction"] > 0
    assert all(np.isfinite(r["loss"]) for r in resumed.history)
    with pytest.raises(ValueError, match="already exists"):
        resumed.save(tmp_path / "stop.pt")


def test_epoch_boundary_resume_and_final_resume_do_not_replay(tmp_path):
    d = dataset(tmp_path / "data")
    full = Trainer(d, recipe())
    full.run()
    stopped = Trainer(d, recipe())
    stopped.run(stop_after=3)
    stopped.save(tmp_path / "epoch.pt")
    resumed = Trainer(d, recipe())
    resumed.load(tmp_path / "epoch.pt")
    resumed.run()
    same(full.state(), resumed.state())
    resumed.save(tmp_path / "final.pt")
    final = Trainer(d, recipe())
    final.load(tmp_path / "final.pt")
    final.run()
    same(full.state(), final.state())


def test_changed_recipe_input_binding_or_cursor_rejected(tmp_path):
    d = dataset(tmp_path / "data")
    t = Trainer(d, recipe())
    t.run(stop_after=1)
    t.save(tmp_path / "s.pt")
    other = Trainer(d, replace(recipe(), learning_rate=1e-3))
    with pytest.raises(ValueError, match="changed"):
        other.load(tmp_path / "s.pt")
    d.binding = {"sha": "different"}
    with pytest.raises(ValueError, match="changed"):
        Trainer(d, recipe()).load(tmp_path / "s.pt")


def test_exact_member_hash_and_original_stat_required(tmp_path):
    d = dataset(tmp_path / "data")
    d[0]
    path = tmp_path / "data" / d.records[0]["file"]
    path.write_bytes(path.read_bytes() + b"change")
    with pytest.raises(ValueError, match="changed"):
        d[0]
    d = dataset(tmp_path / "other")
    d.records[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="failed"):
        d[0]


def test_archive_rejects_link_duplicate_or_unexpected_member(tmp_path):
    d = dataset(tmp_path / "data", archive=True)
    path = tmp_path / "data" / "features.tar"
    with tarfile.open(path, "a") as handle:
        member = tarfile.TarInfo("../outside.npz")
        member.size = 1
        handle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        FrozenFeatures(d.records, d.roots, d.labels, d.masks, d.binding, embedding_dim=8, targets=3)


def test_warmup_cosine_schedule_and_production_updates():
    r = Recipe()
    n = (3887 + r.batch_size - 1) // r.batch_size
    assert n * r.epochs == 2430
    values = [learning_rate(r, i, n) for i in range(n * r.epochs)]
    assert values[0] == r.learning_rate / n
    assert values[n - 1] == r.learning_rate and values[-1] == 0
    assert all(b >= a for a, b in zip(values[: n - 1], values[1:n], strict=True))
    assert all(b <= a for a, b in zip(values[n:-1], values[n + 1 :], strict=True))


@pytest.mark.parametrize(
    "change",
    [
        dict(epochs=1),
        dict(batch_size=0),
        dict(limit=1),
        dict(source_weight=0),
        dict(weight_decay=-1),
        dict(layers=3),
    ],
)
def test_invalid_recipe_rejected(change):
    with pytest.raises(ValueError):
        Recipe(**change)


def test_private_manifest_hash_and_source_binding(tmp_path):
    from rsna.competitive.released_context_train import digest
    from scripts.newton.run_released_context_fit import load_data

    d = dataset(tmp_path / "data")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": d.records}))
    inputs = {
        "index": str(index),
        "index_sha256": digest(index),
        "uids_sha256": hashlib.sha256(
            json.dumps([r["study_uid"] for r in d.records], separators=(",", ":")).encode()
        ).hexdigest(),
        "clinical_evaluation_used": False,
        "source_rows_filtered_before_numeric_read": True,
        "independent_internal_validation": False,
        "source_files": {},
        "roots": {"self": str(tmp_path / "data")},
        "labels": [[0.8] * 12] * len(d),
        "masks": [[True] * 12] * len(d),
        "binding": {"source": "original"},
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(inputs))
    loaded, _ = load_data(path, digest(path))
    assert len(loaded) == 5
    with pytest.raises(ValueError, match="manifest"):
        load_data(path, "0" * 64)
