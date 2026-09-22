"""Independent adversarial review tests: no real labels, weights, network or GPU."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rsna.competitive import infer, train
from rsna.competitive.folds import _digest
from rsna.competitive.sampling import EffectiveBatchSampler, SeededDataset
from rsna.schema import CACHE_VERSION, TARGETS


def _binding():
    release = {
        "schema_version": 1,
        "manifest_sha256": "a" * 64,
        "series_sha256": "b" * 64,
        "source_revision": "synthetic",
        "release_receipt_sha256": "c" * 64,
        "semantic_registry_sha256": "d" * 64,
    }
    release["binding_sha256"] = _digest(release)
    root = Path(train.__file__).parents[1]
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }
    return {
        "release": release,
        "targets": list(TARGETS),
        "cache_version": CACHE_VERSION,
        "source_revision": "synthetic",
        "config": {"epochs": 1, "diagnostic_study_limit": 1},
        "code_sha256": hashes,
    }


@pytest.mark.parametrize(
    "changed", ["competitive/model.py", "preprocess.py", "dicom.py", "infer.py"]
)
def test_inference_refuses_changed_shared_transform_before_encoder_load(
    tmp_path, monkeypatch, changed
):
    binding = _binding()
    binding["code_sha256"][changed] = "0" * 64
    checkpoint = {
        "schema_version": 1,
        "optimizer_boundary": True,
        "epoch": 1,
        "next_update": 0,
        "binding": binding,
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)

    def no_encoder(*args):
        raise AssertionError("Changed transform reached encoder construction")

    monkeypatch.setattr(infer, "create_encoder", no_encoder)
    spec = tmp_path / "encoder.json"
    spec.write_text(json.dumps({"architecture": "diagnostic_tiny"}))
    args = argparse.Namespace(
        device="cpu", checkpoint=tmp_path / "unused", allow_diagnostic=True, encoder_spec=spec
    )
    with pytest.raises(ValueError, match="code differs"):
        infer.run(args)


class _PredictionData(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "images": torch.full((1, 1, 1, 2, 2), float(index)),
            "features": torch.zeros(1, 5),
            "series_mask": torch.ones(1, dtype=torch.bool),
            "slice_mask": torch.ones(1, 1, dtype=torch.bool),
            "positions": torch.zeros(1, 1, 2),
            "labels": torch.full((12,), float(index % 2)),
            "label_mask": torch.ones(12),
            "study_uid": f"study-{index}",
        }


class _PredictionModel(nn.Module):
    def forward(self, images, *args):
        return images[:, 0, 0, 0, 0, 0, None].expand(-1, 12)


def _distributed_prediction_rank(rank, rendezvous, size, interrupted):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        sampler = EffectiveBatchSampler(size, 2, 1, seed=11, epoch=0, rank=rank, world_size=2)
        loader = DataLoader(SeededDataset(_PredictionData(size)), batch_sampler=sampler)
        config = train.TrainConfig(diagnostic_study_limit=10, side=16)
        if interrupted:
            # One rank requests cancellation. Equal schedules coordinate both ranks.
            with pytest.raises(train.PredictionInterrupted):
                train.predict(_PredictionModel(), loader, torch.device("cpu"), config, [rank == 1])
        else:
            predictions, _ = train.predict(
                _PredictionModel(), loader, torch.device("cpu"), config, [False]
            )
            parts = [None, None]
            dist.all_gather_object(parts, predictions)
            if rank == 0:
                combined = pd.concat(parts, ignore_index=True)
                assert not combined.study_uid.duplicated().any()
                assert set(combined.study_uid) == {f"study-{i}" for i in range(size)}
                for row in combined.to_dict("records"):
                    i = int(row["study_uid"].split("-")[1])
                    assert row["ACL"] == pytest.approx(torch.tensor(float(i)).sigmoid().item())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [1, 5])
def test_distributed_prediction_padding_and_empty_rank(tmp_path, size):
    mp.spawn(
        _distributed_prediction_rank,
        args=((tmp_path / "rendezvous").as_uri(), size, False),
        nprocs=2,
        join=True,
    )


def test_distributed_prediction_single_rank_signal_cancels_both(tmp_path):
    mp.spawn(
        _distributed_prediction_rank,
        args=((tmp_path / "rendezvous").as_uri(), 5, True),
        nprocs=2,
        join=True,
    )


def _distributed_release_rank(rank, rendezvous, directory, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        path = Path(directory) / f"manifest-{rank}.bytes"
        path.write_bytes(b"modified" if mode == "local_mismatch" and rank == 1 else b"same")
        expected = hashlib.sha256(b"same").hexdigest()
        calls = []
        frame = pd.DataFrame({"study_uid": ["synthetic-only"], "split": ["train"]})

        def verify(studies, root):
            calls.append(True)
            assert rank == 0, "Noncoordinator replayed the full provenance unexpectedly"
            if mode == "root_failure":
                raise ValueError("synthetic-detail-must-not-be-broadcast")
            return frame, {"manifest_sha256": expected}

        train.load_verified_release = verify
        try:
            result, binding = train.verified_job_release(path, Path(directory), rank, 2)
            pd.testing.assert_frame_equal(result, frame)
            assert binding == {"manifest_sha256": expected}
            outcome = "passed"
        except ValueError as error:
            assert "synthetic-detail" not in str(error)
            outcome = "rejected"
        assert len(calls) == int(rank == 0)
        outcomes = [None, None]
        dist.all_gather_object(outcomes, outcome)
        if mode == "valid":
            assert outcomes == ["passed", "passed"]
        elif mode == "root_failure":
            assert outcomes == ["rejected", "rejected"]
        else:
            assert outcomes == ["passed", "rejected"]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["valid", "root_failure", "local_mismatch"])
def test_verified_job_release_coordinator_replay_and_rank_guards(tmp_path, mode):
    mp.spawn(
        _distributed_release_rank,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), mode),
        nprocs=2,
        join=True,
    )
