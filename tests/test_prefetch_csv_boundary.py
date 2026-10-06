"""The real chunk consumer resets CSV indices; prefetch must bind that exact table."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from rsna.schema import TARGETS


def test_probe_prefetch_uses_the_same_csv_roundtrip_table_as_chunk_consumer(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/kaggle"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prefetch_boundary_probe", scripts / "fp32_prefetch_performance_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [f"1.2.{i}" for i in range(1300)]
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(ids))
    competition = tmp_path / "competition"
    competition.mkdir()
    # Source row order/indices differ from the selected active-prefix CSV.
    source = pd.DataFrame(
        {"StudyInstanceUID": ids[::-1], "SeriesInstanceUID": [uid + ".1" for uid in ids[::-1]]}
    )
    source.to_csv(competition / "train_series.csv", index=False)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "checkpoint.pt").write_bytes(b"original")
    (bundle / "bundle-manifest.json").write_text(
        json.dumps({"inference_policy": {"cache_side": 384}})
    )
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps({"submission_sha256": "a" * 64, "original_kernel_slug": "owner/original"})
    )
    reference = pd.DataFrame(0.2, index=ids, columns=TARGETS)
    monkeypatch.setattr(module, "load_reference", lambda *a: reference)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda i: "T4")
    captured = {}

    class Pool:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

        def invoke(self, *a, **k):
            pass

    class Preparation:
        def __init__(self, frame, *a, **k):
            captured["frame"] = frame.copy()

        def close(self):
            pass

        def prepare(self, frame, *a, **k):
            pd.testing.assert_frame_equal(frame, captured["frame"])

    monkeypatch.setattr(module, "ResidentPool", Pool)
    monkeypatch.setattr(module, "PrefetchPreparation", Preparation)

    def run(options, **kwargs):
        table = pd.read_csv(options.series)
        kwargs["prepare_invoke"](table, ids[:32])
        raise RuntimeError("actual_source_csv_boundary_reached")

    monkeypatch.setattr(module.chunked_infer, "run", run)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--bundle",
            str(bundle),
            "--encoder-spec",
            str(tmp_path / "encoder.json"),
            "--competition",
            str(competition),
            "--roster",
            str(roster),
            "--output",
            str(out),
            "--reference-csv",
            str(tmp_path / "ref.csv"),
            "--reference-receipt",
            str(proof),
            "--study-count",
            "32",
        ],
    )
    with pytest.raises(RuntimeError, match="actual_source_csv_boundary_reached"):
        module.main()
    assert captured["frame"].index.tolist() == list(range(32))
