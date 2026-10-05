"""Two actual finite CPU workers exercise device sharding without a GPU claim."""

import importlib.util
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_competitive_inference import _fixture

from rsna.competitive import infer
from rsna.schema import TARGETS
from scripts.kaggle.resident_pool import ResidentPool

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/kaggle"

spec = importlib.util.spec_from_file_location("resident_chunk_pool", SCRIPTS / "chunked_infer.py")
chunk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk)


def test_two_real_workers_preserve_sigmoid_and_roster_across_chunks(tmp_path, monkeypatch):
    original, _ = _fixture(tmp_path, monkeypatch, size=3)
    original.precision = "fp32"
    infer.run(original)
    expected = pd.read_csv(original.output)
    args = Namespace(
        checkpoint=original.checkpoint,
        encoder_spec=original.encoder_spec,
        encoder_config=None,
        series=original.series,
        template=original.template,
        cache_root=tmp_path / "chunks",
        output=tmp_path / "pooled.csv",
        dicom_root=None,
        source_revision="synthetic-test",
        chunk_size=2,
        workers=0,
        device="cpu",
        precision="fp32",
        allow_diagnostic=True,
        isolate_study_errors=True,
        fallback_on_unusable_study=True,
        resilient_series=False,
    )
    # Existing chunk fallback requires resilient preparation; the actual cached
    # fixture uses direct source paths and still captures resident errors.
    args.isolate_study_errors = False
    args.fallback_on_unusable_study = False
    pool = ResidentPool(workers=2, timeout_seconds=120)
    try:

        def invoke(command, **options):
            def arg(name):
                return Path(command[command.index(name) + 1])

            for uid in pd.read_csv(arg("--template")).StudyInstanceUID:
                src = original.cache / "test_series" / uid
                dest = arg("--cache") / "test_series" / uid
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest)
            return pool.invoke(command, **options)

        result = chunk.run(args, invoke=invoke)
        actual = pd.read_csv(args.output)
        assert actual.StudyInstanceUID.tolist() == expected.StudyInstanceUID.tolist()
        np.testing.assert_allclose(
            actual[list(TARGETS)].to_numpy(), expected[list(TARGETS)].to_numpy(), rtol=0, atol=0
        )
        assert result["chunk_count"] == 2
        first = list(args.cache_root.glob("chunk-0000/attempt-*"))
        metrics = json_metrics(args.cache_root / "chunk-0000/submission.metrics.json")
        assert metrics["resident_device_workers"] == 2
        assert [r["resident_model_load_count"] for r in metrics["resident_shards"]] == [1, 1]
        last = json_metrics(args.cache_root / "chunk-0001/submission.metrics.json")
        assert last["resident_shards"][0]["resident_requests"] == 2
        assert not first
    finally:
        pool.close()
    assert all(p.poll() is not None for p in pool.children)


def json_metrics(path):
    import json

    return json.loads(path.read_text())


@pytest.mark.parametrize("workers", [0, 3])
def test_cpu_gpu_budget_rejects_unadmitted_worker_count(workers):
    with pytest.raises(ValueError, match="Bounded"):
        ResidentPool(workers=workers)
