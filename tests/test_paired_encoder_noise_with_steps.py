"""New combined wrapper compilation only; synthetic sources are not GPU evidence."""

import ast
import base64

import pytest
from test_paired_encoder_noise_compiler import source_fixture

from scripts.kaggle import paired_encoder_noise_with_steps as combined


def fixture(monkeypatch):
    source, files, training, publication = source_fixture(monkeypatch)
    files["rsna/competitive/paired_step_timing.py"] = "# synthetic new timer\n"
    return source, files, training, publication


def test_combined_session_has_independent_fresh_timing_processes_and_reversed_order(monkeypatch):
    args = fixture(monkeypatch)
    source, meta = combined.build(*args)
    compile(source, "new-combined-synthetic-wrapper", "exec")
    benchmark = base64.b64decode(combined.literal(source, "BENCHMARK"), validate=True).decode()
    compile(benchmark, "new-combined-synthetic-timing-worker", "exec")
    main = next(
        n for n in ast.parse(benchmark).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = {
        n.func.id
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "timed_fit" in calls and "fit" not in calls
    assert (
        "class OrdinaryTimingRaptor(ImportanceReadout,CheckpointedEncode,RaptorClassifier)"
        in benchmark
    )
    assert "kind=OrdinaryTimingRaptor if arm=='ordinary' else WholeStudyRaptor" in benchmark
    assert "list(range(16))" in benchmark and "73136096" in benchmark
    assert "enumerate(('ordinary','controlled','controlled','ordinary'))" in source
    assert (
        "timing_records[run][rank]['initial_identity']==timing_records[0][rank]['initial_identity']"
        in source
    )
    assert "Path('/tmp/rsna-paired-update-timing-v2')" in source
    assert "retained_on_VM_only_until_teardown':True" in source
    assert meta["GPU_session_cap_seconds"] == 2700 and meta["benchmark_updates_total"] == 24
    assert meta["temporary_VM_checkpoint_cap_bytes"] == 8 * 2**30
    assert meta["minimum_temporary_disk_free_bytes"] == 12 * 2**30
    assert meta["temporary_checkpoints_published"] is False
    assert meta["competition_entries"] == 0 and meta["production_or_accuracy_admitted"] is False
    assert meta["prepared_only"] is True and combined.STATUS in source
    assert "hashlib.file_digest(parent_stream,'sha256')" in source
    assert (
        combined.literal(source, "FILES")["rsna/competitive/paired_step_timing.py"]
        == args[1]["rsna/competitive/paired_step_timing.py"]
    )


def test_missing_timer_cannot_prepare_combined_session(monkeypatch):
    source, files, training, publication = fixture(monkeypatch)
    files.pop("rsna/competitive/paired_step_timing.py")
    with pytest.raises(ValueError, match="Complete new noise"):
        combined.build(source, files, training, publication)


def test_root_rewrite_cannot_mutate_any_numerical_module_byte(monkeypatch):
    source, files, training, publication = fixture(monkeypatch)
    files["rsna/competitive/paired_step_timing.py"] += "# future literal paired-encoder-noise-v1\n"
    with pytest.raises(ValueError, match="must not alter any numerical module"):
        combined.build(source, files, training, publication)
