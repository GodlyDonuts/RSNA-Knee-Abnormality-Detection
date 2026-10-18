"""New paired execution against the unchanged real optimizer, synthetic inputs only."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "connected-witness-paths-20261017-v1")]
SharedBatch = importlib.import_module("shared_pair").SharedBatch
fit_pair = importlib.import_module("shared_pair").fit_pair
SyntheticOriginal = importlib.import_module("check_reader_recovery").SyntheticOriginal
identical = importlib.import_module("check_reader_recovery").identical
PathCachedData = importlib.import_module("path_reader").PathCachedData
PathTrainer = importlib.import_module("path_reader").PathTrainer
SpatialRecipe = importlib.import_module("rsna.competitive.spatial_train").SpatialRecipe


def source():
    data = PathCachedData(SyntheticOriginal("fitting"), {"new_synthetic_pair_test": True})
    return data


def engine(data, arm):
    return PathTrainer(
        data,
        SpatialRecipe(epochs=2, channels=8, width=4),
        arm=arm,
        device=torch.device("cpu"),
        binding={
            "diagnostic_only": True,
            "path_mechanism_source": {"new_synthetic_pair_test": True},
            "production_eligible": False,
        },
    )


def pair():
    cache = SharedBatch(source(), torch.device("cpu"))
    return cache, {arm: engine(cache, arm) for arm in ("aligned", "null")}


def decoded(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def compare(root, reference):
    for arm in ("aligned", "null"):
        identical(decoded(root / arm / "checkpoint.pt"), decoded(reference / arm / "checkpoint.pt"))
        for epoch in (1, 2):
            identical(
                decoded(root / arm / "epoch-checkpoints" / f"epoch-{epoch:03}.pt"),
                decoded(reference / arm / "epoch-checkpoints" / f"epoch-{epoch:03}.pt"),
            )


def test_shared_encoder_same_complete_states_and_predictions_as_independent_arms(tmp_path):
    reference = tmp_path / "reference"
    for arm in ("aligned", "null"):
        e = engine(source(), arm)
        e.fit(reference / arm)
    cache, engines = pair()
    result = fit_pair(engines, cache, tmp_path / "shared")
    assert result["source_fetches"] == 32 and result["reused_fetches"] == 32
    assert not cache.rows and result["completed_per_arm"] == 4
    compare(tmp_path / "shared", reference)
    for arm, e in engines.items():
        original = engine(source(), arm)
        original.load(reference / arm / "checkpoint.pt")
        test = PathCachedData(SyntheticOriginal("outer"), {"new_synthetic_pair_test": True})
        assert e.predictions(test).to_csv(index=False) == original.predictions(test).to_csv(
            index=False
        )


def test_interrupted_between_arms_rolls_only_lagging_arm_forward(tmp_path):
    cache, engines = pair()
    fit_pair(engines, cache, tmp_path / "full")
    cache, engines = pair()

    def crash(step):
        if step == 1:
            raise RuntimeError("new engineering crash between original arm updates")

    with pytest.raises(RuntimeError, match="engineering crash"):
        fit_pair(engines, cache, tmp_path / "interrupted", between_arms=crash)
    assert decoded(tmp_path / "interrupted/aligned/checkpoint.pt")["completed"] == 1
    assert decoded(tmp_path / "interrupted/null/checkpoint.pt")["completed"] == 0
    cache, engines = pair()
    result = fit_pair(engines, cache, tmp_path / "interrupted", resume=True)
    assert result["source_fetches"] == 32 and result["reused_fetches"] == 24
    compare(tmp_path / "interrupted", tmp_path / "full")


def test_stop_resume_pair_full_states_equal(tmp_path):
    cache, engines = pair()
    fit_pair(engines, cache, tmp_path / "full")
    cache, engines = pair()
    fit_pair(engines, cache, tmp_path / "split", stop_after=1)
    cache, engines = pair()
    fit_pair(engines, cache, tmp_path / "split", resume=True)
    compare(tmp_path / "split", tmp_path / "full")


def test_shared_row_mutation_is_rejected():
    cache = SharedBatch(source(), torch.device("cpu"))
    cache.begin([0], 2)
    row = cache[0]
    row["maps"].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        cache[0]


def test_random_source_cannot_be_shared():
    class RandomSource:
        binding = {"synthetic": True}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"maps": torch.randn(2)}

    cache = SharedBatch(RandomSource(), torch.device("cpu"))
    cache.begin([0], 2)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="randomness"):
        cache[0]
    assert torch.equal(before, torch.get_rng_state())


def test_cache_cannot_outlive_or_cross_update_boundaries():
    cache = SharedBatch(source(), torch.device("cpu"))
    cache.begin([0], 2)
    cache[0]
    with pytest.raises(ValueError):
        cache.finish()
    with pytest.raises(ValueError):
        cache.begin([1], 2)
    cache[0]
    cache.finish()
    with pytest.raises(ValueError):
        cache[0]
    cache.begin([0], 2)
    cache[0]
    cache[0]
    cache.finish()
    assert cache.source_fetches == 2 and cache.reused_fetches == 2


def test_cache_budget_refuses_retention():
    cache = SharedBatch(source(), torch.device("cpu"), max_bytes=1)
    cache.begin([0], 2)
    with pytest.raises(MemoryError):
        cache[0]
    assert not cache.rows and cache.source_fetches == 0


def test_existing_pair_cannot_be_reinitialized(tmp_path):
    cache, engines = pair()
    fit_pair(engines, cache, tmp_path / "pair", stop_after=1)
    cache, engines = pair()
    with pytest.raises(FileExistsError):
        fit_pair(engines, cache, tmp_path / "pair")


def test_actual_new_wrapper_fresh_process_stop_resume(tmp_path):
    import subprocess

    code = """import sys
from pathlib import Path
sys.path[:0]=[sys.argv[3],sys.argv[4]]
from test_shared_pair import pair
from shared_pair import fit_pair
root=Path(sys.argv[1]);phase=sys.argv[2]
cache,engines=pair()
fit_pair(engines,cache,root,resume=phase=='resume',stop_after=1 if phase=='stop' else None)
"""
    for name, phase in [("full", "full"), ("split", "stop"), ("split", "resume")]:
        p = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(tmp_path / name),
                phase,
                str(HERE),
                str(HERE.parent / "connected-witness-paths-20261017-v1"),
            ],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
    compare(tmp_path / "split", tmp_path / "full")
