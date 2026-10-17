"""Synthetic compiler fixtures; actual pinned source is admitted on its owner node."""

import ast
import json
from pathlib import Path

import pytest

from rsna.competitive import licensed_cached_serving as scope
from scripts.kaggle import licensed_cached_serving as compiler

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "initiative/audits/cached-full-study-factorial-20261017-v1"


def evidence():
    return [
        json.loads((AUDIT / n).read_text())
        for n in (
            "actual-four-full-complete.json",
            "actual-all12-private-verified.json",
            "actual-recovery-accepted.json",
        )
    ]


def synthetic_source():
    return """# %% [code]
ARMS=[{'file':'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt','arch':'coatnet_2_rw_224'}]
def load_model(pt_path):
    return torch.load(pt_path, map_location="cpu", weights_only=False)
def preparation():
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
    return pool.imap_unordered(_pre_one, test_ids, chunksize=2)
def queue_factory():
    q = queue.Queue(maxsize=QUEUE_MAX)
def fallback():
    print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
\nif __name__ == "__main__":
    main()"""


def arguments():
    return [
        synthetic_source(),
        (ROOT / "rsna/competitive/bounded_preparation.py").read_text(),
        Path(scope.__file__).read_text(),
        (ROOT / "rsna/competitive/licensed_pixel_cache.py").read_text(),
        (ROOT / "rsna/competitive/licensed_cached_mil.py").read_text(),
        *evidence(),
    ]


@pytest.mark.parametrize("role,windows", scope.FINAL_SHAS)
@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_all_four_exact_fixed_final_candidates_compile_without_altering_parent_arm(
    monkeypatch, role, windows, mode
):
    args = arguments()
    monkeypatch.setattr(compiler, "PARENT_SOURCE_SHA256", compiler.digest(args[0]))
    text, metadata = compiler.build(*args, role, windows, mode=mode)
    tree = ast.parse(text)
    arms = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ARMS" for t in n.targets)
    )
    assert (
        ast.literal_eval(arms.value)[0]["file"]
        == "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
    )
    assert metadata["encoder_weights_changed"] is False and metadata["serving_windows"] == 94
    assert metadata["head_sha256"] == scope.FINAL_SHAS[role, windows]
    assert metadata["qualification_cannot_submit"] is (mode == "qualification")
    assert metadata["source_sha256"] == compiler.digest(text)
    assert "FP32_outside_encoder_autocast" in text
    assert "reference.encode(gpu[None])" in text and "error <= 1e-6" in text
    assert "original_training_Readout" in text and "_counts == dict(started=6" in text
    assert "nproc = 2" in text and "queue.Queue(maxsize=2)" in text
    assert "weights_only=False" not in text
    assert "MODE_VALUE" not in text and "HEAD_SPEC_VALUE" not in text


@pytest.mark.parametrize("part", range(5))
def test_modified_original_or_helper_sources_are_rejected(monkeypatch, part):
    args = arguments()
    monkeypatch.setattr(compiler, "PARENT_SOURCE_SHA256", compiler.digest(args[0]))
    args[part] += "\n# changed original source\n"
    with pytest.raises(ValueError):
        compiler.build(*args, "original", 94, mode="competition")


def test_standalone_scope_preserves_actual_provenance_and_target_order():
    namespace = {"__name__": "synthetic_standalone_scope"}
    exec(compiler.standalone(Path(scope.__file__).read_text()), namespace)
    result = namespace["provenance"](
        *evidence(), "released_S6", 94, scope.FINAL_SHAS["released_S6", 94]
    )
    assert result["role"] == "released_S6" and result["training_windows"] == 94
    assert tuple(namespace["TARGETS"]) == tuple(scope.TARGETS)
