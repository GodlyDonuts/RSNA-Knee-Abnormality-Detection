import ast
from pathlib import Path

import pytest
from test_licensed_parent_serving import accepted_epoch as accepted_epoch

from scripts.kaggle import licensed_parent_serving as compiler

ROOT = Path(__file__).resolve().parents[1]


def fixture_parent():
    return """# Original boot intentionally removed.
# %% [code]
import torch,numpy as np,threading,queue,os,json,time
NORM='imagenet'
ARMS=[{'file':'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt','arch':'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k','res':384,'w':1.0}]
def infer_probs(*args):return np.ones(12,dtype=np.float32)
def gpu_windows(*args,**kwargs):return 'unchanged original images'
def infer_pair_probs(*args):return 'unchanged original two-view math'
def load_model(pt_path,arch_default,res_default,device,ngpu=1):
    assert observed_sha=='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
    return torch.load(pt_path, map_location="cpu", weights_only=False)
def main():
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
    q = queue.Queue(maxsize=QUEUE_MAX)
    for item in pool.imap_unordered(_pre_one, test_ids, chunksize=2):pass
    print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
if __name__ == "__main__":
    main()"""


def sources():
    return [
        (ROOT / n).read_text()
        for n in (
            "rsna/competitive/bounded_preparation.py",
            "rsna/competitive/licensed_parent_serving.py",
            "rsna/competitive/licensed_pixel_cache.py",
            "rsna/competitive/licensed_full_bag.py",
        )
    ]


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_compiled_deployment_keeps_original_math_and_changes_only_exact_checkpoint(
    monkeypatch, accepted_epoch, mode
):
    _, training, pub, _ = accepted_epoch
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE_SHA256", compiler.digest(parent))
    source, binding = compiler.build(parent, *sources(), training, pub, mode=mode)
    ns = {"__name__": "compile_only_not_GPU_acceptance"}
    exec(compile(source, "synthetic_compiler_fixture", "exec"), ns)
    assert ns["ARMS"] == [
        {
            "file": "epoch-001.pt",
            "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
            "res": 384,
            "w": 1.0,
        }
    ]
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert (ns["infer_probs"](None, None, None) == 1).all()
    assert ns["_adapted_counts"]["completed"] == 1
    assert (
        "Original boot intentionally removed" not in source and "pool.imap_unordered" not in source
    )
    assert "weights_only=True" in source and "sid[:16]" not in source
    assert binding["new_weight_Dataset_required"] is False and binding[
        "qualification_cannot_submit"
    ] == (mode == "qualification")
    last = ast.parse(source).body[-1]
    assert isinstance(last, ast.If) and "_adapted_qualify()" in ast.unparse(last)


@pytest.mark.parametrize("change", ["parent", "bounded", "helper", "codec", "mixin", "mode"])
def test_unknown_runtime_helpers_or_mode_fail_before_notebook_generation(
    monkeypatch, accepted_epoch, change
):
    _, training, pub, _ = accepted_epoch
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE_SHA256", compiler.digest(parent))
    helpers = sources()
    mode = "qualification"
    if change == "parent":
        parent += "changed"
    elif change == "mode":
        mode = "partial"
    else:
        helpers[["bounded", "helper", "codec", "mixin"].index(change)] += "\n# changed source\n"
    with pytest.raises(ValueError):
        compiler.build(parent, *helpers, training, pub, mode=mode)
