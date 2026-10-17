"""Require actual qualification and preserve the original MRI/rank math in deployment."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.kaggle import licensed_fov_competition as c

ROOT = Path(__file__).resolve().parents[1]


def qualified():
    return {
        "status": "COMPLETE",
        "source_sha256": c.QUALIFIED_SOURCE,
        "receipt": {
            "status": "actual_original32_bothT4_unchanged_weights_full384_field_qualified",
            "parent_checkpoint_sha256": c.WEIGHTS_SHA256,
            "loaded_parameter_count": 73136096,
            "studies": 32,
            "maximum_cross_GPU384_probability_difference": 0.0,
            "competition_entry": False,
            "devices": [
                {
                    "index": index,
                    "GPU": "Tesla T4",
                    "all_learned_parameters_equal_320_384": True,
                    "control_probability_error": 0.0001,
                    "learned_parameter_sha256": "a" * 64,
                    "roles": [
                        {
                            "input_side": side,
                            "CPU_GPU_inputs_bit_equal": True,
                            "original320_input_hashes_reproduced": True,
                            "studies": 32,
                            "peak_GPU_bytes": 4 * 2**30,
                        }
                        for side in (320, 384)
                    ],
                }
                for index in range(2)
            ],
        },
    }


def fake_parent():
    return """# %% [code]
import numpy as np,threading,queue,json
INPUT_RES = 320
NORM = "imagenet"
def load_model(*args,**kwargs):return 'original model'
def gpu_windows(*args,**kwargs):return 'original windows'
def infer_pair_probs(model,plain,mirrored,device):
    assert plain.shape==mirrored.shape==(94,3,320,320)
    return np.arange(12,dtype=np.float32)/12
def main():
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
    q = queue.Queue(maxsize=QUEUE_MAX)
    for item in pool.imap_unordered(_pre_one, test_ids, chunksize=2):pass
    print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
if __name__ == "__main__":
    main()"""


def test_only_field_and_boundaries_change_the_original_pair_probability_math(monkeypatch):
    parent = fake_parent()
    monkeypatch.setattr(c, "PARENT_SOURCE_SHA256", c.digest(parent))
    bounded = (ROOT / "rsna/competitive/bounded_preparation.py").read_text()
    runner = (ROOT / "rsna/competitive/licensed_fov_qualification.py").read_text()
    source, binding = c.build(parent, bounded, runner, qualified())
    ns = {"__name__": "synthetic_non_GPU_field_deployment"}
    exec(compile(source, "synthetic_original_pair", "exec"), ns)

    class Shape:
        shape = (94, 3, 384, 384)

    assert ns["INPUT_RES"] == 384
    np.testing.assert_array_equal(
        ns["infer_pair_probs"](None, Shape(), Shape(), None), np.arange(12, dtype=np.float32) / 12
    )
    assert ns["_fov_counts"]["pairs_completed"] == 1
    assert "nproc = 2" in source and "limit=4" in source and "maxsize=2" in source
    assert "sid[:16]" not in source and binding["original_94_windows_two_views_and_rank_output"]
    assert binding["new_Dataset_upload_required"] is False


@pytest.mark.parametrize(
    "change", ["pending", "source", "weights", "parameters", "control", "GPU_input", "single_GPU"]
)
def test_pending_partial_or_mismatched_engineering_proof_cannot_serve(change):
    actual = deepcopy(qualified())
    receipt = actual["receipt"]
    if change == "pending":
        actual["status"] = "RUNNING"
    elif change == "source":
        actual["source_sha256"] = "b" * 64
    elif change == "weights":
        receipt["parent_checkpoint_sha256"] = "b" * 64
    elif change == "parameters":
        receipt["devices"][1]["learned_parameter_sha256"] = "b" * 64
    elif change == "control":
        receipt["devices"][1]["control_probability_error"] = 0.0031
    elif change == "GPU_input":
        receipt["devices"][0]["roles"][1]["CPU_GPU_inputs_bit_equal"] = False
    else:
        receipt["devices"] = receipt["devices"][:1]
    with pytest.raises(ValueError):
        c.accepted_parameter_hash(actual)
