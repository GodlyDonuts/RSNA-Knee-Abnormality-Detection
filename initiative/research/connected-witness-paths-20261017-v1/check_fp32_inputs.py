"""Real deployment-class CPU integration; actual CUDA comparison still required."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()):
    from check_online_inputs import Encoder, Fixture

import torch
from online_inputs_fp32 import StreamedFP32PathData

from rsna.competitive.compact_bag import CompactBagClassifier
from rsna.competitive.compact_infer import DeploymentBagClassifier
from rsna.competitive.spatial_data import OnlineSpatialData

with tempfile.TemporaryDirectory() as tmp:
    data = Fixture(Path(tmp))
    torch.manual_seed(13)
    parent = (
        DeploymentBagClassifier(Encoder(), precision="float32", dimension=8)
        .eval()
        .requires_grad_(False)
    )
    saved = copy.deepcopy(parent.state_dict())
    original = OnlineSpatialData(data, parent, {"synthetic_only": True}, seed=20261017)
    expected = original[0]
    providers = {
        s: StreamedFP32PathData(data, parent, {"synthetic_only": True}, sampling=s)
        for s in ("sparse16", "dense")
    }
    # An enclosing lower-precision context must not change the bound FP32 path.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        sparse, dense = providers["sparse16"][0], providers["dense"][0]
    assert torch.equal(sparse["maps"], expected["maps"])
    assert torch.equal(sparse["base_logits"], expected["base_logits"])
    assert torch.equal(dense["base_logits"], expected["base_logits"])
    assert len(dense["maps"]) == 22 and dense["maps"].dtype == torch.float32
    assert all(torch.equal(saved[k], v) for k, v in parent.state_dict().items())
    refusals = 0
    for kind in ("training_class", "float16", "changed_after_admission"):
        try:
            if kind == "training_class":
                StreamedFP32PathData(
                    data,
                    CompactBagClassifier(Encoder(), dimension=8).eval().requires_grad_(False),
                    {},
                    sampling="dense",
                )
            elif kind == "float16":
                StreamedFP32PathData(
                    data,
                    DeploymentBagClassifier(
                        Encoder(), precision="float16", dimension=8
                    ).requires_grad_(False),
                    {},
                    sampling="dense",
                )
            else:
                parent.precision = "float16"
                providers["dense"][0]
        except ValueError:
            refusals += 1
        else:
            raise AssertionError(kind)
    assert refusals == 3
print(
    json.dumps(
        {
            "status": "synthetic_explicit_deployment_FP32_stream_original_maps_teacher_and_state_exact",
            "source_sha256": hashlib.sha256(
                (Path(__file__).parent / "online_inputs_fp32.py").read_bytes()
            ).hexdigest(),
            "wrong_parent_precision_and_post_admission_change_refusals": refusals,
            "enclosing_CPU_BF16_context_does_not_change_FP32_reference": True,
            "actual_MRI_CUDA_encoder_or_training_acceptance": False,
        },
        indent=2,
    )
)
