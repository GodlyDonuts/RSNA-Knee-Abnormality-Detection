"""Prepared matched C100/G100 worker derivation; no allocation or admission."""

import ast
import hashlib
from pathlib import Path

SCIENCE = "39e315c65546aa5205518713bc27349ec63e7538"
ORIGINAL = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
ROOT = Path("/tmp/rsna-predictive-generative-routing-20261020-v1")
ORIGINAL_WORKER_SHA = "78ba933ae78a7d6c3b7199e9f92d0f799c9d77dced13a341968e36e632eb641e"
ROUTING_SHA = "7a88c915209f2e51a293a046402f58d03a588268281d627ea3ac122d8c3add86"
SHARED_INITIAL_SHA = "dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b"
NUMERICAL_SHA = {
    "rsna/competitive/predictive/train.py":
        "066dc5eda8ca8a793548e6fde62c2c6a06480f052d9034aa3fe2e08a5004dc58",
    "rsna/competitive/predictive/density.py":
        "b5f56767dd277af65864983f3a5cd58d89d4cc8153990e9139e6f81b870e64a0",
    "rsna/competitive/predictive/model.py":
        "38c9fc7c2e7e79ea90972f1b14c6e47491aba2f1641496277534e7cd20856f08",
}
ROLES = ("C100", "G100")
EPOCHS = 100
UPDATES = 200
SCHEDULE_UPDATES = 2916
SEED = 20261020
DENSITY_WEIGHT = 100.0
# Four independent final/stopped/resumed checkpoints fit the separate future
# admission's bound. This constant grants no capacity or GPU authority.
CHECKPOINT_CAP = 6 * 2**30

ROUTING_ADAPTER = '''
# isort: split
from rsna.competitive.predictive import train as training_engine
from rsna.competitive.predictive.density import two_mask_loss as original_two_mask_loss
from rsna.competitive.predictive.generative_routing import generative_two_mask_loss


def routed_two_mask_loss(
    evidences, context_logits, targets, label_mask, observations,
    descriptor_width=64, density_weight=0.1,
):
    # train.objective supplies both trailing arguments positionally. The new
    # primitive deliberately requires an explicit keyword density coefficient.
    assert density_weight == 100.0
    return generative_two_mask_loss(
        evidences, context_logits, targets, label_mask, observations,
        descriptor_width, density_weight=density_weight,
    )


assert training_engine.two_mask_loss is original_two_mask_loss
if role == "G100":
    training_engine.two_mask_loss = routed_two_mask_loss
'''

DIGEST_FUNCTION = '''
def state_digests(model):
    """Detached canonical bytes; no RNG or numerical state is changed."""
    values = {name: hashlib.sha256() for name in (
        "model", "shared", "encoder", "head", "encoder_parameters", "head_parameters",
    )}
    parameters = dict(model.named_parameters())
    shared = {"encoder", "project", "position", "attention", "norm", "prior"}
    for name, tensor in sorted(model.state_dict().items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            sort_keys=True, separators=(",", ":")).encode()
        body = tensor.detach().cpu().contiguous().numpy().tobytes()
        groups = ["model", "encoder" if name.startswith("encoder.") else "head"]
        if name in parameters:
            groups.append("encoder_parameters" if name.startswith("encoder.")
                          else "head_parameters")
        if name.split(".")[0] in shared:
            groups.append("shared")
        for group in groups:
            values[group].update(header)
            values[group].update(body)
    return {name: value.hexdigest() for name, value in values.items()}
'''


def sha(body):
    return hashlib.sha256(body).hexdigest()


def replacements():
    """Every modification to the pinned original wrapper, in application order."""
    hashes = {**NUMERICAL_SHA,
              "rsna/competitive/predictive/generative_routing.py": ROUTING_SHA}
    contract = f'''
expected_scientific_hashes = {hashes!r}
for name, expected in expected_scientific_hashes.items():
    assert binding["code_sha256"][name] == expected
'''
    initial = f'''
{DIGEST_FUNCTION}
initial_digests = state_digests(model)
assert seed == {SEED}
assert initial_digests["shared"] == {SHARED_INITIAL_SHA!r}
assert "matched_worker" not in binding
matched_worker = {{
    "role": role, "arm": "C", "seed": seed,
    "initial_model_sha256": initial_digests["model"],
    "initial_shared_tensor_sha256": initial_digests["shared"],
    "initial_encoder_sha256": initial_digests["encoder"],
    "initial_head_sha256": initial_digests["head"],
    "initial_encoder_parameters_sha256": initial_digests["encoder_parameters"],
    "initial_head_parameters_sha256": initial_digests["head_parameters"],
    "original_worker_sha256": {ORIGINAL_WORKER_SHA!r},
    "train_sha256": {NUMERICAL_SHA['rsna/competitive/predictive/train.py']!r},
    "density_sha256": {NUMERICAL_SHA['rsna/competitive/predictive/density.py']!r},
    "routing_sha256": {ROUTING_SHA!r},
    "exam_evidence_gradient": role == "C100",
}}
binding = {{**binding, "matched_worker": matched_worker}}
'''
    return (
        (f'D = Path("{ORIGINAL}")', f'D = Path("{ROOT}")'),
        ('arm, phase = sys.argv[1:]\n'
         'assert arm in ("B", "C", "C0") and phase in ("full", "stop1", "resume")',
         'role, phase = sys.argv[1:]\n'
         'assert role in ("C100", "G100") and phase in ("full", "stop1", "resume")\n'
         'arm = "C"\n' + contract),
        ('from rsna.competitive.predictive.train import Recipe, fit, objective\n',
         'from rsna.competitive.predictive.train import Recipe, fit, objective\n' + ROUTING_ADAPTER),
        ('gc.collect()\nfrom rsna.competitive.predictive.context_cache import load',
         'gc.collect()\n' + initial
         + '\nfrom rsna.competitive.predictive.context_cache import load'),
        ('assert 0 <= index_number < 16 and 0 <= epoch < 12',
         'assert 0 <= index_number < 16 and 0 <= epoch < 100'),
        ('    epochs=12,', '    epochs=100,'),
        ('    density_weight=0 if arm == "C0" else 0.1,', '    density_weight=100.0,'),
        ('root = D / "engineering" / arm', 'root = D / "engineering" / role'),
        ('on_update=lambda r: print(json.dumps({"arm": arm, "phase": phase, **r}), flush=True),',
         'on_update=lambda r: print(json.dumps({"role": role, "arm": arm, '
         '"phase": phase, **r}), flush=True),'),
        ('out = {\n    "arm": arm,',
         'final_digests = state_digests(model)\n'
         'assert state["step"] == (1 if phase == "stop1" else 200)\n'
         'out = {\n    **matched_worker,\n'
         '    "final_encoder_sha256": final_digests["encoder"],\n'
         '    "final_head_sha256": final_digests["head"],\n'
         '    "final_encoder_parameters_sha256": final_digests["encoder_parameters"],\n'
         '    "final_head_parameters_sha256": final_digests["head_parameters"],\n'
         '    "encoder_parameters_changed": initial_digests["encoder_parameters"] '
         '!= final_digests["encoder_parameters"],\n'
         '    "head_parameters_changed": initial_digests["head_parameters"] '
         '!= final_digests["head_parameters"],\n'
         '    "encoder_state_changed": initial_digests["encoder"] != final_digests["encoder"],\n'
         '    "head_state_changed": initial_digests["head"] != final_digests["head"],\n'
         '    "training_epochs": 100,\n    "planned_updates": 200,\n'
         '    "arm": arm,'),
        ('< 12 * 2**30', '< 6 * 2**30'),
    )


def build(original):
    """Return one fresh-process worker; the caller still owns all admission."""
    assert sha(original.encode()) == ORIGINAL_WORKER_SHA, "Exact original worker required"
    result = original
    for before, after in replacements():
        assert result.count(before) == 1, "Worker derivation anchor is not unique"
        result = result.replace(before, after, 1)
    ast.parse(result)
    return result
