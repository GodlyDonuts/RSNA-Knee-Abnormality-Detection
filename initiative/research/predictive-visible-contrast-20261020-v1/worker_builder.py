"""Explicit matched contrast learning worker; preparation grants no GPU dispatch."""

import ast
import hashlib
from pathlib import Path

ORIGINAL = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
ROOT = Path("/tmp/rsna-predictive-contrast-learning-20261020-v1")
ORIGINAL_WORKER_SHA = "78ba933ae78a7d6c3b7199e9f92d0f799c9d77dced13a341968e36e632eb641e"
CANDIDATE_SHA = "70413ce4cfbc10f077a140da3dfe2b41990ae72d45600836eb135881b2394893"
NUMERICAL_SHA = {
    "rsna/competitive/predictive/train.py": "066dc5eda8ca8a793548e6fde62c2c6a06480f052d9034aa3fe2e08a5004dc58",
    "rsna/competitive/predictive/model.py": "38c9fc7c2e7e79ea90972f1b14c6e47491aba2f1641496277534e7cd20856f08",
    "rsna/competitive/predictive/density.py": "b5f56767dd277af65864983f3a5cd58d89d4cc8153990e9139e6f81b870e64a0",
    "rsna/competitive/predictive/context_cache.py": "bf9d59b26bd8a28f4533648d5b10d6c77bd5caf8d2f69de0b77968ce4d4ea3b5",
}
ROLES = ("control", "normalized")
UPDATES = 200
CHECKPOINT_CAP = 6 * 2**30

DIGEST_FUNCTION = '''
def model_digests(model):
    values = {name: hashlib.sha256() for name in ("model", "encoder_parameters", "head_parameters")}
    parameters = dict(model.named_parameters())
    for name, tensor in sorted(model.state_dict().items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode()
        body = tensor.detach().cpu().contiguous().numpy().tobytes()
        keys = ["model"]
        if name in parameters:
            keys.append("encoder_parameters" if name.startswith("encoder.") else "head_parameters")
        for key in keys:
            values[key].update(header)
            values[key].update(body)
    return {key: value.hexdigest() for key, value in values.items()}
'''


def recipe():
    return {"arm": "C", "epochs": 100, "effective_batch": 8, "encoder_lr": 3e-5,
            "head_lr": 3e-5, "weight_decay": .02, "warmup_fraction": .05,
            "end_lr_fraction": .1, "density_weight": .1, "head_only_updates": 3,
            "clip": 1., "seed": 20261020, "schedule_updates": 2916}


def build(original):
    assert hashlib.sha256(original.encode()).hexdigest() == ORIGINAL_WORKER_SHA
    contract = f'''
assert binding["branch"] == "matched_visible_only_contrast"
assert binding["candidate_sha256"] == {CANDIDATE_SHA!r}
assert binding["metadata_width"] == 10 and binding["recipe"] == {recipe()!r}
for name, expected in {NUMERICAL_SHA!r}.items():
    assert binding["code_sha256"][name] == expected
'''
    initial = f'''
{DIGEST_FUNCTION}
assert model.position[0].in_features == 10
initial_digests = model_digests(model)
assert "matched_worker" not in binding
matched_worker = {{"role": role, "arm": "C", "metadata_width": 10,
                  "initial_digests": initial_digests,
                  "original_worker_sha256": {ORIGINAL_WORKER_SHA!r},
                  "candidate_sha256": {CANDIDATE_SHA!r},
                  "initial_state_is_not_the_eight_coordinate_primary_graph": True}}
binding = {{**binding, "matched_worker": matched_worker}}
'''
    substitutions = (
        (f'D = Path("{ORIGINAL}")', f'D = Path("{ROOT}")'),
        ('arm, phase = sys.argv[1:]\nassert arm in ("B", "C", "C0") and phase in ("full", "stop1", "resume")',
         'role, phase = sys.argv[1:]\nassert role in ("control", "normalized") and phase in ("full", "stop1", "resume")\narm = "C"\n' + contract),
        ('model = cls(FixedEncoder(parent.backbone), parent.backbone.num_features, hidden=128).cuda()',
         'model = cls(FixedEncoder(parent.backbone), parent.backbone.num_features, hidden=128, metadata_width=10).cuda()'),
        ('gc.collect()\nfrom rsna.competitive.predictive.context_cache import load',
         'gc.collect()\n' + initial + '\nfrom rsna.competitive.predictive.context_cache import load'),
        ('assert 0 <= index_number < 16 and 0 <= epoch < 12', 'assert 0 <= index_number < 16 and 0 <= epoch < 100'),
        ('row = index[index_number]', 'row = index[index_number]["variants"][role]\n'
         '        assert row["provenance"]["role"] == role\n'
         f'        assert row["provenance"]["candidate_sha256"] == {CANDIDATE_SHA!r}\n'
         '        assert row["provenance"]["metadata_width"] == 10'),
        ('    epochs=12,', '    epochs=100,'),
        ('    density_weight=0 if arm == "C0" else 0.1,', '    density_weight=0.1,'),
        ('root = D / "engineering" / arm', 'root = D / "engineering" / role'),
        ('on_update=lambda r: print(json.dumps({"arm": arm, "phase": phase, **r}), flush=True),',
         'on_update=lambda r: print(json.dumps({"role": role, "arm": arm, "phase": phase, **r}), flush=True),'),
        ('out = {\n    "arm": arm,',
         'assert state["step"] == (1 if phase == "stop1" else 200)\n'
         'final_digests = model_digests(model)\n'
         'out = {\n    **matched_worker,\n    "final_digests": final_digests,\n'
         '    "encoder_parameters_changed": initial_digests["encoder_parameters"] != final_digests["encoder_parameters"],\n'
         '    "head_parameters_changed": initial_digests["head_parameters"] != final_digests["head_parameters"],\n'
         '    "training_epochs": 100,\n    "planned_updates": 200,\n    "arm": arm,'),
        ('< 12 * 2**30', '< 6 * 2**30'),
    )
    output = original
    for old, new in substitutions:
        assert output.count(old) == 1, "Exact frozen worker anchor must be unique"
        output = output.replace(old, new, 1)
    ast.parse(output)
    return output
