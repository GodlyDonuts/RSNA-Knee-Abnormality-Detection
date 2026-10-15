"""Synthetic-only geometry checks; no DICOM files, caches, models, or training.

Executes the unchanged, AST-extracted native order_slices and _indices functions
with synthetic SimpleNamespace slices. Geometry/render checks are NumPy algebra,
not a test of the unavailable PyTorch/pydicom runtime.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).parent
COMMIT = "c83ffa8b58a3bb28dc63e7f10f4805faab0a8893"


def extract(path, name):
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


order_slices = extract(ROOT / "rsna/dicom.py", "order_slices")
indices = extract(ROOT / "rsna/competitive/data.py", "_indices")


def fake_slice(k, instance, u=(1, 0, 0), v=(0, 1, 0), origin=None):
    n = np.cross(u, v)
    return SimpleNamespace(sop_uid=f"synthetic-{k}", instance=instance,
                           position=tuple(np.asarray(origin if origin is not None else k * 2 * n, dtype=float)),
                           orientation=tuple(u) + tuple(v), spacing=(2., 1.),
                           pixels=np.arange(12).reshape(3, 4) + 100 * k)


out = {"source_commit": COMMIT, "only_synthetic": True, "checks": {}}
scrambled = [fake_slice(k, instance) for k, instance in [(2, 1), (0, 3), (1, 2)]]
ordered, method, warnings = order_slices(scrambled)
assert [s.sop_uid for s in ordered] == ["synthetic-0", "synthetic-1", "synthetic-2"]
out["checks"]["geometry_overrides_instance"] = {"method": method, "instance_order": [s.instance for s in ordered], "passed": True}

fallback = [fake_slice(k, instance) for k, instance in [(2, 1), (0, 3), (1, 2)]]
fallback[0].position = ()
ordered_fallback, method_fallback, warnings_fallback = order_slices(fallback)
assert method_fallback == "instance_number" and [s.instance for s in ordered_fallback] == [1, 2, 3]
out["checks"]["missing_geometry_explicit_fallback"] = {"method": method_fallback, "warnings": warnings_fallback, "passed": True}

# A lossless encoding change: reverse image columns, negate first IOP vector,
# shift each IPP to the opposite column, then sort by the resulting normal.
# Each numeric voxel still refers to exactly the same physical location.
original = [fake_slice(k, k + 1) for k in range(3)]
reencoded = []
for s in original:
    u, v = np.array(s.orientation[:3]), np.array(s.orientation[3:])
    t = SimpleNamespace(**vars(s))
    t.orientation = tuple(-u) + tuple(v)
    t.position = tuple(np.array(s.position) + (s.pixels.shape[1] - 1) * s.spacing[1] * u)
    t.pixels = s.pixels[:, ::-1]
    reencoded.append(t)
reordered, method_reencoded, _ = order_slices(reencoded)
assert [s.sop_uid for s in reordered] == ["synthetic-2", "synthetic-1", "synthetic-0"]
for a, b in zip(original, reencoded):
    for j in range(3):
        for i in range(4):
            p = np.array(a.position) + i*a.spacing[1]*np.array(a.orientation[:3]) + j*a.spacing[0]*np.array(a.orientation[3:])
            ii = 3-i
            q = np.array(b.position) + ii*b.spacing[1]*np.array(b.orientation[:3]) + j*b.spacing[0]*np.array(b.orientation[3:])
            np.testing.assert_allclose(p, q)
            assert a.pixels[j, i] == b.pixels[j, ii]
old_n = np.cross(original[0].orientation[:3], original[0].orientation[3:])
new_n = np.cross(reordered[0].orientation[:3], reordered[0].orientation[3:])
old_p = np.array([s.position for s in original]) @ old_n
new_p = np.array([s.position for s in reordered]) @ new_n
old_q = (old_p - (old_p[0] + old_p[-1])/2)/100
new_q = (new_p - (new_p[0] + new_p[-1])/2)/100
old_by_id = dict(zip([s.sop_uid for s in original], old_q))
new_by_id = dict(zip([s.sop_uid for s in reordered], new_q))
assert all(np.isclose(old_by_id[k], -new_by_id[k]) for k in old_by_id)
out["checks"]["physical_identity_is_not_tensor_canonicalization"] = {
    "same_physical_voxels": True, "new_order": [s.sop_uid for s in reordered],
    "old_normal": old_n.tolist(), "new_normal": new_n.tolist(),
    "old_position_by_id": old_by_id, "new_position_by_id": new_by_id, "passed": True}

# Current native sixteen-slice selection does not introduce a reversal mismatch.
for count in range(1, 1001):
    selected = indices(count, 16, False)
    np.testing.assert_array_equal(selected, count - 1 - selected[::-1])
out["checks"]["native_16_slice_reversal_symmetric_selection"] = {"counts_checked": [1, 1000], "passed": True}

# Physical aspect uses row spacing for height, column spacing for width.
h, w, dr, dc, side = 16, 32, 2., 1., 384
scale = side / max(h*dr, w*dc)
rendered = [round(h*dr*scale), round(w*dc*scale)]
assert rendered == [384, 384]
out["checks"]["anisotropic_pixel_physical_aspect"] = {"input_shape": [h, w], "spacing": [dr, dc], "output_content_shape": rendered, "passed": True}

# Conditional reconstruction of PRIOR audited public mirror semantics, not a
# re-execution of current Kaggle source: full slots and central-slot dispatch.
lengths = [26, 22, 18, 12, 18]
slot = np.repeat(np.arange(5), lengths)
plane = np.array(["sag", "sag", "cor", "cor", "ax"])[slot]
rule = np.where(plane == "sag", "channel_reverse", "width_reverse")
cross_slot, cross_plane, mixed_rule = [], [], []
for c in range(1, len(slot)-1):
    ix = np.arange(c-1, c+2)
    if len(set(slot[ix])) > 1: cross_slot.append(c)
    if len(set(plane[ix])) > 1: cross_plane.append(c)
    if len(set(rule[ix])) > 1: mixed_rule.append(c)
assert len(cross_slot) == 8 and len(cross_plane) == 4 and len(mixed_rule) == 2
out["checks"]["prior_public_full_slot_window_boundary_counts"] = {
    "source_status": "conditional reconstruction from prior verified audit, exact notebook not reread",
    "total_windows": 94, "cross_slot_centers_0based": cross_slot,
    "cross_plane_centers_0based": cross_plane, "mixed_mirror_rule_centers_0based": mixed_rule, "passed": True}

# Minimal label-specific gated-attention-style pooling is invariant to jointly
# permuting images' embeddings and attached positions, but not to moving only
# the embeddings. This proves a structural claim, not actual model predictions.
e = np.array([[1., .5], [-.2, .8], [.1, -.7]])
p = np.array([[-.02], [0.], [.02]])
def pooled(e, p):
    full = e + p @ np.array([[5., -4.]])
    scores = np.tanh(full) @ np.array([.8, -.6])
    weights = np.exp(scores - scores.max()); weights /= weights.sum()
    return weights @ full
np.testing.assert_allclose(pooled(e, p), pooled(e[::-1], p[::-1]))
assert not np.allclose(pooled(e, p), pooled(e[::-1], p))
out["checks"]["joint_permutation_vs_reassigned_position"] = {"joint_permutation_preserved": True, "images_only_reversal_changed": True, "passed": True}

out["limitations"] = ["No full PyTorch/pydicom execution; neither dependency is installed", "No real DICOMs, clinical metadata, caches, prediction rows, weights, or training", "No assertion of actual-cohort prevalence or accuracy effect"]
out["source_file_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"rsna/dicom.py", ROOT/"rsna/competitive/data.py"]}
destination = ROOT/"synthetic_geometry_results.json"
destination.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
