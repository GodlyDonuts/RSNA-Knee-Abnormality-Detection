"""Fit a private engineering-only scale from the original twelve training cases.

Reads descriptor members, masks and manifests only. Never loads image members,
clinical labels, models, the four engineering evaluation cases or full-fit data.
"""

import hashlib
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ORIGINAL = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
DESTINATION = Path("/tmp/rsna-predictive-descriptor-scale-20261020-v1")
ACCEPTED_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
ORIGINAL_BINDING_SHA = "43304b78b34b407f55088d4f8e610af8d1cbb112a9960f75311689a0078a75cc"
CONTEXT_VERSION = "exact_FP32_two_complementary_raw_contexts_complete_targets_priors_v1"
CONTEXT_KEYS = {"manifest"} | {
    f"c{i}_{k}" for i in (0, 1) for k in (
        "images", "token_metadata", "region_metadata", "targets", "observed",
        "indptr", "regions", "log_prior", "region_count",
    )
}
CAP = 2**20
# Existing portable/C0 liabilities + complete C100 pilot and prepared full route.
OTHER_NODE = (
    205676085248 + 96 * 2**30 + 12 * 2**30 + 64 * 2**20
    + 6 * 2**30 + 4 * 2**20 + 12 * 2**30 + 16 * 2**20
)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def signature(path):
    s = path.stat()
    if not path.is_file() or path.is_symlink() or s.st_uid != os.getuid():
        raise ValueError("Original owning-node regular file required")
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read_descriptors(row):
    """The explicit member allowlist prevents image and label array reads."""
    path = Path(row["path"])
    if row["proof"]["version"] != CONTEXT_VERSION or signature(path) != row["proof"]["stat"]:
        raise ValueError("Previously admitted immutable cache changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != CONTEXT_KEYS:
            raise ValueError("Only exact original context members may be admitted")
        manifest = json.loads(archive["manifest"].tobytes())
        if manifest != {"version": CONTEXT_VERSION, "source": row["provenance"]}:
            raise ValueError("Original raw/source/renderer binding differs")
        a, b = archive["c0_targets"], archive["c1_targets"]
        o0, o1 = archive["c0_observed"], archive["c1_observed"]
        if (
            a.dtype != np.float32 or b.dtype != np.float32
            or a.ndim != 2 or a.shape[1] != 64 or a.shape[0] == 0
            or a.shape[0] % 16 or not np.isfinite(a).all()
            or not np.array_equal(a, b)
            or o0.dtype != np.bool_ or o1.dtype != np.bool_
            or o0.shape != (len(a),) or o1.shape != o0.shape
            or not np.array_equal(o0, ~o1)
            or int(archive["c0_region_count"]) != len(a)
            or int(archive["c1_region_count"]) != len(a)
        ):
            raise ValueError("Complete unique target roster / complementary masks differ")
        result = torch.from_numpy(a.copy())
    if signature(path) != row["proof"]["stat"]:
        raise ValueError("Context changed while reading descriptor members")
    return result


def main(v):
    start = time.monotonic()
    if (
        os.uname().nodename != "evc27" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("SLURM_JOB_GPUS") or os.environ["SLURM_CPUS_PER_TASK"] != "1"
        or not sys.dont_write_bytecode
    ):
        raise ValueError("Original owning-node one-CPU, zero-GPU admission required")
    os.umask(0o077)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    free = shutil.disk_usage("/tmp").free
    required = OTHER_NODE + 20 * 2**30 + CAP
    if free < required or DESTINATION.exists():
        raise ValueError("Insufficient funded reserve or original scale already exists")
    accepted = (ORIGINAL / "GPU-accepted.json").read_bytes()
    if sha(accepted) != ACCEPTED_SHA:
        raise ValueError("Original twelve/four engineering admission changed")
    old_binding_body = (ORIGINAL / "binding.private.json").read_bytes()
    if sha(old_binding_body) != ORIGINAL_BINDING_SHA:
        raise ValueError("Actual original binding digest differs")
    old = json.loads(old_binding_body)
    if old["source_git"] != "39e315c65546aa5205518713bc27349ec63e7538":
        raise ValueError("Wrong original numerical source")
    # The pinned original input code provides alias-closed canonical region order.
    input_path = ORIGINAL / "code/rsna/competitive/predictive/inputs.py"
    input_sha = sha(input_path.read_bytes())
    if input_sha != old["code_sha256"]["rsna/competitive/predictive/inputs.py"]:
        raise ValueError("Original canonical descriptor source differs")
    index_body = Path(old["context_index_private_path"]).read_bytes()
    if sha(index_body) != old["context_index_sha256"]:
        raise ValueError("Original sixteen-case input index differs")
    index = json.loads(index_body)
    if len(index) != 16:
        raise ValueError("Expected original twelve training plus four evaluation inputs")
    # Bind the exact inherited training partition without opening the label file.
    fitting_roster = canonical({
        "original_engineering_receipt_sha256": ACCEPTED_SHA,
        "original_binding_sha256": sha(old_binding_body),
        "context_index_sha256": sha(index_body),
        "training_ordinals": list(range(12)), "excluded_evaluation_ordinals": [12, 13, 14, 15],
    })
    fit_binding = {
        "split": "fitting", "studies": 12, "fitting_roster_sha256": sha(fitting_roster),
        "source_manifest_sha256": sha(canonical(index[:12])),
        "descriptor_source_sha256": input_sha,
    }
    source = v["normalizer_source"]
    if sha(source.encode()) != v["normalizer_sha256"]:
        raise ValueError("Frozen standalone scale code differs")
    module = types.ModuleType("frozen_descriptor_normalization")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__name__, "exec"), module.__dict__)
    accumulator = module.StudyBalancedMoments(fit_binding)
    region_counts = []
    for i, row in enumerate(index[:12]):
        targets = read_descriptors(row)
        accumulator.add_study(
            i, targets, torch.arange(len(targets), dtype=torch.int64),
            source_receipt_sha256=sha(canonical(row)),
        )
        region_counts.append(len(targets))
    normalizer = accumulator.finalize()
    artifact = normalizer.to_bytes()
    if len(artifact) > CAP // 4:
        raise ValueError("Private scale exceeds its predeclared storage allowance")
    decoded = json.loads(artifact)
    DESTINATION.mkdir(mode=0o700)
    with (DESTINATION / "normalizer.private.json").open("xb") as f:
        f.write(artifact)
    reread = (DESTINATION / "normalizer.private.json").read_bytes()
    restored = module.FrozenDescriptorNormalizer.from_bytes(
        reread, expected_sha256=sha(artifact), expected_binding=fit_binding,
    )
    if restored != normalizer:
        raise ValueError("Private scale serialization or independent readback differs")
    with (DESTINATION / "fitting-binding.private.json").open("xb") as f:
        f.write(canonical({"binding": fit_binding, "roster": json.loads(fitting_roster)}))
    sd = np.sqrt(np.asarray(decoded["variance"], dtype=np.float64))
    scale = np.asarray(decoded["scale"], dtype=np.float64)
    receipt = {
        "status": "actual_twelve_fitting_case_descriptor_scale_engineering_only",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "normalizer_code_sha256": v["normalizer_sha256"],
        "original_engineering_receipt_sha256": ACCEPTED_SHA,
        "original_context_index_sha256": sha(index_body),
        "original_descriptor_source_sha256": input_sha,
        "private_artifact_sha256": sha(artifact), "private_artifact_bytes": len(artifact),
        "private_binding_sha256": sha(canonical(fit_binding)),
        "fitting_studies": 12, "excluded_engineering_studies": 4,
        "canonical_regions": sum(region_counts), "region_count_range": [
            min(region_counts), max(region_counts)],
        "coordinate_SD_range": [float(sd.min()), float(sd.max())],
        "frozen_scale_range": [float(scale.min()), float(scale.max())],
        "floored_coordinates": int((sd < 0.001).sum()),
        "maximum_inverse_scale": float(1 / scale.min()),
        "weighting": "equal_study_then_equal_unique_canonical_region",
        "std_floor_predeclared": 0.001,
        "context_pair_targets_equal_counted_once": True,
        "exact_private_artifact_readback_and_binding_passed": True,
        "clinical_labels_models_images_loaded": False,
        "raw_descriptors_or_normalizer_vectors_exported": False,
        "new_remote_storage_cap_bytes": CAP, "new_shared_bytes": 0, "new_shared_files": 0,
        "node_free_bytes": free, "node_required_bytes_including_reserve": required,
        "optimizer_steps": 0, "GPU_dispatch_or_full_fit_authority": False,
        "scale_does_not_remove_D_vs_exam_descriptor_count_dilution": True,
        "elapsed_seconds": time.monotonic() - start,
    }
    with (DESTINATION / "actual.json").open("xb") as f:
        f.write(canonical(receipt))
    if sum(p.stat().st_size for p in DESTINATION.iterdir()) > CAP:
        raise ValueError("Unexpected private scale growth; preserve originals")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
