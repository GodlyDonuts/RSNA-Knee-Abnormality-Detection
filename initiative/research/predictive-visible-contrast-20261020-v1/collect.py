"""Read-only contrast measurement on the original twelve fitting contexts.

No new normalization is applied. Non-fill values include interpolation borders,
so their quantiles are a rendered-input proxy, not an anatomical tissue mask.
Only aggregate statistics leave the owning node; no pixels, IDs or labels do.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
ACCEPTED_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
BINDING_SHA = "43304b78b34b407f55088d4f8e610af8d1cbb112a9960f75311689a0078a75cc"
INDEX_SHA = "d124b6d96c76e827731dcc0f9307d93cb937ea17f6d1877c142c2bf61f12b5b0"
VERSION = "exact_FP32_two_complementary_raw_contexts_complete_targets_priors_v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def signature(path):
    st = path.stat()
    assert path.is_file() and not path.is_symlink() and st.st_uid == os.getuid()
    assert not st.st_mode & 0o077
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def image_statistics(images):
    assert images.dtype == np.float32 and images.ndim == 4 and images.shape[1:] == (3, 320, 320)
    assert 0 < len(images) <= 24 and np.isfinite(images).all()
    assert images.min() >= 0 and images.max() <= 1
    flat = images.reshape(-1)
    fill = flat == np.float32(.5)
    nonfill = flat[~fill]
    assert len(nonfill), "All-fill context is not meaningful image evidence"
    q = np.quantile(nonfill, [.01, .5, .99])
    return {
        "exact_half_fill_fraction": float(fill.mean()),
        "nonfill_q01": float(q[0]), "nonfill_median": float(q[1]),
        "nonfill_q99": float(q[2]), "nonfill_q99_minus_q01": float(q[2] - q[0]),
        "nonfill_std": float(nonfill.std(dtype=np.float64)),
        "near_zero_fraction": float((flat <= .01).mean()),
        "near_one_fraction": float((flat >= .99).mean()),
        "rendered_windows": len(images),
    }


def main(v):
    start = time.monotonic()
    assert os.uname().nodename == "evc27" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ["SLURM_CPUS_PER_TASK"] == "1"
    assert sys.dont_write_bytecode
    assert sha((ROOT / "GPU-accepted.json").read_bytes()) == ACCEPTED_SHA
    binding_raw = (ROOT / "binding.private.json").read_bytes()
    assert sha(binding_raw) == BINDING_SHA
    binding = json.loads(binding_raw)
    index_raw = Path(binding["context_index_private_path"]).read_bytes()
    assert sha(index_raw) == INDEX_SHA == binding["context_index_sha256"]
    index = json.loads(index_raw)
    assert len(index) == 16
    stats, dc, windows, counts = [], [], 0, []
    for row in index[:12]:
        path = Path(row["path"])
        assert row["proof"]["version"] == VERSION and signature(path) == row["proof"]["stat"]
        with np.load(path, allow_pickle=False) as arrays:
            assert json.loads(arrays["manifest"].tobytes()) == {
                "version": VERSION, "source": row["provenance"]}
            per_case = []
            for mask in (0, 1):
                value = image_statistics(arrays[f"c{mask}_images"])
                windows += value.pop("rendered_windows")
                per_case.append(value)
            # Equal context weight, then equal study weight. No labels are read.
            stats.append({k: float(np.mean([x[k] for x in per_case])) for k in per_case[0]})
            targets = arrays["c0_targets"]
            assert targets.dtype == np.float32 and targets.ndim == 2 and targets.shape[1] == 64
            assert np.isfinite(targets).all() and targets[:, 0].min() >= 0
            assert targets[:, 0].max() <= 1.000001
            dc.append(float(np.median(targets[:, 0])))
            counts.append(len(targets))
        assert signature(path) == row["proof"]["stat"], "Original context changed during read"
    result = {
        "status": "actual_twelve_fitting_case_rendered_contrast_readonly",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "collector_sha256": v["collector_sha256"], "original_context_index_sha256": INDEX_SHA,
        "original_engineering_receipt_sha256": ACCEPTED_SHA,
        "fitting_studies": 12, "excluded_engineering_studies": 4,
        "rendered_windows_including_both_masks": windows, "canonical_regions": sum(counts),
        "equal_context_then_study_summary": {k: summary([x[k] for x in stats]) for k in stats[0]},
        "study_median_descriptor_DC_summary": summary(dc),
        "study_count_mean_nonfill_span_below_point1": sum(x["nonfill_q99_minus_q01"] < .1 for x in stats),
        "clinical_labels_models_or_evaluation_inputs_read": False,
        "pixels_IDs_or_per_study_records_exported": False,
        "new_remote_files_or_changed_cache_bytes": 0, "optimizer_updates": 0,
        "nonfill_is_an_interpolation_contaminated_proxy_not_a_tissue_mask": True,
        "global_or_clinical_accuracy_claim": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
