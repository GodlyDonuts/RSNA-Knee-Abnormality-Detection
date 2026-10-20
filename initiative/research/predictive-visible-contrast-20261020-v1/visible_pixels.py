"""Exact unmasked contribution measurement in the original twelve MRI contexts.

Reuses original raw cache and renderer. A diagnostic triplet hook returns the
unchanged original tensor and measures only pixels with zero hidden-source
interpolation weight. All source arrays and case-level statistics stay in RAM.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

RAW = Path("/tmp/rsna-predictive-raw16-cache-20261020-v1")
INDEX_SHA = "e3585fb3cafab3e0ebbca0fb03953e38bd879376956a7a142cc5b4781784008d"
SOURCE = Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pure_visible_values(rendered, hidden):
    assert rendered.dtype == torch.float32 and hidden.dtype == torch.bool
    assert rendered.ndim == hidden.ndim == 4 and rendered.shape[:2] == hidden.shape[:2]
    assert not rendered.requires_grad and rendered.device.type == hidden.device.type == "cpu"
    hidden_weight = F.interpolate(hidden.float(), size=rendered.shape[-2:],
                                  mode="bilinear", align_corners=False)
    visible = hidden_weight == 0
    values = rendered[visible].numpy().copy()
    assert values.size and np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    return values, int(visible.sum()), visible.numel()


def summary(values):
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def main(v):
    assert os.uname().nodename == "evc33" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ["SLURM_CPUS_PER_TASK"] == "1"
    assert sys.dont_write_bytecode
    for name, expected in v["scientific_modules"].items():
        assert sha((SOURCE / name).read_bytes()) == expected
    sys.path.insert(0, str(SOURCE))
    from rsna.competitive.predictive import cache, cases

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    accepted = json.loads((RAW / "actual.json").read_bytes())
    raw = (RAW / "cache-index.private.json").read_bytes()
    assert accepted["CPU_job"] == "978232" and accepted["fitting_studies"] == 16
    assert sha(raw) == INDEX_SHA == accepted["private_index_sha256"]
    index = json.loads(raw)
    assert len(index) == 16
    started, observations, frames, windows = time.monotonic(), [], 0, 0
    original_triplets = cases.triplets
    for row in index[:12]:
        groups, _ = cache.load(Path(row["path"]), row["proof"])
        chunks = []

        def measured_triplets(study, centers, neighbors, *, side=320):
            rendered = original_triplets(study, centers, neighbors, side=side)
            chunks.append(pure_visible_values(rendered, study.raw_hidden[neighbors]))
            return rendered

        cases.triplets = measured_triplets
        try:
            contexts = cases.assemble(groups, side=320, context_cap=24, classes=12)
        finally:
            cases.triplets = original_triplets
        assert len(chunks) % 2 == 0
        per_mask = []
        half = len(chunks) // 2
        for part in (chunks[:half], chunks[half:]):
            values = np.concatenate([x[0] for x in part])
            q = np.quantile(values, [.01, .5, .99])
            per_mask.append({"q01": float(q[0]), "median": float(q[1]), "q99": float(q[2]),
                "q99_minus_q01": float(q[2] - q[0]), "std": float(values.std(dtype=np.float64)),
                "pure_visible_fraction": sum(x[1] for x in part) / sum(x[2] for x in part)})
        observations.append({k: float(np.mean([x[k] for x in per_mask])) for k in per_mask[0]})
        frames += sum(len(g.frames) for g in groups)
        windows += sum(len(c["images"]) for c in contexts)
        assert cache.signature(Path(row["path"])) == row["proof"]["stat"]
        chunks.clear()
        del groups, contexts, values
    result = {
        "status": "actual_twelve_fitting_cases_exact_visible_MRI_contrast_readonly",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "collector_sha256": v["collector_sha256"], "original_raw_index_sha256": INDEX_SHA,
        "scientific_modules": v["scientific_modules"], "fitting_studies": 12,
        "excluded_engineering_studies": 4, "original_frames": frames,
        "rendered_windows_including_both_masks": windows,
        "equal_context_then_study_summary": {k: summary([x[k] for x in observations]) for k in observations[0]},
        "study_count_mean_visible_span_below_point1": sum(x["q99_minus_q01"] < .1 for x in observations),
        "study_count_mean_visible_span_below_point05": sum(x["q99_minus_q01"] < .05 for x in observations),
        "mask_test": "bilinear_interpolated_hidden_indicator_equals_exact_zero",
        "visible_pixels_can_include_anatomical_background": True,
        "clinical_labels_models_or_evaluation_inputs_read": False,
        "pixels_IDs_or_per_study_records_exported": False,
        "new_remote_files_or_changed_cache_bytes": 0, "optimizer_updates": 0,
        "global_or_clinical_accuracy_claim": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    assert not torch.cuda.is_initialized()
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
