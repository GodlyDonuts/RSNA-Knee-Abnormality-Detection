"""Actual read-only fitting-input timing, no model, labels, cache writes or GPUs."""

import hashlib
import json
import os
import resource
import sys
import time
import types
from pathlib import Path

SOURCE = Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code")
RAW = Path("/tmp/rsna-predictive-raw16-cache-20261020-v1")
INDEX_SHA = "e3585fb3cafab3e0ebbca0fb03953e38bd879376956a7a142cc5b4781784008d"
ORIGINAL_SHA = "70413ce4cfbc10f077a140da3dfe2b41990ae72d45600836eb135881b2394893"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def equal_contexts(left, right, torch):
    assert len(left) == len(right) == 2
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys() == {"images", "token_metadata", "region_metadata",
                                      "targets", "observed", "supports"}
        for key in a:
            if key == "supports":
                assert a[key].region_count == b[key].region_count
                for field in ("indptr", "regions", "log_prior"):
                    x, y = getattr(a[key], field), getattr(b[key], field)
                    assert x.dtype == y.dtype and torch.equal(x, y)
            else:
                assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])


def main(v):
    assert os.uname().nodename == "evc33" and not os.environ.get("SLURM_JOB_GPUS")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["SLURM_CPUS_PER_TASK"] == "1"
    assert sys.dont_write_bytecode
    for name, expected in v["scientific_modules"].items():
        assert sha((SOURCE / name).read_bytes()) == expected
    assert sha(v["original_source"].encode()) == ORIGINAL_SHA
    assert sha(v["fast_source"].encode()) == v["fast_sha256"]
    sys.path.insert(0, str(SOURCE))
    import torch

    import rsna.competitive.predictive as package
    from rsna.competitive.predictive import cache, cases

    modules = {}
    for short, source in (("visible_contrast", v["original_source"]),
                          ("visible_contrast_fast", v["fast_source"])):
        name = "rsna.competitive.predictive." + short
        assert name not in sys.modules
        module = types.ModuleType(name)
        sys.modules[name] = module
        setattr(package, short, module)
        exec(compile(source, name, "exec"), module.__dict__)
        modules[short] = module
    original, fast = modules.values()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    accepted = json.loads((RAW / "actual.json").read_bytes())
    body = (RAW / "cache-index.private.json").read_bytes()
    assert accepted["CPU_job"] == "978232" and accepted["fitting_studies"] == 16
    assert sha(body) == INDEX_SHA == accepted["private_index_sha256"]
    index = json.loads(body)
    assert len(index) == 16
    started, observations = time.monotonic(), []
    rng = torch.get_rng_state().clone()
    untouched = (cases.assemble, cases.masked_study, original.assemble, original.acquisition_scales)
    for ordinal, row in enumerate(index[:12]):
        groups, _ = cache.load(Path(row["path"]), row["proof"])
        values, timings, scales = {}, {}, {}
        order = ("original", "fast") if ordinal % 2 == 0 else ("fast", "original")
        for mode in order:
            t = time.monotonic()
            values[mode], scales[mode] = (original if mode == "original" else fast).assemble(
                groups, side=320, context_cap=24, classes=12)
            timings[mode] = time.monotonic() - t
        equal_contexts(values["original"], values["fast"], torch)
        assert scales["original"] == scales["fast"]
        assert cache.signature(Path(row["path"])) == row["proof"]["stat"]
        observations.append({"original_seconds": timings["original"], "fast_seconds": timings["fast"],
            "original_first": order[0] == "original", "frames": sum(len(g.frames) for g in groups),
            "windows": sum(len(c["images"]) for c in values["original"])})
        del groups, values, scales
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert untouched == (cases.assemble, cases.masked_study, original.assemble, original.acquisition_scales)
    old_sum = sum(r["original_seconds"] for r in observations)
    new_sum = sum(r["fast_seconds"] for r in observations)
    result = {"status": "actual_twelve_fitting_MRI_equivalent_contrast_geometry_timing",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "original_candidate_sha256": ORIGINAL_SHA, "fast_source_sha256": v["fast_sha256"],
        "scientific_modules": v["scientific_modules"], "original_index_sha256": INDEX_SHA,
        "fitting_studies": 12, "excluded_engineering_studies": 4,
        "original_frames": sum(r["frames"] for r in observations),
        "windows_per_variant": sum(r["windows"] for r in observations),
        "all_context_and_support_fields_bitwise_equal": True, "all_visible_scales_exactly_equal": True,
        "global_RNG_and_original_functions_unchanged": True,
        "original_render_seconds": old_sum, "fast_render_seconds": new_sum,
        "paired_render_sum_ratio": old_sum / new_sum,
        "per_study_speedup_range": [min(r["original_seconds"] / r["fast_seconds"] for r in observations),
                                    max(r["original_seconds"] / r["fast_seconds"] for r in observations)],
        "alternating_first_order_counts": {"original": 6, "fast": 6},
        "elapsed_seconds_including_load_and_comparison": time.monotonic() - started,
        "peak_host_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "remote_files_written": 0, "source_downloads": 0,
        "labels_or_model_or_evaluation_images_read": False,
        "Kaggle_runtime_or_clinical_acceptance": False,
        "GPU_pilot_and_primary_numerical_sources_changed": False}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
