"""Bounded real-input contrast experiment; source arrays never leave this CPU."""

import hashlib
import json
import os
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

RAW = Path("/tmp/rsna-predictive-raw16-cache-20261020-v1")
SOURCE = Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code")
INDEX_SHA = "e3585fb3cafab3e0ebbca0fb03953e38bd879376956a7a142cc5b4781784008d"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_ram_module(name, source, expected):
    assert sha(source.encode()) == expected and name not in sys.modules
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, "<" + name + ">", "exec"), module.__dict__)
    return module


def assert_targets_equal(a, b):
    for x, y in zip(a, b, strict=True):
        for field in ("targets", "observed"):
            assert torch.equal(x[field], y[field])
        for field in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(x["supports"], field), getattr(y["supports"], field))


def summarize_render(chunks):
    assert len(chunks) % 2 == 0
    half, masks = len(chunks) // 2, []
    for part in (chunks[:half], chunks[half:]):
        values = np.concatenate([x[0] for x in part])
        q = np.quantile(values, [.01, .5, .99])
        masks.append({"q99_minus_q01": float(q[2] - q[0]), "median": float(q[1]),
                      "zero_fraction": float(np.mean(values == 0)),
                      "one_fraction": float(np.mean(values == 1)),
                      "pure_visible_fraction": sum(x[1] for x in part) / sum(x[2] for x in part)})
    return {key: float(np.mean([x[key] for x in masks])) for key in masks[0]}


def main(v):
    assert os.uname().nodename == "evc33" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ["SLURM_CPUS_PER_TASK"] == "1"
    assert sys.dont_write_bytecode
    for name, expected in v["scientific_modules"].items():
        assert sha((SOURCE / name).read_bytes()) == expected
    sys.path.insert(0, str(SOURCE))
    from rsna.competitive.predictive import cache, cases
    from rsna.competitive.predictive.inputs import masked_study

    candidate = load_ram_module("rsna.competitive.predictive.visible_contrast",
                               v["candidate_source"], v["candidate_sha256"])
    measurement = load_ram_module("original_visible_measurement",
                                 v["measurement_source"], v["measurement_sha256"])
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
    timings = {"original": 0., "matched_control": 0., "normalized": 0.}
    original_triplets = cases.triplets
    intervention = None
    for ordinal, row in enumerate(index[:12]):
        groups, _ = cache.load(Path(row["path"]), row["proof"])
        chunks = []

        def measured_triplets(study, centers, neighbors, *, side=320):
            rendered = original_triplets(study, centers, neighbors, side=side)
            chunks.append(measurement.pure_visible_values(rendered, study.raw_hidden[neighbors]))
            return rendered

        cases.triplets = measured_triplets
        try:
            t = time.monotonic()
            original = cases.assemble(groups, side=320, context_cap=24, classes=12)
            timings["original"] += time.monotonic() - t
            original_summary = summarize_render(chunks)
            chunks.clear()
            t = time.monotonic()
            normalized, scales = candidate.assemble(groups, side=320, context_cap=24, classes=12)
            timings["normalized"] += time.monotonic() - t
            normalized_summary = summarize_render(chunks)
            chunks.clear()
        finally:
            cases.triplets = original_triplets
        t = time.monotonic()
        control, control_scales = candidate.assemble(groups, side=320, context_cap=24, classes=12,
                                                    normalize_images=False)
        timings["matched_control"] += time.monotonic() - t
        assert scales == control_scales
        assert_targets_equal(original, normalized)
        assert_targets_equal(original, control)
        for old, new, matched in zip(original, normalized, control, strict=True):
            assert torch.equal(old["images"], matched["images"])
            assert not torch.equal(old["images"], new["images"])
            assert torch.isfinite(new["images"]).all()
            for field in ("token_metadata", "region_metadata"):
                assert new[field].shape[-1] == 10 and torch.isfinite(new[field]).all()
                assert torch.equal(new[field], matched[field])
                assert torch.equal(new[field][:, :8], old[field])
        observations.append({"original": original_summary, "normalized": normalized_summary})
        if ordinal == 0:
            changed = []
            for g in groups:
                hidden = masked_study(g.frames, g.aliases, g.acquisition, g.ordinal,
                                      pattern=0, cached_descriptors=g.descriptors).raw_hidden
                pixels = g.frames.clone()
                pixels[hidden] = 1 - pixels[hidden]
                changed.append(replace(g, frames=pixels, descriptors=None))
            altered, altered_scales = candidate.assemble(changed, side=320, context_cap=24, classes=12)
            assert scales[0] == altered_scales[0]
            for field in ("images", "token_metadata", "region_metadata", "observed"):
                assert torch.equal(normalized[0][field], altered[0][field])
            assert not torch.equal(normalized[0]["targets"], altered[0]["targets"])
            intervention = {"fitting_studies": 1, "hidden_pixels_replaced": "one_minus_original",
                            "context0_images_metadata_masks_and_scales_bitwise_equal": True,
                            "original_units_target_descriptors_changed": True,
                            "other_complement_is_not_required_to_remain_equal": True}
            del changed, pixels, hidden, altered, altered_scales
        frames += sum(len(g.frames) for g in groups)
        windows += sum(len(c["images"]) for c in original)
        assert cache.signature(Path(row["path"])) == row["proof"]["stat"]
        del groups, original, normalized, control
    aggregate = {role: {key: measurement.summary([x[role][key] for x in observations])
                        for key in observations[0][role]} for role in ("original", "normalized")}
    prior = v["prior_visible_summary"]
    for field in ("q99_minus_q01", "median", "pure_visible_fraction"):
        for quantile, value in prior[field].items():
            assert abs(aggregate["original"][field][quantile] - value) < 1e-12
    result = {
        "status": "actual_twelve_fitting_MRI_visible_only_contrast_candidate",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "collector_sha256": v["collector_sha256"], "candidate_sha256": v["candidate_sha256"],
        "measurement_sha256": v["measurement_sha256"], "candidate_version": candidate.VERSION,
        "scientific_modules": v["scientific_modules"], "original_raw_index_sha256": INDEX_SHA,
        "fitting_studies": 12, "excluded_engineering_studies": 4, "original_frames": frames,
        "windows_per_variant_including_both_masks": windows, "equal_context_then_study_summary": aggregate,
        "studies_below_point05_span": {role: sum(x[role]["q99_minus_q01"] < .05 for x in observations)
                                      for role in ("original", "normalized")},
        "prior_original_visible_measurement_reproduced": True,
        "all_targets_masks_complete_priors_and_first_eight_metadata_columns_unchanged": True,
        "matched_ten_metadata_control_has_original_images": True,
        "hidden_source_intervention": intervention, "render_seconds_for_twelve_studies": timings,
        "model_optimizer_labels_evaluation_inputs_read": False, "new_remote_files": 0,
        "pixels_IDs_or_per_study_results_exported": False, "optimizer_updates": 0,
        "clinical_accuracy_or_GPU_training_acceptance": False, "elapsed_seconds": time.monotonic() - started,
    }
    assert intervention and not torch.cuda.is_initialized()
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
