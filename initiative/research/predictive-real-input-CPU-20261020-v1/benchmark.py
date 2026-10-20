"""Read-only retained-MRI CPU timing; never a model or Kaggle qualification."""

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path


def equal_contexts(left, right, torch):
    """Compare every input and support field, including dtype and CSR extent."""
    keys = {"images", "token_metadata", "region_metadata", "targets", "observed", "supports"}
    if len(left) != 2 or len(right) != 2:
        raise ValueError("Both complementary contexts required")
    for a, b in zip(left, right):
        if set(a) != keys or set(b) != keys:
            raise ValueError("Unexpected or missing context field")
        for key in keys - {"supports"}:
            if a[key].dtype != b[key].dtype or not torch.equal(a[key], b[key]):
                raise ValueError("Context tensors differ")
        if a["supports"].region_count != b["supports"].region_count:
            raise ValueError("Support region count differs")
        for key in ("indptr", "regions", "log_prior"):
            x, y = getattr(a["supports"], key), getattr(b["supports"], key)
            if x.dtype != y.dtype or not torch.equal(x, y):
                raise ValueError("Support tensors differ")


def main():
    payload = json.load(sys.stdin)
    assert os.uname().nodename == "evc33"
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 2
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    source = Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code")
    for name, expected in payload["scientific_modules"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    helper = payload["descriptor_helper"]
    assert hashlib.sha256(helper.encode()).hexdigest() == payload["descriptor_helper_sha256"]
    sys.path.insert(0, str(source))
    import torch

    from rsna.competitive.predictive import cache
    from rsna.competitive.predictive.cases import assemble

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("serving_descriptors", None))
    exec(compile(helper, "frozen_serving_descriptor_helper", "exec"), module.__dict__)
    raw = Path("/tmp/rsna-predictive-raw16-cache-20261020-v1")
    original = json.loads((raw / "actual.json").read_text())
    assert original["CPU_job"] == "978232" and original["fitting_studies"] == 16
    index_body = (raw / "cache-index.private.json").read_bytes()
    assert hashlib.sha256(index_body).hexdigest() == original["private_index_sha256"] == (
        "e3585fb3cafab3e0ebbca0fb03953e38bd879376956a7a142cc5b4781784008d"
    )
    rows = json.loads(index_body)
    assert len(rows) == 16
    rng = torch.get_rng_state().clone()
    started = time.monotonic()

    def study(item):
        index, row = item
        torch.set_num_threads(1)
        load_started = time.monotonic()
        groups, _ = cache.load(Path(row["path"]), row["proof"])
        load_seconds = time.monotonic() - load_started
        fresh = [replace(group, descriptors=None) for group in groups]
        timings, values = {}, {}
        order = ("original", "descriptor_once") if index % 2 == 0 else ("descriptor_once", "original")
        for mode in order:
            began = time.monotonic()
            prepared = fresh if mode == "original" else module.cache_descriptors_once(fresh)
            prepared_seconds = time.monotonic() - began
            values[mode] = assemble(prepared, side=320, context_cap=24, classes=12)
            timings[mode] = {
                "seconds": time.monotonic() - began,
                "descriptor_preparation_seconds": prepared_seconds,
            }
        equal_contexts(values["original"], values["descriptor_once"], torch)
        assert cache.signature(Path(row["path"])) == row["proof"]["stat"]
        return {
            "study_index": index,
            "raw_cache_load_and_normalize_seconds": load_seconds,
            "original_frames": sum(len(group.frames) for group in fresh),
            "regions": row["proof"]["regions"],
            "original_first": order[0] == "original",
            "timing": timings,
            "complete_contexts_bitwise_equal": True,
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        measurements = list(pool.map(study, enumerate(rows)))
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    baseline = sum(item["timing"]["original"]["seconds"] for item in measurements)
    optimized = sum(item["timing"]["descriptor_once"]["seconds"] for item in measurements)
    receipt = {
        "status": "actual_retained16_MRI_descriptor_once_CPU_comparison",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": payload["source_git"],
        "benchmark_sha256": payload["benchmark_sha256"],
        "descriptor_helper_sha256": payload["descriptor_helper_sha256"],
        "scientific_modules": payload["scientific_modules"],
        "original_cache_job": "978232",
        "original_index_sha256": original["private_index_sha256"],
        "studies": 16,
        "workers": 2,
        "torch_threads_per_worker": 1,
        "torch_version": torch.__version__,
        "measurements": measurements,
        "sum_original_render_seconds": baseline,
        "sum_descriptor_once_render_seconds": optimized,
        "ratio_of_paired_render_sums": baseline / optimized,
        "total_wall_seconds_including_load_and_equality": time.monotonic() - started,
        "peak_host_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "all_complete_contexts_bitwise_equal": True,
        "global_RNG_unchanged": True,
        "remote_files_written": 0,
        "source_downloads": 0,
        "model_or_GPU_or_labels_read": False,
        "Kaggle_or_DICOM_decode_or_hidden_runtime_acceptance": False,
        "clinical_accuracy_claim": False,
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
