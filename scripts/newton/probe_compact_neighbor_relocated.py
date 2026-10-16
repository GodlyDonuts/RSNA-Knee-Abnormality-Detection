"""CPU-only byte/tensor/runtime admission of the frozen neighbor recipe on evc39."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def validate_relocated_binding(binding, original, cache):
    """Only the physical cache location changes; scientific bindings stay exact."""
    if (
        binding.get("cache") != str(cache.resolve())
        or {
            **binding,
            "cache": original.get("cache"),
        }
        != original
    ):
        raise ValueError("Relocation changed more than the exact physical cache location")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-manifest-sha256", required=True)
    p.add_argument("--transport-proof", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.uname().nodename != "evc39" or os.environ.get("SLURM_JOB_GPUS") or args.output.exists():
        raise ValueError("Fresh CPU-only original evc39 endpoint required")
    source = Path("/tmp/rsna-compact-matched-neighbor-inputs-20261016-v1/code")
    original = source.parent
    root = Path("/tmp/rsna-compact-neighbor-evc39-20261016-v1")
    cache = root / "cache"
    sys.path[:0] = [str(source), "/tmp/rsna-compact-native-n3-assets-20261016-v1/runtime-overlay"]
    import torch

    from rsna.competitive.checkpoint import atomic_json, file_sha256
    from rsna.competitive.compact_bag import (
        CompactBagClassifier,
        candidate_windows,
        load_encoder,
        window_tensor,
    )
    from rsna.competitive.compact_source_neighbors import (
        POLICY,
        MatchedSourceNeighborDataset,
        select_window_tensor,
    )
    from rsna.competitive.compact_train import admitted_datasets, input_census

    torch.set_num_threads(1)
    if not args.transport_proof.resolve().is_relative_to(root.resolve()):
        raise ValueError("Private transport proof must belong to this endpoint")
    runtime = json.loads(args.transport_proof.read_text())
    native = json.loads((root / "native-cache-verified.json").read_text())
    if (
        runtime["manifest_sha256"] != args.runtime_manifest_sha256
        or runtime["status"] != "actual_original_cache_runtime_TLS_prefix_suffix_verified"
        or runtime["private_TLS_and_RAM_token"] is not True
        or runtime["runtime_files"] < 1000
    ) or (
        native["status"] != "native_original_cache_evc39_whole_bytes_verified"
        or native["members"] != 37812
        or native["bytes"] != 140626483351
        or native["registry_sha256"]
        != "73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770"
        or not native["all_original_member_size_SHA_verified_during_transfer"]
        or file_sha256(root / "cache-signatures.json") != native["cache_signatures_sha256"]
    ):
        raise ValueError("Original transport whole-byte proofs required")
    # A stat replay reuses the transfer hashes. No duplicate full-cache hashing.
    signatures = json.loads((root / "cache-signatures.json").read_text())
    for name, expected in signatures.items():
        path = root / name
        s = path.stat()
        if (
            path.is_symlink()
            or [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] != expected
        ):
            raise ValueError("Transferred original cache signature changed")
    old = json.loads((original / "inputs/inputs.json").read_text())
    old_admission = json.loads((original / "accepted.json").read_text())
    old_run = json.loads(
        Path(
            "/tmp/rsna-compact-native-n3-production-evc33-20261016-v1/production/run.json"
        ).read_text()
    )["binding"]
    if (
        file_sha256(original / "accepted.json")
        != "7e129302861de2a4b584f3df628163cbbfe6d537073c636a1f2b4e6c1589b352"
    ):
        raise ValueError("Original evc33 input authority changed")
    actual = {str(f.relative_to(source)) for f in (source / "rsna").rglob("*.py")}
    expected = old["binding"]["source_sha256"]
    if actual != set(expected) or any(
        file_sha256(source / name) != h for name, h in expected.items()
    ):
        raise ValueError("Complete frozen numerical source differs")
    artifact = Path("/home/sa305415/rsna-knee/code-stopfix-1f486765")
    data_args = argparse.Namespace(
        studies=artifact / "initiative/releases/report-supervision-v1/studies.parquet",
        folds=artifact / "initiative/experiments/competitive-v1/folds",
        series=artifact
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
        artifact_root=artifact,
        cache=cache,
        fitting_supervision=Path("/tmp/rsna-public-label-repeat-inputs-20261012-v1/overlay.json"),
        engineering=False,
    )
    fitting, outer, binding = admitted_datasets(data_args)
    binding.update(input_policy=dict(POLICY), source_sha256=expected)
    validate_relocated_binding(binding, old["binding"], cache)
    if (len(fitting), len(outer)) != (2247, 1156):
        raise ValueError("Original labels/masks/population/folds/source/policy differ")
    census = {role: input_census(data) for role, data in [("fitting", fitting), ("outer", outer)]}
    if census != old["all_study_window_census"]:
        raise ValueError("Complete original window census differs")
    started = time.monotonic()
    tensors, legacy = hashlib.sha256(), hashlib.sha256()
    changed, total, fallbacks = 0, 0, 0
    for base in (fitting, outer):
        view = MatchedSourceNeighborDataset(base)
        for index in range(16):
            a, b = base[index], view[index]
            windows = candidate_windows(a["series_mask"], a["slice_mask"])
            first, second = window_tensor(a["images"], windows), select_window_tensor(b, windows)
            if not torch.equal(first[:, 1], second[:, 1]):
                raise ValueError("Original grayscale centers differ")
            tensors.update(second.contiguous().numpy().tobytes())
            tensors.update(b["source_window_plan_sha256"].encode())
            legacy.update(first.contiguous().numpy().tobytes())
            for key in ("labels", "label_mask", "series_mask", "slice_mask"):
                if not torch.equal(a[key], b[key]):
                    raise ValueError("Original masks/labels/window membership differ")
                tensors.update(b[key].contiguous().numpy().tobytes())
                legacy.update(a[key].contiguous().numpy().tobytes())
            changed += int((first != second).reshape(len(windows), -1).any(dim=1).sum())
            total += len(windows)
            fallbacks += b["legacy_edge_fallback_windows"]
    if tensors.hexdigest() != old["engineering32_tensors_sha256"] or (
        changed != old["engineering32_changed_donor_windows"]
        or total != old["engineering32_total_windows"]
        or fallbacks != old["engineering32_legacy_fallback_windows"]
    ):
        raise ValueError("Actual original32 MRI tensors differ after relocation")
    if legacy.hexdigest() != "183097af0095317c7a85069e73c8b2c654e39cfc12cdbc7104ff3139a30bfe9b":
        raise ValueError("Actual original32 N3 window tensors differ after relocation")
    weights = Path(
        "/tmp/rsna-compact-native-n3-assets-20261016-v1/initialization/model.safetensors"
    )
    torch.manual_seed(20260922)
    model = CompactBagClassifier(load_encoder(weights))
    head = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("encoder."):
            head.update(name.encode())
            head.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    if (
        head.hexdigest() != old_run["initial_head_sha256"]
        or sum(p.numel() for p in model.parameters()) != 73136096
    ):
        raise ValueError("Exact full model and matched initial head differ")
    free = shutil.disk_usage("/tmp").free
    if free < 581021415363 + 2 * 2**20:
        raise ValueError("Conservative preexisting obligations and new fit must remain funded")
    args.output.mkdir(mode=0o700)
    atomic_json(args.output / "inputs/inputs.json", {**old, "binding": binding})
    receipt = dict(
        old_admission,
        node="evc39",
        cache=str(cache),
        root=str(args.output),
        job_id=os.environ["SLURM_JOB_ID"],
        input_receipt_sha256=file_sha256(args.output / "inputs/inputs.json"),
        actual_original32_tensor_replay_bit_equal=True,
        original_N3_engineering32_window_tensor_sha256=legacy.hexdigest(),
        actual_original_initial_head_equal=True,
        original_evc33_admission_sha256="7e129302861de2a4b584f3df628163cbbfe6d537073c636a1f2b4e6c1589b352",
        runtime_manifest_sha256=runtime["manifest_sha256"],
        native_registry_sha256=native["registry_sha256"],
        relocation_CPU_seconds=time.monotonic() - started,
        free_after_conservative_all_liabilities=free - 581021415363 - 2 * 2**20,
    )
    atomic_json(args.output / "accepted.json", receipt)
    print(
        json.dumps({**receipt, "accepted_sha256": file_sha256(args.output / "accepted.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
