"""Original32 MRI streaming equivalence; no model/optimizer/clinical evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

# A generated private notebook prefixes immutable source hashes/overlays.
CODE_OVERRIDES = globals().get("CODE_OVERRIDES", {})
CODE_OVERRIDE_HASHES = globals().get("CODE_OVERRIDE_HASHES", {})
if __name__ == "__main__" and "--help" not in sys.argv:
    control = Path(sys.argv[sys.argv.index("--control") + 1])
    destination = Path(sys.argv[sys.argv.index("--output") + 1])
    source = destination.parent / "streaming-input-source"
    source.mkdir(mode=0o700)
    original = json.loads((control / "original-input-proof.json").read_text())
    required = {"rsna/data.py", "rsna/competitive/data.py", "rsna/competitive/streaming_cache.py"}
    if set(CODE_OVERRIDES) != required or set(CODE_OVERRIDE_HASHES) != required:
        raise ValueError("Exact streaming-only source overlay required")
    for name, sha in original["source_sha256"].items():
        raw = (control / "code" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Original scientific source differs")
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for name, text in CODE_OVERRIDES.items():
        if hashlib.sha256(text.encode()).hexdigest() != CODE_OVERRIDE_HASHES[name]:
            raise ValueError("Streaming source differs")
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    sys.path.insert(0, str(source))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rsna.competitive.data import CompetitiveStudyDataset  # noqa: E402
from rsna.competitive.folds import load_folds  # noqa: E402
from rsna.competitive.streaming_cache import StreamingCache  # noqa: E402
from rsna.competitive.train import _uid_hash, select_development  # noqa: E402
from rsna.preprocess import Source  # noqa: E402
from rsna.release import load_verified_release  # noqa: E402


def digest_tensors(row):
    records = {}
    for name, value in row.items():
        if isinstance(value, torch.Tensor):
            a = value.contiguous().numpy()
            records[name] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            }
        else:
            records[name] = value
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "reference", "competition", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    proof = json.loads((args.control / "original-input-proof.json").read_text())
    artifact = args.control / "artifacts"
    frame, binding = load_verified_release(
        artifact / "initiative/releases/report-supervision-v1/studies.parquet", artifact
    )
    folds = load_folds(artifact / "initiative/experiments/competitive-v1/folds", frame, binding)
    train, outer = select_development(frame, folds, 0)
    if len(train) != 2247 or len(outer) != 1156:
        raise ValueError("Original fitting/development membership differs")
    if (
        _uid_hash(train.head(16)) != proof["training_16_sha256"]
        or _uid_hash(outer.head(16)) != proof["outer_16_sha256"]
    ):
        raise ValueError("Original engineering membership differs")
    series_path = (
        artifact / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    )
    if hashlib.sha256(series_path.read_bytes()).hexdigest() != binding["series_sha256"]:
        raise ValueError("Original series sampling differs")
    series = pd.read_csv(series_path)
    provider = StreamingCache(
        Source(args.competition), args.output / "scratch", capacity_bytes=256 * 2**20
    )
    verified = {}
    for role, roster in [("fitting", train.head(16)), ("engineering_outer", outer.head(16))]:
        ordinary = CompetitiveStudyDataset(
            roster, series, args.reference, slices=16, side=384, max_series=6
        )
        streamed = CompetitiveStudyDataset(
            roster,
            series,
            args.reference,
            slices=16,
            side=384,
            max_series=6,
            volume_provider=provider.read,
        )
        hashes = []
        for i in list(range(16)) + [0]:
            a = ordinary[i]
            b = streamed[i]
            for name, value in a.items():
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, b[name]):
                        raise ValueError(
                            "Original MRI/geometry/labels/masks changed through streaming"
                        )
                elif value != b[name]:
                    raise ValueError("Original study identity changed")
            hashes.append(digest_tensors(b))
            print(
                json.dumps(
                    {
                        "role": role,
                        "studies_verified": i + 1,
                        "resident_cache_bytes": provider.used_bytes,
                    }
                ),
                flush=True,
            )
        verified[role] = {
            "studies": 16,
            "repeat_first_equal": hashes[0] == hashes[-1],
            "whole_tensor_sequence_sha256": hashlib.sha256(json.dumps(hashes).encode()).hexdigest(),
        }
    result = {
        "status": "actual_original32_MRI_streaming_tensors_exactly_equal",
        "full_fitting_studies": 2247,
        "full_disjoint_outer_studies": 1156,
        "roles": verified,
        "cache": provider.statistics(),
        "source_overlay_sha256": CODE_OVERRIDE_HASHES,
        "GPU_model_optimizer_executed": False,
        "clinical_accuracy_evidence": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_host_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "CPU_receipt_sha256": hashlib.sha256(
            (args.control.parent / "witness-inputs-CPU-accepted.json").read_bytes()
        ).hexdigest(),
    }
    (args.output / "streaming-input-accepted.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
