"""Private pixel-only audit of the scored parent's discarded image periphery.

No reports, labels, encoder, optimizer, new pixel archive or competition action.
The exact private producer supplies the original renderer; no external body is
vendored here. This does not accept a new model resolution or claim accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REGISTRY_SHA = "8cc5c194d4795f6ab574c28b2cf8eed510c5f92fc6b5baff63049d16323fb1df"
PRODUCER_SHA = "a1e397b02232597f24419d6f8a7641146880d9af23ed51a6d9fd90702e969fc9"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def central_subset(plain, wider):
    """Exact nested fields, without interpolation or intensity normalization changes."""
    if (
        plain.shape != (94, 3, 320, 320)
        or wider.shape != (94, 3, 384, 384)
        or plain.dtype != torch.float32
        or wider.dtype != torch.float32
        or not torch.equal(plain, wider[:, :, 32:352, 32:352])
    ):
        raise ValueError("Original central320 values must be bit-equal inside full384 field")


def audit(root, output, source_git):
    if os.uname().nodename != "evc39" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Read original owning-node CPU allocation only")
    if output.exists() or len(source_git) != 40:
        raise ValueError("Fresh immutable small output and source Git required")
    started = time.monotonic()
    registry = root / "complete-registry.private.json"
    if digest(registry.read_bytes()) != REGISTRY_SHA:
        raise ValueError("Exact completed full fitting registry required")
    proof = json.loads((root / "actual-all-verified.json").read_text())
    if proof["studies"] != 3887 or not proof["all_new_inputs_and_original32_proofs_accepted"]:
        raise ValueError("Original full population preservation proof required")
    directory = root / "shard-0"
    source = (directory / "source.private.py").read_text()
    if digest(source.encode()) != PRODUCER_SHA:
        raise ValueError("Exact original private CPU producer/renderer required")
    ns: dict[str, Any] = {
        "__name__": "private_fov_pixel_audit",
        "__file__": str(directory / "source.private.py"),
    }
    exec(compile(source, "original_private_renderer", "exec"), ns)
    index = json.loads((directory / "collected/index.private.json").read_text())
    if digest((directory / "collected/index.private.json").read_bytes()) != (
        "a0f01b906b8200a299d95440517f7deaae4bbb254ca883b36898bd2837620b6d"
    ):
        raise ValueError("Exact original first-shard membership required")
    rows = [row for row in index["records"] if row["source_key"] == "self"][:32]
    if len(rows) != 32 or len({row["study_uid"] for row in rows}) != 32:
        raise ValueError("Exactly32 distinct already-admitted new fitting cases required")
    band = np.ones((384, 384), dtype=bool)
    band[32:352, 32:352] = False
    total, signal = 0, {level: 0 for level in (0, 8, 32)}
    narrow_hash, wide_hash = hashlib.sha256(), hashlib.sha256()
    present_slices, views = 0, 0
    with tarfile.open(directory / "collected/pixels.tar", "r:") as archive:
        for row in rows:
            member = archive.getmember(row["file"])
            if not member.isreg() or member.size != row["bytes"]:
                raise ValueError("Exact original safe member required")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Original member missing")
            volume, mask = ns["decode"](handle.read(), row["sha256"])
            if (
                digest(volume.tobytes()) != row["volume_sha256"]
                or digest(mask.tobytes()) != row["mask_sha256"]
            ):
                raise ValueError("Original MRI pixels or presence mask changed")
            pixels = volume[mask.astype(bool)][:, band]
            present_slices += int(mask.sum())
            total += pixels.size
            for level in signal:
                signal[level] += int((pixels > level).sum())
            for mirrored, key in (
                (False, "original_plain_input_sha256"),
                (True, "original_mirrored_input_sha256"),
            ):
                ns["INPUT_RES"] = 320
                narrow = ns["eval_windows"](volume, mask, 94, 320, norm=ns["NORM"], mirror=mirrored)
                old = digest(narrow.contiguous().numpy().tobytes())
                if old != row[key]:
                    raise ValueError("The actual source320 renderer was not reproduced")
                ns["INPUT_RES"] = 384
                wider = ns["eval_windows"](volume, mask, 94, 384, norm=ns["NORM"], mirror=mirrored)
                central_subset(narrow, wider)
                narrow_hash.update(old.encode())
                wide_hash.update(digest(wider.contiguous().numpy().tobytes()).encode())
                views += 1
                del narrow, wider
    result = {
        "status": "actual32_original_MRI_nested320_full384_fields_and_peripheral_signal_audited",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": source_git,
        "registry_sha256": REGISTRY_SHA,
        "producer_source_sha256": PRODUCER_SHA,
        "studies": len(rows),
        "two_view_pairs_checked": views,
        "windows_per_view": 94,
        "original320_input_hashes_exact": True,
        "all_original320_values_bit_equal_inside_full384": True,
        "additional_real_image_area_relative_to320": 0.44,
        "outer_band_pixels_on_present_slices": total,
        "present_slices": present_slices,
        "outer_band_fraction_above_uint8_threshold": {
            str(level): count / total for level, count in signal.items()
        },
        "aggregate_original320_tensor_sha256": narrow_hash.hexdigest(),
        "aggregate_new384_tensor_sha256": wide_hash.hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "maximum_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "GPU_used": False,
        "model_weights_or_labels_read": False,
        "new_archive_created": False,
        "model384_acceptance": False,
        "disease_or_competition_accuracy_claim": False,
    }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with output.open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(audit(args.root, args.output, args.source_git)), flush=True)


if __name__ == "__main__":
    main()
