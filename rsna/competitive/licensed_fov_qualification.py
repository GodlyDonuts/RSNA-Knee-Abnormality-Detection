"""Label-free, unchanged-weight field-of-view qualification on original MRI pixels.

The licensed source and private input indexes are supplied at runtime. Only
aggregate engineering evidence is emitted; no images, case IDs or predictions.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import resource
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

PIXEL_INDEX_SHA = "3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3"
FEATURE_INDEX_SHA = "a1bba9e618444f16ddd453efa1bb8650245f8259dcd9a00348b17efd87c1e100"
PARENT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PARAMETERS = 73136096
# Reuse the already accepted cross-cache parent probability bound, before
# observing this experiment. Neither a failed bound nor accuracy selects a size.
CONTROL_PROBABILITY_TOLERANCE = 0.003


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def locate_index(size, sha):
    matches = []
    for directory in Path("/kaggle/input").iterdir():
        if not directory.is_dir():
            continue
        for root, dirs, files in os.walk(directory, followlinks=False):
            if any(
                n in files or n in dirs
                for n in ("train.csv", "test.csv", "train_images", "train_series")
            ):
                dirs[:] = []
                continue
            if len(Path(root).relative_to(directory).parts) >= 5:
                dirs[:] = []
            if "index.private.json" in files:
                p = Path(root) / "index.private.json"
                if (
                    not p.is_symlink()
                    and p.stat().st_size == size
                    and digest(p.read_bytes()) == sha
                ):
                    matches.append(p)
    if len(matches) != 1:
        raise ValueError("One exact immutable original input index required")
    return matches[0]


def parameter_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def original_cases(codec):
    pixels = locate_index(21324, PIXEL_INDEX_SHA)
    features = locate_index(8574, FEATURE_INDEX_SHA)
    rows = json.loads(pixels.read_text())["records"]
    feature_index = json.loads(features.read_text())
    expected = {row["study_uid"]: row for row in feature_index["records"]}
    if (
        len(rows) != 32
        or len(expected) != 32
        or set(expected) != {row["study_uid"] for row in rows}
        or feature_index["weights_sha256"] != PARENT_SHA
        or (pixels.parent / "pixels.tar").stat().st_size != 326799360
    ):
        raise ValueError("Exactly the original paired32 MRI pixel/feature studies required")
    namespace: dict[str, Any] = {"__name__": "licensed_fov_original_codec"}
    exec(compile(codec, "original_pixel_codec", "exec"), namespace)
    cases = []
    with tarfile.open(pixels.parent / "pixels.tar", "r:") as archive:
        members = {m.name: m for m in archive.getmembers()}
        if len(members) != 32 or set(members) != {row["file"] for row in rows}:
            raise ValueError("Exact original archive membership required")
        for row in rows:
            member = members[row["file"]]
            if not member.isreg() or member.size != row["bytes"]:
                raise ValueError("Original regular pixel member required")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Missing original pixel member")
            volume, mask = namespace["decode"](handle.read(), row["sha256"])
            if (
                digest(volume.tobytes()) != row["volume_sha256"]
                or digest(mask.tobytes()) != row["mask_sha256"]
            ):
                raise ValueError("Original MRI volume or mask changed")
            feature = expected[row["study_uid"]]
            raw = (features.parent / feature["file"]).read_bytes()
            if len(raw) != feature["bytes"] or digest(raw) != feature["sha256"]:
                raise ValueError("Original saved teacher record changed")
            with np.load(io.BytesIO(raw), allow_pickle=False) as record:
                logits = record["parent_logits"].copy()
            if logits.shape != (2, 12) or not np.isfinite(logits).all():
                raise ValueError("Original two-view parent logits required")
            probability = torch.from_numpy(logits).sigmoid().mean(0).numpy()
            cases.append((row, volume, mask, probability))
    return cases


def qualify(parent_body, codec):
    started = time.monotonic()
    if torch.cuda.device_count() != 2 or not all(
        "T4" in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise ValueError("Actual two-T4 engineering allocation required")
    if Path("/kaggle/working/submission.csv").exists():
        raise ValueError("Qualification must not create or reuse a competition CSV")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = original_cases(codec)
    namespaces = []
    for _ in range(2):
        ns: dict[str, Any] = {"__name__": "licensed_fov_parent_definitions"}
        exec(compile(parent_body, "original_licensed_parent_definitions", "exec"), ns)
        if (ns["INPUT_RES"], ns["IMG"], ns["K_EVAL"]) != (320, 384, 94) or len(ns["ARMS"]) != 1:
            raise ValueError("Unchanged original 94-window MRI graph required")
        namespaces.append(ns)
    # Both roles use the same backend policy. The saved original control has an
    # existing 0.003 cross-cache probability bound, not a newly tuned tolerance.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    def worker(index):
        ns = namespaces[index]
        device = "cuda:" + str(index)
        outputs, records, learned = {}, [], []
        with torch.cuda.device(device), torch.inference_mode():
            for side in (320, 384):
                ns["INPUT_RES"] = side
                torch.cuda.reset_peak_memory_stats(index)
                model, actual_side = ns["load_model"](
                    ns["find_weight_file"](ns["ARMS"][0]["file"]),
                    ns["ARMS"][0]["arch"],
                    384,
                    device,
                )
                if actual_side != side or sum(p.numel() for p in model.parameters()) != PARAMETERS:
                    raise ValueError("Exact complete unchanged classifier required")
                if any(m.training for m in model.modules()):
                    raise ValueError("Inference must not update BN or activate dropout")
                learned.append(parameter_digest(model))
                probabilities, durations = [], []
                for row, volume, mask, _ in cases:
                    began = time.monotonic()
                    values = []
                    for mirror in (False, True):
                        cpu = ns["eval_windows"](
                            volume, mask, 94, side, norm=ns["NORM"], mirror=mirror
                        )
                        key = (
                            "original_mirrored_input_sha256"
                            if mirror
                            else "original_plain_input_sha256"
                        )
                        if side == 320:
                            original = cpu
                        else:
                            ns["INPUT_RES"] = 320
                            original = ns["eval_windows"](
                                volume, mask, 94, 320, norm=ns["NORM"], mirror=mirror
                            )
                            ns["INPUT_RES"] = side
                            if not torch.equal(original, cpu[:, :, 32:352, 32:352]):
                                raise ValueError(
                                    "Original central320 values changed inside384 field"
                                )
                        if digest(original.contiguous().numpy().tobytes()) != row[key]:
                            raise ValueError("Actual original320 tensor hashes must reproduce")
                        gpu = ns["gpu_windows"](
                            volume, mask, 94, side, device, norm=ns["NORM"], mirror=mirror
                        )
                        if not torch.equal(cpu, gpu.cpu()):
                            raise ValueError("CPU/GPU original-image tensors must be bit equal")
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(gpu[None]).float()
                        if logits.shape != (1, 12) or not torch.isfinite(logits).all():
                            raise ValueError("Finite full94 model logits required")
                        values.append(logits.sigmoid()[0].cpu().numpy())
                        del cpu, original, gpu
                    probabilities.append(0.5 * (values[0] + values[1]))
                    torch.cuda.synchronize(index)
                    durations.append(time.monotonic() - began)
                    if time.monotonic() - started >= 1700:
                        raise TimeoutError(
                            "Bounded original32 qualification exceeded its allocation"
                        )
                outputs[side] = np.stack(probabilities)
                if torch.cuda.max_memory_allocated(index) >= 14 * 2**30:
                    raise ValueError("Each role must retain twoGiB of T4 VRAM margin")
                records.append(
                    {
                        "input_side": side,
                        "studies": 32,
                        "two_views_each": True,
                        "CPU_GPU_inputs_bit_equal": True,
                        "original320_input_hashes_reproduced": True,
                        "median_cached_input_study_seconds": float(np.median(durations)),
                        "p95_cached_input_study_seconds": float(np.quantile(durations, 0.95)),
                        "peak_GPU_bytes": torch.cuda.max_memory_allocated(index),
                    }
                )
                del model
                torch.cuda.empty_cache()
            if learned[0] != learned[1]:
                raise ValueError("The384 model must retain every learned parameter tensor value")
            expected = np.stack([case[3] for case in cases])
            control_error = float(np.abs(outputs[320] - expected).max())
            if control_error > CONTROL_PROBABILITY_TOLERANCE:
                raise ValueError("Original320 control exceeded its predeclared saved-output bound")
            delta = np.abs(outputs[384] - outputs[320])
            return (
                {
                    "index": index,
                    "GPU": torch.cuda.get_device_name(index),
                    "roles": records,
                    "all_learned_parameters_equal_320_384": True,
                    "learned_parameter_sha256": learned[0],
                    "control_probability_error": control_error,
                    "control_probability_tolerance": CONTROL_PROBABILITY_TOLERANCE,
                    "mean_absolute_probability_change_384_vs320": float(delta.mean()),
                    "maximum_absolute_probability_change_384_vs320": float(delta.max()),
                },
                outputs[384],
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, i) for i in range(2)]
        checked = [future.result() for future in futures]
    if checked[0][0]["learned_parameter_sha256"] != checked[1][0]["learned_parameter_sha256"]:
        raise ValueError("Both devices must load the same unchanged learned tensors")
    cross_gpu = float(np.abs(checked[0][1] - checked[1][1]).max())
    if cross_gpu > 1e-5 or Path("/kaggle/working/submission.csv").exists():
        raise ValueError("Both-device384 numerical parity and absence of competition CSV required")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= 27 * 2**30:
        raise ValueError("Qualification must retain a system-memory reserve")
    return {
        "status": "actual_original32_bothT4_unchanged_weights_full384_field_qualified",
        "parent_checkpoint_sha256": PARENT_SHA,
        "original_pixel_index_sha256": PIXEL_INDEX_SHA,
        "original_feature_index_sha256": FEATURE_INDEX_SHA,
        "studies": 32,
        "windows_per_view": 94,
        "trainable_or_optimized_parameters": 0,
        "loaded_parameter_count": PARAMETERS,
        "devices": [value[0] for value in checked],
        "maximum_cross_GPU384_probability_difference": cross_gpu,
        "elapsed_seconds": time.monotonic() - started,
        "maximum_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "individual_predictions_or_records_exported": False,
        "DICOM_decoding_timing_included": False,
        "full_hidden_runtime_guarantee": False,
        "clinical_labels_reports_read": False,
        "clinical_or_competition_accuracy_claim": False,
        "competition_entry": False,
    }
