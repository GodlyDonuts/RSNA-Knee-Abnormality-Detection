"""Own-code wrapper embedded after the unchanged private competition functions."""
# ruff: noqa: F821

import hashlib
import json
import os
import time


def _rank_delta_public_volumes(root, ids):
    import pandas as pd

    assert len(ids) == 3
    series = pd.read_csv(root / "test_series.csv")
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    groups = {k: v.to_dict("records") for k, v in series.groupby("StudyInstanceUID")}
    directory = root / "test_series"
    if not directory.is_dir():
        directory = root / "test_images"
    assert directory.is_dir() and all(uid in groups for uid in ids)
    reader = _load_fastread().make_builder(
        FASTREAD_VARIANT, IMG, SLOTS, hdr_threads=FR_HDR, px_threads=FR_PX,
    )
    arrays = []
    for uid in ids:
        volume, mask = reader(uid, groups, str(directory))
        assert volume.shape == (96, 384, 384) and volume.dtype == np.uint8
        assert mask.shape == (96,) and mask.dtype == np.uint8 and mask.any()
        assert np.array_equal(mask, (volume.reshape(96, -1).sum(1) > 0).astype(np.uint8))
        arrays.append((volume, mask))
    assert sum(v.nbytes + m.nbytes for v, m in arrays) <= 43 * 2**20
    return arrays


def _rank_delta_check_device(index, arrays, training_graph):
    started = time.monotonic()
    model = reference = None
    maximum = 0.0
    cpu_input_hashes = []
    with torch.cuda.device(index), torch.no_grad():
        device = "cuda:" + str(index)
        try:
            model, _ = load_model(str(_adapted_model_path), ARMS[0]["arch"], 384, device)
            reference = training_graph(
                build_backbone(ARMS[0]["arch"], pretrained=False), F_dim=1024,
            ).to(device)
            reference.load_state_dict(model.state_dict(), strict=True)
            reference.eval()
            actual_state, reference_state = model.state_dict(), reference.state_dict()
            assert actual_state.keys() == reference_state.keys()
            assert all(torch.equal(value, reference_state[key]) for key, value in actual_state.items())
            del actual_state, reference_state
            assert all(not module.training for module in model.modules())
            for volume, mask in arrays:
                original, served = [], []
                for mirror in (False, True):
                    cpu = eval_windows(volume, mask, 94, 320, norm=NORM, mirror=mirror)
                    gpu = gpu_windows(volume, mask, 94, 320, device, norm=NORM, mirror=mirror)
                    assert cpu.shape == gpu.shape == (94, 3, 320, 320)
                    assert torch.equal(cpu, gpu.cpu())
                    cpu_input_hashes.append(hashlib.sha256(cpu.contiguous().numpy().tobytes()).hexdigest())
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = reference(gpu[None]).float()
                    assert logits.shape == (1, 12) and torch.isfinite(logits).all()
                    original.append(logits.sigmoid()[0].cpu().numpy())
                    # Exact old infer_probs, without competition counters altered by this selftest.
                    value = _adapted_original_infer(model, gpu, device)
                    assert value.shape == (12,) and np.isfinite(value).all()
                    served.append(value)
                    del cpu, gpu, logits, value
                error = float(np.max(np.abs(.5 * (served[0] + served[1]) - .5 * (original[0] + original[1]))))
                maximum = max(maximum, error)
                assert error <= 1e-6, "New ranking weights differ between training and serving graphs"
            assert len(cpu_input_hashes) == 6
            torch.cuda.synchronize(index)
            return {
                "index": index, "GPU": torch.cuda.get_device_name(index), "MRI_studies": 3,
                "both_original94_input_views_bit_equal": True,
                "complete_learned_graph_tensor_values_equal": True,
                "maximum_original_class_vs_training_graph_probability_error": maximum,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "input_views_aggregate_sha256": hashlib.sha256(json.dumps(cpu_input_hashes).encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
            }
        finally:
            del model, reference
            torch.cuda.empty_cache()


def _rank_delta_public_check(root, ids, roster_sha):
    from concurrent.futures import ThreadPoolExecutor

    started = time.monotonic()
    assert roster_sha == _RANK_DELTA_BINDING["public_roster_sha256"] and len(ids) == 3
    assert _ADAPTED_SHA == _RANK_DELTA_BINDING["model_sha256"]
    assert _ADAPTED_TRAINING["role"] == "teacher_rank" and _ADAPTED_TRAINING["epoch"] == 3
    assert _ADAPTED_TRAINING["receipt"]["completed"] == 1458
    assert _ADAPTED_TRAINING["source_sha256"] == _RANK_DELTA_BINDING["training_source_sha256"]
    assert torch.cuda.device_count() == 2 and all(
        "T4" in torch.cuda.get_device_name(i) for i in range(2)
    )
    before = dict(_adapted_counts)
    assert before == {"started": 0, "completed": 0, "failed": 0, "decode_errors": 0, "empty_views": 0}
    mix, importance = {"__name__": "original_full94_mixin"}, {"__name__": "original_readout"}
    exec(compile(_ADAPTED_MIXIN, "original_full94_mixin", "exec"), mix)
    exec(compile(_ADAPTED_IMPORTANCE, "original_readout", "exec"), importance)

    class TrainingGraph(importance["ImportanceReadout"], mix["CheckpointedEncode"], RaptorClassifier):
        pass

    arrays = _rank_delta_public_volumes(root, ids)
    decode_seconds = time.monotonic() - started
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_rank_delta_check_device, i, arrays, TrainingGraph) for i in range(2)]
        devices = [future.result() for future in futures]
    assert [d["index"] for d in devices] == [0, 1]
    assert devices[0]["input_views_aggregate_sha256"] == devices[1]["input_views_aggregate_sha256"]
    assert _adapted_counts == before
    del arrays
    return {
        **_RANK_DELTA_BINDING,
        "status": "actual_teacher_rank_public3_bothT4_delta_qualified",
        "devices": devices, "views_per_study": 2, "windows_per_view": 94,
        "actual_new_weight_studies_per_device": 3, "inherited_code_qualification_studies": 32,
        "inherited32_reexecuted": False, "clinical_accuracy_claim": False,
        "public_decode_seconds": decode_seconds, "elapsed_seconds": time.monotonic() - started,
    }


def _rank_delta_main():
    timer = _RANK_DELTA_EARLY["timer"]
    assert not _RANK_DELTA_OUTPUT.exists() and not _RANK_DELTA_RECEIPT.exists()
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        root, ids, roster_sha = (
            _RANK_DELTA_EARLY[key] for key in ("root", "ids", "sha256")
        )
        public = roster_sha == _RANK_DELTA_BINDING["public_roster_sha256"]
        assert (timer is not None) == public
        _adapted_boot()  # Original complete finite learned-state/provenance validation.
        acceptance = _rank_delta_public_check(root, ids, roster_sha) if public else None
        result = _adapted_competition()
        assert result["test_roster_sha256"] == roster_sha and result["studies"] == len(ids)
        assert result["model_sha256"] == _RANK_DELTA_BINDING["model_sha256"]
        assert result["prediction_counts"] == {
            "started": 2 * len(ids), "completed": 2 * len(ids),
            "failed": 0, "decode_errors": 0, "empty_views": 0,
        }, "No inference or preprocessing fallback is accepted"
        result.update(
            role=_ADAPTED_TRAINING["role"], training_windows=16, serving_windows=94,
            proposal_sha256="e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e",
        )
        if public:
            result["inline_new_weight_acceptance"] = acceptance
            result["public_total_elapsed_seconds"] = time.monotonic() - _RANK_DELTA_EARLY["started"]
            assert result["public_total_elapsed_seconds"] <= _RANK_DELTA_BINDING["public_watchdog_seconds"]
        _RANK_DELTA_RECEIPT.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException:
        _rank_delta_discard_outputs()
        raise
    finally:
        if timer is not None:
            timer.cancel()
