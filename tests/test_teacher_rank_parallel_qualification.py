from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "initiative/research/teacher-rank-parallel-qualification-20261020-v1"
SPEC = importlib.util.spec_from_file_location("prepared_rank_parallel", HERE / "adapter.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def test_exact_compiler_and_one_function_replacement_preserve_every_other_byte():
    compiler = (ROOT / "scripts/kaggle/licensed_importance_serving.py").read_text()
    original = adapter._original_function(compiler)
    prefix = "# synthetic private-parent placeholder\n@torch.no_grad()\n"
    suffix = "\n# original qualification dispatch and competition definitions unchanged\n"
    source = prefix + original + suffix
    result = adapter._replace_qualification(source, original)
    assert result == prefix + adapter.PARALLEL_FUNCTION + suffix.lstrip("\n")
    assert result.startswith(prefix) and result.endswith(suffix.lstrip("\n"))
    compile(result, "synthetic_original_qualification_only", "exec")
    with pytest.raises(ValueError, match="compiler"):
        adapter._original_function(compiler + "\n")
    with pytest.raises(ValueError, match="final ranking"):
        adapter.adapt(source, compiler_source=compiler)
    with pytest.raises(ValueError, match="body changed"):
        adapter._replace_qualification(source.replace("error<=1e-6", "error<=1e-3"), original)
    with pytest.raises(ValueError, match="decorator"):
        adapter._replace_qualification(source.replace("@torch.no_grad()", "@torch.inference_mode()"), original)
    with pytest.raises(ValueError, match="One exact"):
        adapter._replace_qualification(source + "\n" + original, original)


def synthetic_function(tmp_path, monkeypatch, *, failure=None):
    """Execute the real generated loop with tiny synthetic CPU tensors and two CUDA stand-ins."""
    index = tmp_path / "index.private.json"
    path = tmp_path / "pixels.tar"
    rows = []

    def windows(volume, mirror):
        return torch.full((94, 3, 2, 2), float(volume[0]) / 32 + int(mirror) / 4)

    with tarfile.open(path, "w:") as tar:
        for case in range(32):
            body = bytes([case])
            member = tarfile.TarInfo(f"synthetic-{case:02d}.bin")
            member.size = len(body)
            tar.addfile(member, io.BytesIO(body))
            volume, mask = np.array([case], dtype=np.uint8), np.ones(1, dtype=np.uint8)
            rows.append({
                "file": member.name, "bytes": len(body), "sha256": sha(body),
                "volume_sha256": sha(volume.tobytes()), "mask_sha256": sha(mask.tobytes()),
                "original_plain_input_sha256": sha(windows(volume, False).numpy().tobytes()),
                "original_mirrored_input_sha256": sha(windows(volume, True).numpy().tobytes()),
            })
    if failure == "pixel_sha":
        rows[0]["sha256"] = "0" * 64
    if failure == "input_sha":
        rows[0]["original_plain_input_sha256"] = "0" * 64
    index.write_text(json.dumps({"records": rows}))
    counts, lock = Counter(), threading.Lock()
    threads = threading.local()
    barrier = threading.Barrier(2)
    archives = []
    original_open = tarfile.open

    class ObservedTar:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.thread = threading.get_ident()
            self.closed = False

        def __enter__(self):
            with lock:
                archives.append(self)
            return self

        def __exit__(self, *args):
            self.wrapped.close()
            self.closed = True

        def getmembers(self):
            return self.wrapped.getmembers()

        def extractfile(self, member):
            assert threading.get_ident() == self.thread
            with lock:
                counts[("read", threads.device)] += 1
            return self.wrapped.extractfile(member)

    def open_tar(*args, **kwargs):
        assert args == (path, "r:")
        return ObservedTar(original_open(*args, **kwargs))

    monkeypatch.setattr(tarfile, "open", open_tar)

    class CUDA:
        @contextmanager
        def device(self, index):
            assert not hasattr(threads, "device")
            threads.device = index
            try:
                yield
            finally:
                del threads.device

        def synchronize(self, index):
            assert threads.device == index

        def empty_cache(self):
            with lock:
                counts[("cleanup", threads.device)] += 1

        def get_device_name(self, index):
            assert threads.device == index
            return "Tesla T4 synthetic"

        def max_memory_allocated(self, index):
            assert threads.device == index
            return 2 * 2**30

    @contextmanager
    def autocast(kind, dtype):
        assert kind == "cuda" and dtype is torch.float16
        assert not torch.is_grad_enabled()
        assert threads.device in (0, 1)
        yield

    fake_torch = SimpleNamespace(
        cuda=CUDA(), no_grad=torch.no_grad, equal=torch.equal,
        isfinite=torch.isfinite, float16=torch.float16, autocast=autocast,
    )

    class Model(torch.nn.Module):
        def __init__(self, backbone=None, F_dim=None):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.register_buffer("running_mean", torch.tensor(0.5))
            self.kind = "reference"

        def to(self, device):
            assert device == f"cuda:{threads.device}"
            return self

        def load_state_dict(self, state, strict):
            assert strict is True
            result = super().load_state_dict(state, strict=True)
            if failure == "learned_state" and threads.device == 1:
                self.running_mean.add_(1)
            return result

        def forward(self, value):
            assert not self.training and not torch.is_grad_enabled()
            assert tuple(value.shape) == (1, 94, 3, 2, 2)
            with lock:
                counts[(self.kind, threads.device)] += 1
            result = (value.mean() * self.weight + self.running_mean).expand(1, 12).clone()
            if failure == "nonfinite" and self.kind == "reference" and threads.device == 1:
                result.fill_(float("nan"))
            if failure == "probability" and self.kind == "served" and threads.device == 1:
                result.add_(2)
            return result

    def load_model(_path, arch, size, device):
        assert arch == "synthetic" and size == 384
        assert device == f"cuda:{threads.device}" and not torch.is_grad_enabled()
        # Both workers must be live together, not just scheduled serially.
        barrier.wait(timeout=3)
        with lock:
            counts[("load", threads.device)] += 1
        model = Model()
        model.kind = "served"
        model.eval()
        return model, 320

    def evaluate(volume, mask, count, side, *, norm, mirror):
        assert count == 94 and side == 320 and norm == "original"
        assert mask.tolist() == [1] and not torch.is_grad_enabled()
        with lock:
            counts[("CPU_view", threads.device, mirror)] += 1
        return windows(volume, mirror)

    def prepare_gpu(volume, mask, count, side, device, *, norm, mirror):
        assert count == 94 and side == 320 and device == f"cuda:{threads.device}"
        assert norm == "original"
        with lock:
            counts[("GPU_view", threads.device, mirror)] += 1
        value = windows(volume, mirror)
        if failure == "GPU_input" and threads.device == 1:
            value.add_(1)
        return value

    def infer_pair(model, plain, mirrored, device):
        assert device == f"cuda:{threads.device}"
        return .5 * (
            model(plain[None]).sigmoid()[0].numpy()
            + model(mirrored[None]).sigmoid()[0].numpy()
        )

    def locate(name, size, digest):
        assert name == "index.private.json" and size == 21324
        assert digest == "3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3"
        return index

    codec = (
        "import hashlib,numpy as np\n"
        "def decode(body,digest):\n"
        " assert hashlib.sha256(body).hexdigest()==digest\n"
        " return np.frombuffer(body,dtype=np.uint8).copy(),np.ones(1,dtype=np.uint8)\n"
    )
    namespace = {
        "torch": fake_torch, "Path": lambda _path: tmp_path / "absent-submission.csv",
        "time": time, "json": json, "hashlib": hashlib, "np": np,
        "RaptorClassifier": Model, "ARMS": [{"arch": "synthetic"}], "NORM": "original",
        "_ADAPTED_CODEC": codec, "_ADAPTED_MIXIN": "class CheckpointedEncode: pass\n",
        "_ADAPTED_IMPORTANCE": "class ImportanceReadout: pass\n",
        "_ADAPTED_TRAINING": {"epoch": 3, "source_sha256": "b" * 64},
        "_ADAPTED_SHA": adapter.MODEL_SHA, "_adapted_model_path": tmp_path / "no-model-read",
        "_adapted_locate": locate, "load_model": load_model,
        "build_backbone": lambda *args, **kwargs: None,
        "eval_windows": evaluate, "gpu_windows": prepare_gpu, "infer_pair_probs": infer_pair,
    }
    exec(compile(adapter.PARALLEL_FUNCTION, "actual_prepared_parallel_function", "exec"), namespace)
    return namespace[adapter.FUNCTION_NAME], counts, archives


def test_parallel_execution_preserves_all_checks_and_records_synchronized_timings(tmp_path, monkeypatch):
    run, counts, archives = synthetic_function(tmp_path, monkeypatch)
    result = run()
    assert len(archives) == 2 and len({a.thread for a in archives}) == 2
    assert archives[0].wrapped is not archives[1].wrapped and all(a.closed for a in archives)
    for index in (0, 1):
        assert counts[("load", index)] == counts[("cleanup", index)] == 1
        assert counts[("read", index)] == 32
        assert counts[("reference", index)] == counts[("served", index)] == 64
        for mirror in (False, True):
            assert counts[("CPU_view", index, mirror)] == counts[("GPU_view", index, mirror)] == 32
        record = result["devices"][index]
        assert record["MRI_studies"] == 32
        assert record["maximum_original_class_vs_training_graph_probability_error"] == 0
        assert record["complete_learned_graph_tensor_values_equal"] is True
        assert record["both_original94_input_views_bit_equal"] is True
        assert len(record["per_study_seconds"]) == 32
        assert all(value >= 0 for value in record["per_study_seconds"])
        total = sum(record[key] for key in (
            "startup_seconds", "preparation_seconds",
            "reference_forward_seconds", "serving_forward_seconds",
        ))
        assert 0 <= total <= record["elapsed_seconds"]
    assert result["parallel_device_workers"] == 2 and result["qualified_studies"] == 32
    assert result["runtime_admission_inferred"] is False
    assert result["VM_startup_and_adapted_boot_timing_included"] is False
    assert result["competition_entry"] is False
    assert result["qualification_function_elapsed_seconds"] > 0
    assert not (tmp_path / "absent-submission.csv").exists()


@pytest.mark.parametrize("failure", [
    "pixel_sha", "input_sha", "GPU_input", "learned_state", "nonfinite", "probability",
])
def test_each_original_check_still_fails_and_worker_errors_propagate(tmp_path, monkeypatch, failure):
    run, counts, archives = synthetic_function(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(AssertionError):
        run()
    assert all(a.closed for a in archives)
    assert counts[("load", 0)] == counts[("load", 1)] == 1
    assert counts[("cleanup", 0)] == counts[("cleanup", 1)] == 1
    assert not (tmp_path / "absent-submission.csv").exists()
