"""Owning-node CPU wrapper guards and bounded local original-object transport."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna.competitive import public_input_cache as pc

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_public_triplet_cache", ROOT / "scripts/newton/run_public_triplet_cache.py"
)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

CAPS = dict(
    max_study_objects=4000,
    max_study_source_bytes=1 << 30,
    max_total_source_bytes=1 << 40,
    max_member_bytes=9_000_000,
    max_cache_bytes=32 << 30,
    reserve_bytes=20 << 30,
)


def binding():
    return dict(
        node="evc31",
        git_head="b" * 40,
        files={name: wrapper.sha256(ROOT / name) for name in wrapper.MODULES},
        runtime={name: "1" for name in wrapper.RUNTIME},
        inputs={name: "0" * 64 for name in wrapper.INPUTS},
        caps=CAPS,
        expected_counts=[2247, 1156],
        fitting_uids_sha256="1" * 64,
        evaluation_uids_sha256="2" * 64,
        source_revision="0b5e410288232f570c69ba4f44723356e6ae6ab7",
        raw_source_revision="0b5e410288232f570c69ba4f44723356e6ae6ab7",
    )


def write(tmp_path, value):
    path = tmp_path / "binding.json"
    path.write_bytes(json.dumps(value).encode())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_binding_is_sha_pinned_and_complete(tmp_path):
    path, digest = write(tmp_path, binding())
    assert wrapper.load_binding(path, digest)["node"] == "evc31"
    with pytest.raises(ValueError):
        wrapper.load_binding(path, "0" * 64)
    for change in (
        {"expected_counts": [2248, 1155]},
        {"files": {}},
        {"extra": 1},
        {"inputs": {"source": "0" * 64}},
    ):
        path, digest = write(tmp_path, binding() | change)
        with pytest.raises(ValueError):
            wrapper.load_binding(path, digest)


ENV = {"OMP_NUM_THREADS": "1", "SLURM_CPUS_PER_TASK": "1"}


@pytest.mark.parametrize(
    "environ,node,cuda",
    [
        (ENV, "evc34", False),
        (ENV, "evc31", True),
        (ENV | {"CUDA_VISIBLE_DEVICES": "0"}, "evc31", False),
        (ENV | {"SLURM_JOB_GPUS": "3"}, "evc31", False),
        (ENV | {"SLURM_SUBMIT_COMMAND": "sbatch"}, "evc31", False),
        ({"OMP_NUM_THREADS": "8"}, "evc31", False),
        (ENV | {"SLURM_CPUS_PER_TASK": "2"}, "evc31", False),
    ],
)
def test_environment_refuses_gpu_other_node_dispatch_or_threads(environ, node, cuda):
    with pytest.raises(ValueError):
        wrapper.check_environment(binding(), environ, node, cuda)


def test_cpu_only_owning_node_environment_admitted():
    wrapper.check_environment(binding(), ENV | {"CUDA_VISIBLE_DEVICES": ""}, "evc31", False)


def test_runtime_and_source_bound():
    value = binding()
    wrapper.check_runtime(value, value["runtime"])
    with pytest.raises(ValueError):
        wrapper.check_runtime(value, value["runtime"] | {"torch": "2"})
    wrapper.check_source(value, ROOT, lambda root: "b" * 40)
    with pytest.raises(ValueError):
        wrapper.check_source(value, ROOT, lambda root: "dirty")
    value["files"]["rsna/competitive/public_input.py"] = "0" * 64
    with pytest.raises(ValueError):
        wrapper.check_source(value, ROOT, lambda root: "b" * 40)
    wrapper.check_loaded_modules(ROOT)


def test_inputs_bound_by_bytes(tmp_path):
    value, paths = binding(), {}
    for name in wrapper.INPUTS:
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
        value["inputs"][name] = wrapper.sha256(paths[name])
    wrapper.check_inputs(value, paths)
    paths["folds"].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        wrapper.check_inputs(value, paths)


def test_storage_envelope_private_fresh_and_capped(tmp_path):
    value = binding()
    plenty = lambda path: SimpleNamespace(free=1 << 50)  # noqa: E731
    output, scratch = tmp_path / "cache", tmp_path / "scratch"
    wrapper.check_storage(value, output, scratch, False, plenty)
    with pytest.raises(ValueError):
        wrapper.check_storage(value, ROOT / "cache", scratch, False, plenty)
    with pytest.raises(ValueError):
        wrapper.check_storage(value, Path("relative"), scratch, False, plenty)
    tight = lambda path: SimpleNamespace(free=CAPS["max_cache_bytes"])  # noqa: E731
    with pytest.raises(ValueError):
        wrapper.check_storage(value, output, scratch, False, tight)
    output.mkdir()
    with pytest.raises(ValueError):
        wrapper.check_storage(value, output, scratch, False, plenty)
    wrapper.check_storage(value, output, scratch, True, plenty)
    scratch.mkdir()
    with pytest.raises(ValueError):
        wrapper.check_storage(value, output, scratch, True, plenty)


def test_inventory_objects_are_native_python_records(tmp_path):
    import pandas as pd

    frame = pd.DataFrame(
        [
            dict(
                split="train",
                study_uid=study,
                series_uid=f"{study}.1",
                sop_uid=f"{study}.1.1",
                path=f"train_series/{study}/{study}.1/{study}.1.1.dcm",
                sha256="a" * 64,
                bytes=10,
                storage="lfs",
            )
            for study in ("1.2.3", "1.2.4", "1.2.5")
        ]
    )
    frame.to_parquet(tmp_path / "inventory.parquet")
    objects = wrapper.inventory_objects(tmp_path / "inventory.parquet", {"1.2.3", "1.2.5"})
    assert sorted(objects) == ["1.2.3", "1.2.5"]
    assert type(objects["1.2.3"][0]["bytes"]) is int
    caps = pc.Caps(**CAPS)
    assert pc.validate_objects(objects, {"1.2.3", "1.2.5"}, caps)["objects"] == 2


def original(root, payload=b"dicom" * 50):
    path = root / "train_series/1.2/1.2.3/1.2.3.4.dcm"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return dict(
        path="train_series/1.2/1.2.3/1.2.3.4.dcm",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_local_transport_copies_exact_bounded_bytes(tmp_path):
    record = original(tmp_path / "raw")
    transport = pc.LocalTransport(tmp_path / "raw", block=16)
    transport.fetch(record, tmp_path / "copy.dcm")
    assert (tmp_path / "copy.dcm").read_bytes() == b"dicom" * 50
    assert transport.objects == 1 and transport.bytes == 250


@pytest.mark.parametrize("mode", ["larger", "smaller", "sha", "missing", "symlink", "escape"])
def test_local_transport_failures_leave_nothing(tmp_path, mode):
    raw = tmp_path / "raw"
    record = original(raw)
    source = raw / record["path"]
    if mode == "larger":
        source.write_bytes(source.read_bytes() + b"x")
    elif mode == "smaller":
        source.write_bytes(source.read_bytes()[:-1])
    elif mode == "sha":
        source.write_bytes(b"x" + source.read_bytes()[1:])
    elif mode == "missing":
        source.unlink()
    elif mode == "symlink":
        outside = tmp_path / "outside.dcm"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    else:
        record = record | {"path": "train_series/../../outside.dcm"}
    target = tmp_path / "copy.dcm"
    with pytest.raises(pc.TransportFailure) as failure:
        pc.LocalTransport(raw, block=16).fetch(record, target)
    assert "1.2.3" not in str(failure.value) and str(tmp_path) not in str(failure.value)
    assert not target.exists()


def test_local_transport_never_overwrites(tmp_path):
    record = original(tmp_path / "raw")
    target = tmp_path / "copy.dcm"
    target.write_bytes(b"prior")
    with pytest.raises(FileExistsError):
        pc.LocalTransport(tmp_path / "raw").fetch(record, target)
    assert target.read_bytes() == b"prior"
