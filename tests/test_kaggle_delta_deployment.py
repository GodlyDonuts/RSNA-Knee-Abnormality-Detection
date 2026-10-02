"""Offline base/delta equivalence and whole-examination deployment regressions."""

import copy
import importlib.util
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_competitive_inference import _fixture
from test_kaggle_chunked_inference import chunked
from test_kaggle_submission import workflow
from torch import nn

from rsna.competitive import backbones, infer
from rsna.competitive.adapters import ResidualLinear, configure_encoder_training
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.model import CompetitiveStudyNet
from rsna.schema import TARGETS


class SyntheticBase(nn.Module):
    """A real residual-linear partition, with no large pretrained-model fixture."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 8)
        self.residual = ResidualLinear(nn.Linear(8, 8), 2, 2, 0)
        self.projection.requires_grad_(False)

    def forward(self, value):
        return self.residual(self.projection(value.mean((-1, -2))))


def delta_fixture(tmp_path, monkeypatch, context):
    original, state = _fixture(tmp_path, monkeypatch, size=3, whole=True)
    provenance = {
        "architecture": "dinov3_l16",
        "pretrained": True,
        "weights_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "adapter": dict(
            rank=2, alpha=2, dropout=0, ffn_last_layers=0, gradient_checkpointing=False
        ),
    }
    source_spec = {"architecture": "dinov3_l16", "adapter": provenance["adapter"]}

    def create(spec):
        assert spec == source_spec
        with torch.random.fork_rng():
            torch.manual_seed(45)
            encoder = SyntheticBase()
        return encoder, 8, copy.deepcopy(provenance)

    monkeypatch.setattr(infer, "create_encoder", create)
    monkeypatch.setattr(backbones, "create_encoder", create)
    encoder, dimension, _ = create(source_spec)
    configure_encoder_training(encoder, True)
    model = CompetitiveStudyNet(encoder, dimension, hidden_dim=8, slice_context=context)
    with torch.no_grad():
        encoder.residual.lora_B.fill_(0.15)
    state["binding"]["config"].update(
        encoder=provenance, checkpoint_format="frozen_base_delta", slice_context=context
    )
    state.update(FrozenBaseDelta(model, provenance).checkpoint_model(model))
    torch.save(state, original.checkpoint)
    original.encoder_spec.write_text(json.dumps(source_spec))
    original.scheduled_epoch = True
    original.precision = "fp32"
    full = copy.deepcopy(state)
    full.pop("model_storage")
    full["schema_version"] = 1
    full["model"] = model.state_dict()
    full["binding"]["config"]["checkpoint_format"] = "full"
    full_file = tmp_path / "full-reference.pt"
    torch.save(full, full_file)
    return original, state, full_file, create


@pytest.mark.parametrize("context", ["none", "neighbor3"])
def test_offline_chunked_delta_matches_expanded_full_model_and_direct_delta(
    tmp_path, monkeypatch, context
):
    original, _, full_file, _ = delta_fixture(tmp_path, monkeypatch, context)
    reference = Namespace(**vars(original))
    reference.checkpoint = full_file
    reference.output = tmp_path / "full.csv"
    infer.run(reference)
    expected = pd.read_csv(reference.output)
    infer.run(original)
    np.testing.assert_array_equal(
        pd.read_csv(original.output)[list(TARGETS)], expected[list(TARGETS)]
    )
    args = Namespace(
        checkpoint=original.checkpoint,
        encoder_config=None,
        encoder_spec=original.encoder_spec,
        series=original.series,
        template=original.template,
        cache_root=tmp_path / "chunk-cache",
        output=tmp_path / "chunked.csv",
        dicom_root=None,
        source_revision="synthetic-test",
        chunk_size=2,
        workers=0,
        device="cpu",
        precision="fp32",
        allow_diagnostic=True,
    )

    def invoke(command, *, check):
        assert check and "--encoder-spec" in command and "--encoder-config" not in command

        def option(name):
            return Path(command[command.index(name) + 1])

        cache = option("--cache")
        for uid in pd.read_csv(option("--template")).StudyInstanceUID:
            destination = cache / "test_series" / uid
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(original.cache / "test_series" / uid, destination)
        selected = Namespace(**vars(original))
        for name in ("series", "template", "cache", "output", "encoder_spec"):
            setattr(selected, name, option("--" + name.replace("_", "-")))
        infer.run(selected)

    receipt = chunked.run(args, invoke=invoke)
    actual = pd.read_csv(args.output)
    assert actual.StudyInstanceUID.tolist() == expected.StudyInstanceUID.tolist()
    np.testing.assert_allclose(actual[list(TARGETS)], expected[list(TARGETS)], rtol=0, atol=1e-7)
    assert receipt["chunk_count"] == 2 and receipt["diagnostic_only"] is True
    assert not list(args.cache_root.rglob("*.npz"))


def test_delta_packaging_rejects_a_different_original_base(tmp_path, monkeypatch):
    args, state, _, create = delta_fixture(tmp_path, monkeypatch, "neighbor3")
    spec = json.loads(args.encoder_spec.read_text())
    workflow.validate_delta_base(args.checkpoint, spec, state["binding"])

    def wrong_base(value):
        encoder, dim, provenance = create(value)
        with torch.no_grad():
            encoder.projection.weight.add_(1)
        return encoder, dim, provenance

    monkeypatch.setattr(backbones, "create_encoder", wrong_base)
    with pytest.raises(ValueError, match="frozen base"):
        workflow.validate_delta_base(args.checkpoint, spec, state["binding"])


def test_legacy_delta_packaging_does_not_import_a_later_relation_architecture(
    tmp_path, monkeypatch
):
    import builtins

    args, state, _, _ = delta_fixture(tmp_path, monkeypatch, "neighbor3")
    original_import = builtins.__import__

    def old_source_import(name, *args, **kwargs):
        if name == "rsna.competitive.relations":
            raise ModuleNotFoundError("Absent from the exact legacy training source")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", old_source_import)
    spec = json.loads(args.encoder_spec.read_text())
    workflow.validate_delta_base(args.checkpoint, spec, state["binding"])
    state["binding"]["config"]["relation"] = {"mode": "physical"}
    with pytest.raises(ModuleNotFoundError, match="exact legacy"):
        workflow.validate_delta_base(args.checkpoint, spec, state["binding"])


def test_all_series_deployment_retains_every_healthy_series_beyond_six(tmp_path, monkeypatch):
    study = "1.2.5"
    rows = [
        dict(
            StudyInstanceUID=study,
            SeriesInstanceUID=f"{study}.{n}",
            Anatomical_Plane=("Sagittal", "Coronal", "Axial")[n % 3],
            Fluid_Sensitive=n % 2,
            Fat_Suppression=0,
        )
        for n in range(1, 10)
    ]
    root = tmp_path / "dicoms"
    for row in rows:
        p = root / "test_series" / study / row["SeriesInstanceUID"]
        p.mkdir(parents=True)
        (p / "slice.dcm").write_bytes(b"cache-boundary fixture")
    seen = []

    def cache(source, paths, directory, side, revision):
        seen.append(Path(paths[0]).parts[-2])

    monkeypatch.setattr(chunked, "cache_series", cache)
    selected, receipt = chunked.prepare_usable_series(
        pd.DataFrame(rows),
        [study],
        dicom_root=root,
        cache=tmp_path / "cache",
        cache_side=256,
        max_series=6,
        source_revision="fixture",
        series_sampling="all",
    )
    assert set(selected.SeriesInstanceUID) == set(seen) == {r["SeriesInstanceUID"] for r in rows}
    assert receipt["selected_series"] == 9 and receipt["series_sampling"] == "all"
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"binding": {"config": {"max_series": 6, "series_sampling": "all"}, "cache_side": 256}},
        checkpoint,
    )
    chunked.verify_cache_contract(checkpoint, max_series=6, cache_side=256, series_sampling="all")
    with pytest.raises(ValueError, match="differs from the trained checkpoint"):
        chunked.verify_cache_contract(checkpoint, max_series=6, cache_side=256)


def notebook_module():
    spec = importlib.util.spec_from_file_location(
        "delta_notebook", Path(workflow.__file__).with_name("notebook_entry.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_fixture(tmp_path):
    entry = notebook_module()
    bundle = tmp_path / "dataset"
    bundle.mkdir()
    weights = bundle / "base.safetensors"
    weights.write_bytes(b"synthetic local original bytes")
    config = bundle / "config.json"
    config.write_text("{}")
    spec = {
        "weights_path": weights.name,
        "weights_sha256": entry.sha256(weights),
        "config_path": config.name,
        "config_sha256": entry.sha256(config),
    }
    (bundle / "encoder.json").write_text(json.dumps(spec))
    manifest = {
        "checkpoint_format": "frozen_base_delta",
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": entry.sha256(p)}
            for p in [weights, config]
        },
    }
    working = tmp_path / "working"
    working.mkdir()
    return entry, bundle, working, manifest


def test_offline_notebook_resolves_bundled_weights_without_changing_lineage(tmp_path):
    entry, bundle, working, manifest = notebook_fixture(tmp_path)
    command = entry.encoder_arguments(bundle, working, manifest)
    assert command[0] == "--encoder-spec"
    runtime = json.loads(Path(command[1]).read_text())
    assert Path(runtime["weights_path"]).is_absolute()
    assert Path(runtime["weights_path"]).read_bytes() == (bundle / "base.safetensors").read_bytes()
    assert runtime["weights_sha256"] == entry.sha256(bundle / "base.safetensors")
    manifest["checkpoint_format"] = "full"
    assert entry.encoder_arguments(bundle, working, manifest) == [
        "--encoder-config",
        str((bundle / "config.json").resolve()),
    ]


@pytest.mark.parametrize("role", ["diagnostic", "production", "missing", "eligible"])
def test_probe_entry_never_uses_the_competition_output_path(tmp_path, role):
    entry, bundle, working, manifest = notebook_fixture(tmp_path)
    if role != "missing":
        manifest.update(
            diagnostic_only=role != "production",
            competition_submission_eligible=role == "eligible",
        )
    if role != "diagnostic":
        with pytest.raises(ValueError, match="ineligible delta diagnostic"):
            entry.diagnostic_probe_command(bundle, working, manifest, tmp_path, "python")
        assert not list(working.iterdir())
        return
    command = entry.diagnostic_probe_command(bundle, working, manifest, tmp_path, "python")
    assert "submission.csv" not in " ".join(command)
    assert command[command.index("--output") + 1] == str(working / "delta-diagnostic")
    assert command[command.index("--encoder-spec") + 1] == str(working / "encoder-runtime.json")


@pytest.mark.parametrize("change", ["weights", "path", "source_hash", "manifest_hash"])
def test_offline_notebook_refuses_wrong_base_before_inference(tmp_path, change):
    entry, bundle, working, manifest = notebook_fixture(tmp_path)
    spec = json.loads((bundle / "encoder.json").read_text())
    if change == "weights":
        (bundle / "base.safetensors").write_bytes(b"changed weights")
    elif change == "path":
        spec["weights_path"] = "../unbound.safetensors"
    elif change == "source_hash":
        spec["weights_sha256"] = "f" * 64
    else:
        manifest["files"]["base.safetensors"]["sha256"] = "f" * 64
    (bundle / "encoder.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="encoder bytes/path"):
        entry.encoder_arguments(bundle, working, manifest)
    assert not list(working.iterdir())


@pytest.mark.parametrize("change", [None, "order", "probability", "nan", "out_of_range"])
def test_diagnostic_reference_requires_identical_finite_mri_probabilities(tmp_path, change):
    path = Path(workflow.__file__).with_name("delta_probe.py")
    spec = importlib.util.spec_from_file_location("delta_probe_test", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    frame = pd.DataFrame(
        [
            {"StudyInstanceUID": f"1.2.{n}", **{target: (n + 1) / 4 for target in TARGETS}}
            for n in range(3)
        ],
        columns=["StudyInstanceUID", *TARGETS],
    )
    delta, full = tmp_path / "delta.csv", tmp_path / "reference.csv"
    frame.to_csv(delta, index=False)
    if change == "order":
        frame = frame.iloc[::-1]
    elif change == "probability":
        frame.loc[0, TARGETS[0]] += 0.01
    elif change == "nan":
        frame.loc[0, TARGETS[0]] = float("nan")
    elif change == "out_of_range":
        frame.loc[0, TARGETS[0]] = 1.01
    frame.to_csv(full, index=False)
    if change is None:
        assert probe.compare_outputs(delta, full)["csv_bytes_equal"] is True
    else:
        with pytest.raises(ValueError, match="differs|differ"):
            probe.compare_outputs(delta, full)


@pytest.mark.parametrize(
    "change", [None, "normalization", "backbone_count", "low_space", "existing"]
)
def test_expanded_probe_binds_official_tensors_and_gray_normalization_buffers(
    tmp_path, monkeypatch, change
):
    path = Path(workflow.__file__).with_name("delta_probe.py")
    spec = importlib.util.spec_from_file_location("delta_probe_partition_test", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    original, state, _, _ = delta_fixture(tmp_path, monkeypatch, "neighbor3")
    provenance = copy.deepcopy(state["binding"]["config"]["encoder"])
    provenance["adapter"]["target_modules"] = [f"projection.{n}" for n in range(120)]

    def create(_):
        with torch.random.fork_rng():
            torch.manual_seed(45)
            backbone = nn.Module()
            backbone.residuals = nn.ModuleList(
                [ResidualLinear(nn.Linear(1, 1), 1, 1, 0) for _ in range(120)]
            )
            backbone.other = nn.ParameterList(
                [nn.Parameter(torch.ones(1)) for _ in range(175 + (change == "backbone_count"))]
            )
            encoder = backbones._GrayEncoder(backbone, 16)
        return encoder, 8, copy.deepcopy(provenance)

    monkeypatch.setattr(probe, "create_encoder", create)
    available = (3 if change == "low_space" else 6) * 1024**3
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda _: Namespace(free=available))
    encoder, dimension, _ = create(None)
    configure_encoder_training(encoder, True)
    model = CompetitiveStudyNet(encoder, dimension, hidden_dim=8, slice_context="neighbor3")
    config = state["binding"]["config"]
    config.update(
        encoder=provenance,
        diagnostic_study_limit=16,
        slice_sampling="all",
        series_sampling="all",
        pooling="attention",
        hidden_dim=8,
        encode_chunk_size=64,
    )
    state["step"] = 4
    state.update(FrozenBaseDelta(model, provenance).checkpoint_model(model))
    if change == "normalization":
        state["model_storage"]["frozen_tensors"]["encoder.mean"]["sha256"] = "f" * 64
    torch.save(state, original.checkpoint)
    output = tmp_path / "probe-expanded.pt"
    if change == "existing":
        output.write_bytes(b"prior reference is never replaced")
    if change is not None:
        with pytest.raises(
            ValueError, match="frozen base|partition differs|workload budget|must be new"
        ):
            probe.expanded_reference(original.checkpoint, original.encoder_spec, output)
        if change == "existing":
            assert output.read_bytes() == b"prior reference is never replaced"
        else:
            assert not output.exists()
        return
    result = probe.expanded_reference(original.checkpoint, original.encoder_spec, output)
    assert {key: value for key, value in result.items() if key != "reference_scratch"} == {
        "frozen_base_tensors_verified": 415,
        "frozen_encoder_tensors_verified": 417,
        "normalization_buffers_verified": 2,
        "adapter_modules_verified": 120,
    }
    budget = result["reference_scratch"]
    assert output.stat().st_size <= budget["reference_upper_bound_bytes"]
    assert budget["free_before_reference_bytes"] == available < 22 * 1024**3
    assert budget["required_free_bytes"] <= available
    expanded = torch.load(output, weights_only=True)
    assert expanded["schema_version"] == 1 and "model_storage" not in expanded
    assert expanded["binding"]["config"]["checkpoint_format"] == "full"
    assert expanded["model"].keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expanded["model"][name])


@pytest.mark.parametrize("change", [None, "helper_hash", "chunked_bytes", "eligible"])
def test_private_probe_source_override_reuses_verified_base_without_modifying_dataset(
    tmp_path, change
):
    import hashlib

    entry, bundle, working, manifest = notebook_fixture(tmp_path)
    manifest.update(diagnostic_only=True, competition_submission_eligible=change == "eligible")
    code = bundle / "code"
    code.mkdir()
    chunked = code / "chunked_infer.py"
    chunked.write_text("# pinned shared inference source\n")
    (code / "delta_probe.py").write_text("# preserved old private probe\n")
    manifest["files"]["code/chunked_infer.py"] = {
        "bytes": chunked.stat().st_size,
        "sha256": entry.sha256(chunked),
    }
    source = "# corrected private engineering probe\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    if change == "helper_hash":
        digest = "f" * 64
    if change == "chunked_bytes":
        chunked.write_text("# unbound modified inference\n")
    if change is not None:
        with pytest.raises(ValueError, match="byte-bound|source differs|ineligible"):
            entry.diagnostic_probe_command(
                bundle,
                working,
                manifest,
                tmp_path,
                "python",
                probe_source=source,
                probe_source_sha256=digest,
            )
        assert not list(working.iterdir())
        return
    before = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    command = entry.diagnostic_probe_command(
        bundle,
        working,
        manifest,
        tmp_path,
        "python",
        probe_source=source,
        probe_source_sha256=digest,
    )
    runtime = working / "delta-probe-runtime-code"
    assert Path(command[1]) == runtime / "delta_probe.py"
    assert (runtime / "delta_probe.py").read_text() == source
    assert (runtime / "chunked_infer.py").read_bytes() == chunked.read_bytes()
    assert command[command.index("--output") + 1] == str(working / "delta-diagnostic")
    assert "submission.csv" not in command
    assert {
        p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()
    } == before
