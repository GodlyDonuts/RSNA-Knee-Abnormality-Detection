"""Synthetic checks of the patch-focus admission/recovery wrappers.

These are engineering tests only; they are not real-MRI CPU admission or GPU acceptance.
"""

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from test_patch_focus import D, SLOT_STRATEGY, STRATEGIES, Backbone, window

from rsna.competitive import patch_focus as pf
from rsna.competitive import source_slot as ss
from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import admit_patch_focus_inputs as admit
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import run_patch_focus_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = json.loads((ROOT / dense.ORIGINAL_RECEIPT).read_text())
GIB = 2**30


def current_code():
    return {str(p.relative_to(ROOT / "rsna")): gate.sha(p) for p in (ROOT / "rsna").rglob("*.py")}


def funding(**changes):
    values: dict = dict(
        experiment=admit.EXPERIMENT,
        funded_by="root",
        original_input_receipt_sha256=dense.ORIGINAL_RECEIPT_SHA256,
        merged_cache_receipt_sha256=dense.MERGED_RECEIPT_SHA256,
        new_CPU_source_and_receipt_bytes=dense.CONTROL_BYTES,
        new_GPU_or_checkpoint_bytes=0,
        new_shared_bytes_files=0,
        prefix_cache_bytes=0,
        existing_liabilities_bytes=50 * GIB,
        reserve_bytes=20 * GIB,
    )
    values |= changes
    values.setdefault(
        "required_free_bytes",
        values["new_CPU_source_and_receipt_bytes"]
        + values["existing_liabilities_bytes"]
        + values["reserve_bytes"],
    )
    return values


def test_recipes_and_original_receipt_are_the_tracked_bytes():
    assert gate.sha(ROOT / recovery.REFERENCE) == admit.RECIPE_SHA256["reference"]
    for arm, name in recovery.ARMS.items():
        assert gate.sha(ROOT / recovery.RECIPE / name) == admit.RECIPE_SHA256[arm]
    assert gate.sha(ROOT / dense.ORIGINAL_RECEIPT) == dense.ORIGINAL_RECEIPT_SHA256
    assert set(admit.REQUIRED_SOURCE) >= {admit.SELF, admit.WRAPPER, str(recovery.REFERENCE)}


def test_arms_are_the_admitted_public_recipe_plus_source_slot_and_focus_policy():
    configs = admit.load_arms(ROOT, ORIGINAL["configs"]["public"])
    assert set(configs) == set(admit.NAMES)
    for values in configs.values():
        assert values["encoder"]["adapter"]["rank"] == 32
        assert values["encoder"]["adapter"]["ffn_last_layers"] == 12
        assert (values["epochs"], values["batch_size"] * values["accumulation_steps"]) == (10, 8)
        assert values["fitting_supervision"]["kind"] == "label_source_v1"
    assert configs["control"]["classifier_strategy"]["patch_focus"] == "disabled"
    assert configs["candidate"]["classifier_strategy"]["patch_focus"] == "query_residual"


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.update(epochs=2),
        lambda c: c.update(fold=1),
        lambda c: c["encoder"]["adapter"].update(rank=16),
        lambda c: c["encoder"].update(weights_sha256="0" * 64),
        lambda c: c["fitting_supervision"].update(sha256="0" * 64),
        lambda c: c.update(classifier_strategy={"kind": "source_matched_slot_v1"}),
    ],
)
def test_admitted_public_recipe_drift_refused(edit):
    public = copy.deepcopy(ORIGINAL["configs"]["public"])
    edit(public)
    with pytest.raises(ValueError):
        admit.load_arms(ROOT, public)


@pytest.mark.parametrize(
    "name,edit",
    [
        ("control-r32-config.json", lambda c: c["classifier_strategy"].update(hidden=128)),
        ("control-r32-config.json", lambda c: c["classifier_strategy"].update(patch_focus="x")),
        ("candidate-r32-config.json", lambda c: c.update(head_lr=1e-3)),
        ("candidate-r32-config.json", lambda c: c["encoder"]["adapter"].update(rank=8)),
        ("candidate-r32-config.json", lambda c: c["classifier_strategy"].update(dropout=0.1)),
    ],
)
def test_arm_recipe_tampering_refused(tmp_path, name, edit):
    for path in (recovery.REFERENCE, *(recovery.RECIPE / n for n in recovery.ARMS.values())):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, tmp_path / path)
    values = json.loads((tmp_path / recovery.RECIPE / name).read_text())
    edit(values)
    (tmp_path / recovery.RECIPE / name).write_text(json.dumps(values))
    with pytest.raises(ValueError):
        recovery.load_recipe(tmp_path)


def test_encoder_spec_must_be_the_pinned_mounted_official_spec():
    configs = admit.load_arms(ROOT, ORIGINAL["configs"]["public"])
    spec = {k: v for k, v in configs["control"]["encoder"].items() if k != "adapter"}
    assert admit.encoder_spec(configs, spec)["encoder_spec_sha256"] == gate.canonical(spec)
    for change in ({"weights_path": "/tmp/other.safetensors"}, {"architecture": "dinov3_b16"}):
        with pytest.raises(ValueError):
            admit.encoder_spec(configs, spec | change)


def test_code_delta_is_exact_and_includes_patch_focus():
    delta = admit.code_delta(ORIGINAL["code_sha256"], current_code())
    assert delta["input_label_code_changed"] == ["competitive/train.py", "release_cache.py"]
    assert (delta["changed_files"], delta["added_files"], delta["removed_files"]) == (2, 4, 0)
    for edit in (
        lambda c: c.update({"competitive/patch_focus.py": "0" * 64}),
        lambda c: c.update({"competitive/source_slot.py": "0" * 64}),
        lambda c: c.update({"competitive/public_input.py": "0" * 64}),
        lambda c: c.pop("labels.py"),
    ):
        code = current_code()
        edit(code)
        with pytest.raises(ValueError, match="reviewed delta"):
            admit.code_delta(ORIGINAL["code_sha256"], code)


def test_funding_is_patch_focus_cpu_only():
    assert admit.validate_funding(funding())["experiment"] == admit.EXPERIMENT
    for bad in (
        {"experiment": "dense_suffix_capacity_v1"},
        {"new_GPU_or_checkpoint_bytes": 12 * GIB},
        {"reserve_bytes": GIB},
    ):
        with pytest.raises(ValueError):
            admit.validate_funding(funding(**bad))
    with pytest.raises(ValueError):
        admit.validate_funding({k: v for k, v in funding().items() if k != "experiment"})


# --- Matched initialization and real-input gradient evidence, on a tiny encoder ---------


def _builder(query=None, gates=None):
    original = {"encoder_provenance": {"public": {"tiny": True}}}
    original["initial_learned_state"] = {"public": {"frozen_base_sha256": "base"}}

    def build(name):
        torch.manual_seed(3)
        if name == "reference":
            model = ss.SourceSlotStudyNet(Backbone(), D, SLOT_STRATEGY)
        else:
            model = pf.PatchFocusSlotStudyNet(Backbone(), D, STRATEGIES[name])
            if gates and name in gates:
                model.focus_gate = gates[name]
            if query is not None and name == "candidate":
                with torch.no_grad():
                    model.patch_query.add_(query)
        state = {
            "frozen_base_sha256": "base",
            "post_factory_RNG_sha256": gate.rng_digest(),
            "learned_parameters": sum(p.numel() for p in model.parameters()),
        }
        return model, {"tiny": True}, state

    return build, original


def _probe_inputs():
    inputs = window(batch=1)
    labels = torch.randint(0, 2, (1, 12), generator=torch.Generator().manual_seed(1)).float()
    mask = torch.ones(1, 12)
    mask[:, [2, 7]] = 0
    return inputs, labels, mask


def test_matched_initialization_and_target_routed_gradients(monkeypatch):
    monkeypatch.setattr(recovery, "EMBEDDING_DIM", D)
    build, original = _builder()
    inputs, labels, mask = _probe_inputs()
    summary, _ = admit.matched_initialization(build, inputs, labels, mask, 0, original)
    assert summary["control_gradients_bitwise_reference"]
    assert (summary["candidate_observed_targets"], summary["candidate_masked_targets"]) == (10, 2)
    assert summary["candidate_query_gradient_norm"] > 0
    assert summary["only_extra_parameter"] == "patch_query"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": 0.1},  # candidate query not zero-initialized
        {"gates": {"control": 1.0}},  # control residual enabled
    ],
)
def test_matched_initialization_refuses_nonidentity_arms(monkeypatch, kwargs):
    monkeypatch.setattr(recovery, "EMBEDDING_DIM", D)
    build, original = _builder(**kwargs)
    inputs, labels, mask = _probe_inputs()
    with pytest.raises(ValueError):
        admit.matched_initialization(build, inputs, labels, mask, 0, original)


def test_fully_observed_probe_cannot_claim_masked_gradient_evidence(monkeypatch):
    monkeypatch.setattr(recovery, "EMBEDDING_DIM", D)
    build, original = _builder()
    inputs, labels, _ = _probe_inputs()
    with pytest.raises(ValueError, match="observed targets"):
        admit.matched_initialization(build, inputs, labels, torch.ones(1, 12), 0, original)


def test_gradient_study_needs_observed_and_masked_fitting_targets():
    import pandas as pd

    from rsna.schema import TARGETS

    full = {f"{t}__mask": [1.0, 1.0] for t in TARGETS}
    frame = pd.DataFrame(full)
    with pytest.raises(ValueError):
        admit.gradient_study(frame)
    frame.loc[1, f"{TARGETS[3]}__mask"] = 0.0
    assert admit.gradient_study(frame) == 1


# --- GPU wrapper comparison helpers ---------------------------------------------------


def _row(loss=0.5, norm=2.0, encoder=1.0):
    return {"loss": loss, "gradient_norm": norm, "encoder_gradient_norm": encoder}


def test_first_update_must_match_the_source_slot_fit():
    rows = {name: [_row()] for name in admit.NAMES}
    assert recovery.compare_first_update(rows)["control_gradient_norm_bitwise_reference"]
    for name, change in (("candidate", {"loss": 0.6}), ("control", {"norm": 2.1})):
        bad = dict(rows, **{name: [_row(**change)]})
        with pytest.raises(ValueError):
            recovery.compare_first_update(bad)
    with pytest.raises(ValueError):
        recovery.compare_first_update(dict(rows, control=[_row(encoder=0.0)]))


def test_bindings_may_differ_only_in_focus_policy():
    def binding(strategy):
        return {"config": {"seed": 1, "classifier_strategy": dict(strategy)}, "code_sha256": {}}

    bindings = {
        "reference": binding(SLOT_STRATEGY),
        **{arm: binding(STRATEGIES[arm]) for arm in recovery.ARMS},
    }
    recovery.compare_matched(bindings)
    bad = copy.deepcopy(bindings)
    bad["candidate"]["config"]["seed"] = 2
    with pytest.raises(ValueError):
        recovery.compare_matched(bad)
    bad = copy.deepcopy(bindings)
    bad["control"]["config"]["classifier_strategy"]["hidden"] = 128
    with pytest.raises(ValueError):
        recovery.compare_matched(bad)


def _checkpoint(query):
    names = [
        f"encoder.backbone.m{i}.{s}"
        for i in range(recovery.TARGET_MODULES)
        for s in ("lora_A", "lora_B")
    ]
    each = recovery.ENCODER_TRAINABLE // len(names)
    last = recovery.ENCODER_TRAINABLE - each * (len(names) - 1)
    encoder = {n: torch.zeros(1, last if n == names[-1] else each) for n in names}
    head = {"head.out.weight": torch.zeros(12, 256)}
    storage = {"frozen_tensors": {"encoder.backbone.w": "x"}}
    reference = {"model": encoder | head, "model_storage": storage}
    arm = {"model": encoder | head | {"patch_query": query}, "model_storage": storage}
    return reference, arm


def test_graph_is_source_slot_plus_only_the_query():
    reference, arm = _checkpoint(torch.zeros(12, 1024))
    assert sum(v.numel() for k, v in arm["model"].items() if k.startswith("encoder.")) == (
        recovery.ENCODER_TRAINABLE
    )
    assert recovery.check_graph(arm, reference)["patch_query_parameters"] == 12 * 1024
    extra = copy.deepcopy(arm)
    extra["model"]["head.extra"] = torch.zeros(1)
    for bad in (extra, _checkpoint(torch.zeros(12, 512))[1]):
        with pytest.raises(ValueError):
            recovery.check_graph(bad, reference)


def test_control_query_never_moves_and_candidate_query_must():
    zero, moved = {"model": {"patch_query": torch.zeros(12, 4)}}, {
        "model": {"patch_query": torch.ones(12, 4)}
    }
    assert recovery.check_query("control", zero)["patch_query_rows_moved"] == 0
    assert recovery.check_query("candidate", moved)["patch_query_rows_moved"] == 12
    with pytest.raises(ValueError):
        recovery.check_query("control", moved)
    with pytest.raises(ValueError):
        recovery.check_query("candidate", zero)


def test_added_resources_are_measured_against_source_slot():
    reference = {"elapsed_seconds": 100.0, "cuda_peak_memory_bytes": 10 * GIB}
    arm = {"elapsed_seconds": 120.0, "cuda_peak_memory_bytes": 11 * GIB}
    assert recovery.added(arm, reference) == {
        "added_elapsed_seconds": 20.0,
        "elapsed_ratio": 1.2,
        "added_cuda_peak_memory_bytes": GIB,
    }


def test_output_cap_cannot_exceed_its_explicit_bound(tmp_path):
    args = recovery.parser().parse_args(
        [
            *(f"--{n}={tmp_path}" for n in ("source", "encoder-spec", "artifact-root", "cache")),
            f"--source-manifest={tmp_path / 'm.json'}",
            f"--overlay={tmp_path}",
            f"--output={tmp_path / 'out'}",
            "--remaining-seconds=10",
        ]
    )
    assert args.max_output_bytes == recovery.OUTPUT_CAP_BYTES
