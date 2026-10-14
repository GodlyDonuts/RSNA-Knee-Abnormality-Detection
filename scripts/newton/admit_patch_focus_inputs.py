"""Owning-node actual-input CPU admission for the matched source patch-focus arms.

One CPU-only, zero-GPU step on the original owning node. It reuses the original
public-input CPU receipt, root's complete-cache receipt and the whole-byte official
DINOv3-L proof (fresh file-stat binding only), revalidates the complete 2247/1156
fold/release/supervision/cohort boundary, replays the 16 fitting / 16 development
public tensors through the staged loader code, and builds the reviewed source-slot
r32 model plus both patch-focus arms in the canonical trainer seed/factory order.

On one real fitting-only MRI study it runs one train-mode forward/backward per model
on CPU. Naturally mixed masks are used when present; otherwise a separately recorded
backward-only cloned-mask negative control hides the last target without changing
the accepted fitting frame, original dataset batch mask, or production supervision. It
and requires: bitwise-identical initial logits, encoder/head values and RNG; only a
zero ``patch_query`` extra; control gradients bitwise the source-slot gradients with
a zero query gradient; candidate query gradients finite and nonzero exactly on the
observed targets and exactly zero on masked targets.

Nothing here steps an optimizer, reserves a GPU, downloads, renders, transfers or
rehashes the cohort. One sanitized receipt (hashes/counts only) is written
exclusively. It does not accept GPU recovery; root must still fund and run
``run_patch_focus_recovery.py`` once. Engineering evidence only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import torch

from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import run_patch_focus_recovery as recovery

STATUS = "source_patch_focus_actual_inputs_and_matched_initialization_gradients_CPU_accepted"
EXPERIMENT = "source_patch_focus_v1"
NAMES = ("reference", *recovery.ARMS)
WRAPPER = "scripts/newton/run_patch_focus_recovery.py"
SELF = "scripts/newton/admit_patch_focus_inputs.py"
RECIPE_SHA256 = {
    "reference": "e2bfb43c4577dd78c13be3958e99c41ca250c9a3b7251236ddbff8a380fa99de",
    "control": "efdca23d70f175b2aff79583e7ef1222b14e71453c1432af12828e9084b94c29",
    "candidate": "b402ed54f7773a58dfd8774fda34c7a17f0963d022dd5c7109d5f1e7b930dae5",
}
REQUIRED_SOURCE = (
    SELF,
    WRAPPER,
    "scripts/newton/admit_dense_suffix_inputs.py",
    "scripts/newton/run_dense_suffix_recovery.py",
    "scripts/newton/probe_public_input_recipe_inputs.py",
    "scripts/newton/probe_patch_witness_trainer_recovery.py",
    "scripts/newton/enforce_deterministic_runtime.py",
    "tools/research/public_dinov2_input_reference/dino_adapter.py",
    dense.ORIGINAL_RECEIPT,
    str(recovery.REFERENCE),
    *(f"{recovery.RECIPE}/{name}" for name in recovery.ARMS.values()),
)
# Exact rsna/ delta since the original receipt's code (reviewed source-slot,
# window-bag, weight-average, profiler/archive and patch-focus integration). Anything else is refused.
CODE_DELTA = {
    "changed": {
        "archive.py": "8d746f111bf9b7a5ed9c194a8edddde5b61f7871d8c4534b9b5b0585921c3447",
        "competitive/throughput.py": (
            "d1b2cc9d81d660811d6065915104dbfec07c41e627c237a4361bd62b5f576b19"
        ),
        "competitive/train.py": "5a176badbfa074384f149ca985da768d80eb1ccc0eaa8f379e734bc17045dc44",
    },
    "added": {
        "competitive/patch_focus.py": (
            "5df0cd272b8308cac50d0b0606883a796e5ad834e151e69069547a87d466d522"
        ),
        "competitive/weight_average.py": (
            "c50d0427e6ff1369aa4f60a91dd3c8472150d516dad241a83ad9ffe1e9c15253"
        ),
        "competitive/window_bag.py": (
            "d99ae8e86d0d008c016a27e73cbee186141db230f056113a2246e0e26c22aedb"
        ),
        "release_cache.py": "83660e78e3b6ce745bbf89275157474f97290a43bb0c00a4a3da687352f38b6a",
    },
    "removed": [],
}
QUERY = recovery.QUERY
CONTROL_BYTES = dense.CONTROL_BYTES
RESERVE_BYTES = dense.RESERVE_BYTES
RECOVERY_OUTPUT_CAP_BYTES = recovery.OUTPUT_CAP_BYTES


def validate_funding(funding: Mapping) -> dict:
    """Patch-focus CPU control bytes only; no GPU/checkpoint bytes, liabilities retained."""
    if not isinstance(funding, Mapping) or funding.get("experiment") != EXPERIMENT:
        raise ValueError("Funding must be declared for the patch-focus experiment itself")
    dense.validate_funding({k: v for k, v in funding.items() if k != "experiment"})
    return dict(funding)


def verify_source(code: Path, manifest: Mapping) -> dict[str, str]:
    """Exact staged files; every rsna module bound; no links, escapes or extras."""
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if (
        set(manifest) != {"source_git", "files"}
        or not isinstance(manifest["source_git"], str)
        or len(manifest["source_git"]) != 40
        or not isinstance(files, Mapping)
        or not set(REQUIRED_SOURCE) <= set(files)
    ):
        raise ValueError("Source manifest lacks the exact admission/wrapper/recipe binding")
    package = {str(p.relative_to(code)) for p in (code / "rsna").rglob("*.py") if p.is_file()}
    if package != {name for name in files if name.startswith("rsna/")}:
        raise ValueError("Source manifest must bind exactly every canonical rsna module")
    total = 0
    for name, value in files.items():
        path = code / name
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.is_file()
            or not gate._is_sha(value)
        ):
            raise ValueError("Unsafe or missing staged source file")
        total += path.stat().st_size
        if gate.sha(path) != value:
            raise ValueError("Staged source differs from its manifest")
    if total > dense.SOURCE_CAP_BYTES:
        raise ValueError("Staged source exceeds its explicit cap")
    recipes = {
        "reference": str(recovery.REFERENCE),
        **{arm: f"{recovery.RECIPE}/{name}" for arm, name in recovery.ARMS.items()},
    }
    if any(files[path] != RECIPE_SHA256[name] for name, path in recipes.items()):
        raise ValueError("Predeclared source-slot or patch-focus recipe changed")
    if files[dense.ORIGINAL_RECEIPT] != dense.ORIGINAL_RECEIPT_SHA256:
        raise ValueError("Original public-input CPU receipt changed")
    return {n.removeprefix("rsna/"): h for n, h in files.items() if n.startswith("rsna/")}


def code_delta(original: Mapping[str, str], current: Mapping[str, str]) -> dict:
    delta = {
        "changed": {
            k: current[k] for k in sorted(current) if k in original and original[k] != current[k]
        },
        "added": {k: current[k] for k in sorted(current) if k not in original},
        "removed": sorted(k for k in original if k not in current),
    }
    if delta != CODE_DELTA:
        raise ValueError("Staged code differs from the original receipt beyond the reviewed delta")
    touched = sorted(dense.INPUT_LABEL_CODE & (set(delta["changed"]) | set(delta["added"])))
    return {
        "code_delta_sha256": gate.canonical(delta),
        "changed_files": len(delta["changed"]),
        "added_files": len(delta["added"]),
        "removed_files": len(delta["removed"]),
        "input_label_code_changed": touched,
        "input_label_code_unchanged_sha256": gate.canonical(
            {
                k: current[k]
                for k in sorted(dense.INPUT_LABEL_CODE & set(current))
                if k not in touched
            }
        ),
    }


def load_arms(code: Path, original_public: Mapping) -> dict[str, dict]:
    """Reference equals the admitted public recipe plus only the source-slot classifier."""
    configs = recovery.load_recipe(code)
    reference = {k: v for k, v in configs["reference"].items() if k != "classifier_strategy"}
    admitted = {k: v for k, v in original_public.items() if k != "classifier_strategy"}
    if reference != admitted or original_public.get("classifier_strategy") is not None:
        raise ValueError("Source-slot recipe differs from the admitted public recipe")
    return configs


def encoder_spec(configs: Mapping[str, dict], spec: Mapping) -> dict:
    for name in NAMES:
        recovery.engineering_config(configs[name], spec)
        if dict(spec) != {k: v for k, v in configs[name]["encoder"].items() if k != "adapter"}:
            raise ValueError("Encoder spec differs from the pinned recipe paths")
    return {"encoder_spec_sha256": gate.canonical(dict(spec))}


def gradient_study(fitting) -> int:
    """First fitting row with both observed and masked fitting targets (labels only)."""
    from rsna.schema import TARGETS

    masks = fitting[[f"{t}__mask" for t in TARGETS]].to_numpy()
    for index, row in enumerate(masks):
        if 0 < (row > 0).sum() < len(TARGETS):
            return index
    raise ValueError("No fitting study has both observed and masked targets")


def routing_probe(fitting) -> tuple[int, torch.Tensor, dict]:
    """A separate backward-only mask fixture; never changes fitting supervision.

    The public source supplies either twelve labels or none. When no naturally
    mixed row exists, remove the last observed target ONLY in a cloned probe mask
    to exercise the masked-gradient negative control. No optimizer uses this mask.
    """
    from rsna.schema import TARGETS

    masks = torch.tensor(fitting[[f"{t}__mask" for t in TARGETS]].to_numpy())
    if (
        masks.ndim != 2
        or masks.shape[1] != len(TARGETS)
        or not bool(((masks == 0) | (masks == 1)).all())
    ):
        raise ValueError("Routing fixture requires original binary fitting masks")
    counts = masks.sum(1)
    natural = torch.nonzero((counts > 0) & (counts < len(TARGETS))).flatten()
    if len(natural):
        position = int(natural[0])
        method = "unchanged_naturally_mixed_fitting_mask"
        removed = None
    else:
        eligible = torch.nonzero(counts >= 2).flatten()
        if not len(eligible):
            raise ValueError("No fitting row supports an observed/masked routing fixture")
        position = int(eligible[0])
        method = "backward_only_cloned_mask_negative_control"
        removed = int(torch.nonzero(masks[position] > 0).flatten()[-1])
    probe_mask = masks[position : position + 1].clone().float()
    if removed is not None:
        probe_mask[0, removed] = 0
    return (
        position,
        probe_mask,
        {
            "method": method,
            "removed_target_index": removed,
            "original_observed_targets": int(counts[position]),
            "probe_observed_targets": int(probe_mask.sum()),
            "accepted_fitting_masks_changed": False,
            "optimizer_steps": 0,
        },
    )


def _values(model) -> dict:
    records = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            array = parameter.detach().contiguous().numpy()
            records[name] = [list(array.shape), str(array.dtype), hashlib.sha256(array).hexdigest()]
    return records


def probe(model, inputs: tuple, labels, mask, seed: int) -> dict:
    """One canonical train-mode forward and effective-batch backward, no optimizer step."""
    from rsna.competitive.loss import effective_batch_loss, observed_counts

    model.train()
    values = _values(model)
    torch.manual_seed(seed)
    logits = model(*inputs)
    rng = gate.rng_digest()
    loss = effective_batch_loss(logits, labels, mask, observed_counts([mask]))
    loss.backward()
    gradients = {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    return {
        "values": values,
        "logits": logits.detach().clone(),
        "post_forward_RNG_sha256": rng,
        "loss": float(loss.detach()),
        "gradients": gradients,
    }


def check_evidence(evidence: Mapping[str, dict], mask: torch.Tensor) -> dict:
    """Identity of initialization/logits/control gradients; target-routed query gradients."""
    reference, control, candidate = (evidence[name] for name in NAMES)
    for arm in recovery.ARMS:
        values = dict(evidence[arm]["values"])
        query = values.pop(QUERY, None)
        if (
            values != reference["values"]
            or query is None
            or query[0] != [mask.shape[1], recovery.EMBEDDING_DIM]
            or set(evidence[arm]["gradients"]) - set(reference["gradients"]) != {QUERY}
        ):
            raise ValueError("Arm initialization differs from source-slot plus a patch query")
        if not torch.equal(evidence[arm]["logits"], reference["logits"]) or (
            evidence[arm]["post_forward_RNG_sha256"] != reference["post_forward_RNG_sha256"]
        ):
            raise ValueError("Zero-query initial logits or RNG differ from source-slot")
    if QUERY in reference["values"]:
        raise ValueError("Reference unexpectedly owns a patch query")
    if any(
        not torch.equal(control["gradients"][k], g) for k, g in reference["gradients"].items()
    ) or bool(control["gradients"][QUERY].count_nonzero()):
        raise ValueError("Control gradients differ from source-slot or its query moved")
    encoder = [g for k, g in reference["gradients"].items() if k.startswith("encoder.")]
    norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in encoder]))
    if not encoder or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Real-input adapter gradients are zero or nonfinite")
    query = candidate["gradients"][QUERY]
    observed = (mask > 0).any(0)
    if (
        not torch.isfinite(query).all()
        or not bool(observed.any())
        or bool(observed.all())
        or not bool((query[observed].abs().sum(1) > 0).all())
        or bool(query[~observed].count_nonzero())
    ):
        raise ValueError("Candidate query gradients are not routed only to observed targets")
    difference = max(
        float((candidate["gradients"][k] - g).abs().max())
        for k, g in reference["gradients"].items()
    )
    return {
        "initial_values_sha256": gate.canonical(reference["values"]),
        "initial_logits_sha256": hashlib.sha256(reference["logits"].numpy()).hexdigest(),
        "initial_logits_bitwise_equal": True,
        "post_forward_RNG_equal": True,
        "only_extra_parameter": QUERY,
        "patch_query_zero_initialized": True,
        "first_microbatch_loss": reference["loss"],
        "reference_encoder_gradient_norm": float(norm),
        "control_gradients_bitwise_reference": True,
        "control_query_gradient_zero": True,
        "candidate_observed_targets": int(observed.sum()),
        "candidate_masked_targets": int((~observed).sum()),
        "candidate_observed_query_gradients_finite_nonzero": True,
        "candidate_masked_query_gradients_exactly_zero": True,
        "candidate_query_gradient_norm": float(query.norm()),
        "candidate_shared_gradient_max_abs_difference": difference,
    }


def matched_initialization(
    build: Callable[[str], tuple], inputs: tuple, labels, mask, seed: int, original: Mapping
) -> tuple[dict, dict]:
    """Fresh reference/control/candidate in trainer order, one probe each, then released."""
    evidence, states = {}, {}
    for name in NAMES:
        model, provenance, state = build(name)
        if provenance != original["encoder_provenance"]["public"] or (
            state["frozen_base_sha256"]
            != original["initial_learned_state"]["public"]["frozen_base_sha256"]
        ):
            raise ValueError("Official encoder provenance or frozen base changed")
        if name != "reference" and bool(dict(model.named_parameters())[QUERY].count_nonzero()):
            raise ValueError("Patch query is not zero-initialized")
        states[name] = state
        evidence[name] = probe(model, inputs, labels, mask, seed)
        del model
        gc.collect()
    rng = {state["post_factory_RNG_sha256"] for state in states.values()}
    if len(rng) != 1:
        raise ValueError("Factory RNG differs between source-slot and patch-focus arms")
    summary = check_evidence(evidence, mask)
    summary["post_factory_RNG_sha256"] = rng.pop()
    summary["learned_parameters"] = {name: states[name]["learned_parameters"] for name in NAMES}
    return summary, states


def future_obligations(configs: Mapping[str, dict], learned: Mapping[str, int]) -> dict:
    """Separate patch-focus GPU/checkpoint obligations; none are funded or reserved."""
    production = {
        arm: (configs[arm]["epochs"] // configs[arm]["submission_every_epochs"] + 1)
        * 3
        * 4
        * learned[arm]
        + dense.GIB
        for arm in recovery.ARMS
    }
    return {
        "GPU_recovery": {
            "wrapper": WRAPPER,
            "runs": {
                "reference": ["full"],
                **{arm: ["full", "stop1", "resume"] for arm in recovery.ARMS},
            },
            "sequential_on_one_assigned_GPU": True,
            "extra_GPU_node": False,
            "output_cap_bytes": RECOVERY_OUTPUT_CAP_BYTES,
            "reserve_bytes": RESERVE_BYTES,
            "dense_engineering_12GiB_reused_as_funding": False,
            "funded": False,
        },
        "production_fixed10": {
            "arms": list(recovery.ARMS),
            "checkpoint_upper_bound_bytes": production,
            "requires_accepted_GPU_recovery_first": True,
            "funded": False,
        },
    }


def run(root: Path) -> dict:
    from rsna.competitive import backbones
    from rsna.competitive.data import collate_studies
    from rsna.competitive.folds import load_folds
    from rsna.competitive.patch_focus import classifier_model
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from rsna.competitive.source_slot import training_inputs
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development, verified_job_release
    from rsna.schema import TARGETS

    gate.check_placement()
    if torch.cuda.is_available() or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("Genuine one-CPU zero-GPU admission required")
    torch.set_num_threads(1)
    code = root / "code"
    manifest_path = root / "source.json"
    manifest = json.loads(manifest_path.read_text())
    code_sha256 = verify_source(code, manifest)
    funding_path = root / "funding.json"
    funding = validate_funding(json.loads(funding_path.read_text()))
    observed_free = dense.check_storage(root, funding)
    recovery_manifest_path = root / "recovery-source-manifest.json"
    if json.loads(recovery_manifest_path.read_text()) != manifest["files"]:
        raise ValueError("GPU wrapper source manifest differs from the admitted source")
    if recovery.verify_source(code, manifest["files"]) != code_sha256:
        raise ValueError("GPU wrapper would bind different numerical source")
    original = json.loads((code / dense.ORIGINAL_RECEIPT).read_text())
    inputs = dense.validate_original(original)
    delta = code_delta(original["code_sha256"], code_sha256)
    configs = load_arms(code, original["configs"]["public"])
    spec_path = root / "encoder-spec.json"
    spec = encoder_spec(configs, json.loads(spec_path.read_text()))

    config = configs["reference"]
    frame, binding = verified_job_release(
        Path(inputs["studies"]), Path(inputs["artifact-root"]), 0, 1
    )
    folds_path = Path(inputs["folds"])
    folds = load_folds(folds_path, frame, binding)
    fitting, outer = select_development(frame, folds, config["fold"], config["fit_scope"])
    if (len(fitting), len(outer)) != (gate.FITTING_STUDIES, gate.OUTER_STUDIES) or set(
        fitting.leakage_group
    ) & set(outer.leakage_group):
        raise ValueError("Complete fitting/evaluation population or leakage boundary changed")
    fitting, supervision = apply_fitting_supervision(
        fitting,
        outer,
        Path(inputs["overlay"]),
        config["fitting_supervision"],
        release_sha256=binding["manifest_sha256"],
        folds_sha256=gate.sha(folds_path / "fold.csv"),
        fold=config["fold"],
    )
    if supervision != original["supervision"]:
        raise ValueError("Fitting labels/masks/fold supervision differ from the original receipt")
    cache = Path(gate.PUBLIC_CACHE)
    policy = config["public_input"]
    cohort = verify_public_cohort(cache, policy, fitting, outer, binding["source_revision"])
    if cohort != original["public_cohort"]:
        raise ValueError("Complete public cohort membership differs from the original receipt")
    receipt_path = Path(gate.MERGED_RECEIPT)
    if gate.sha(receipt_path) != dense.MERGED_RECEIPT_SHA256:
        raise ValueError("Root complete-cache verification receipt changed")
    gate.validate_merged_receipt(json.loads(receipt_path.read_text()), cohort, cache)
    series_bytes = Path(inputs["series"]).read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != binding["series_sha256"]:
        raise ValueError("Series sampling table differs from the accepted release")
    import pandas as pd

    series = pd.read_csv(io.BytesIO(series_bytes))
    limit = gate.ENGINEERING_STUDIES
    parts = {"fitting": fitting.head(limit), "development": outer.head(limit)}
    common = {
        name: config[name]
        for name in ("slices", "side", "max_series", "slice_sampling", "series_sampling")
    }

    def dataset(part, soft: bool):
        return PublicTripletStudyDataset(
            part,
            series,
            cache,
            public_input=policy,
            source_revision=binding["source_revision"],
            **common,
            soft_targets=soft,
        )

    proofs: dict[str, dict] = {}
    windows: list[torch.Tensor] = []
    for role, part in parts.items():
        data = dataset(part, role == "fitting")
        proofs[role], first = gate.tensor_proofs(part, data, "public")
        windows.append(first)
        del data
        gc.collect()
    dense.check_replay(proofs, {role: _uid_hash(part) for role, part in parts.items()}, original)

    # Real fitting-only MRI study for the gradient probe; development rows are never read.
    position, routing_mask, routing_fixture = routing_probe(fitting)
    item = dataset(fitting.iloc[position : position + 1], True)[0]
    batch = collate_studies([item])
    batch["study_uid"] = list(fitting.study_uid.iloc[position : position + 1])
    seed = config["seed"]
    probe_inputs = training_inputs(
        batch, torch.device("cpu"), config["classifier_strategy"], seed=seed, epoch=0, update=0
    )
    labels, mask = batch["labels"].float(), batch["label_mask"].float()
    original_mask = mask.clone()
    fitting_mask = torch.tensor(
        fitting.iloc[position : position + 1][[f"{t}__mask" for t in TARGETS]].to_numpy(),
        dtype=torch.float32,
    )
    if not torch.equal(original_mask, fitting_mask):
        raise ValueError("Dataset mask differs from the admitted fitting row")
    mask = routing_mask
    probe_study = {
        "fitting_position": position,
        "images_sha256": hashlib.sha256(item["images"].numpy().tobytes()).hexdigest(),
        "labels_and_mask_sha256": hashlib.sha256(
            labels.numpy().tobytes() + original_mask.numpy().tobytes()
        ).hexdigest(),
        "routing_fixture": routing_fixture,
        "routing_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
        "acquired_slots": int(batch["series_mask"].sum()),
    }

    proof_path = Path(gate.BASE_PROOF)
    if gate.sha(proof_path) != gate.BASE_PROOF_SHA256:
        raise ValueError("Original whole-byte scientific asset proof changed")
    asset_proof = json.loads(proof_path.read_text())
    checked = backbones._checked_file
    stats: dict[str, list[int]] = {}

    def original_file(values, key):
        if key == "weights":
            path = Path(values["weights_path"])
            record = asset_proof["scientific_assets"][str(path)]
            stats[str(path)] = gate.signature(path)
            if (
                stats[str(path)] != record["signature"]
                or stats[str(path)] != original["official_base_fresh_stats"].get(str(path))
                or values["weights_sha256"] != record["sha256"]
            ):
                raise ValueError("Whole-byte-verified official base stat binding changed")
            return path
        return checked(values, key)

    normalized: list[str] = []

    def build(name: str) -> tuple:
        values = configs[name]
        strategy = values["classifier_strategy"]

        def factory(encoder, dim, **kwargs):
            return classifier_model(strategy)(
                encoder,
                dim,
                strategy,
                encode_chunk_size=values["encode_chunk_size"],
                input_channels=3,
            )

        model, provenance, state = gate.initialize(
            values, "public", backbones.create_encoder, factory
        )
        normalized.append(gate.check_imagenet_once(model.encoder, windows[0]))
        return model, provenance, state

    backbones._checked_file = original_file
    try:
        evidence, states = matched_initialization(build, probe_inputs, labels, mask, seed, original)
    finally:
        backbones._checked_file = checked
    if set(normalized) != {original["first_window_imagenet_sha256"]["public"]}:
        raise ValueError("Public 336 RGB triplets no longer normalize exactly as admitted")

    value = {
        "status": STATUS,
        "experiment": EXPERIMENT,
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": manifest["source_git"],
        "source_manifest_sha256": gate.sha(manifest_path),
        "recovery_source_manifest_sha256": gate.sha(recovery_manifest_path),
        "wrapper_sha256": manifest["files"][WRAPPER],
        "admission_sha256": manifest["files"][SELF],
        "recipe_sha256": dict(RECIPE_SHA256),
        "code_sha256": code_sha256,
        "code_delta_from_original": delta,
        "encoder_spec": spec | {"path_sha256": gate.sha(spec_path)},
        "funding_sha256": gate.sha(funding_path),
        "fresh_storage": {
            "observed_free_bytes": observed_free,
            "required_free_bytes": funding["required_free_bytes"],
            "existing_liabilities_bytes": funding["existing_liabilities_bytes"],
            "reserve_bytes": funding["reserve_bytes"],
        },
        "original_CPU_receipt_sha256": dense.ORIGINAL_RECEIPT_SHA256,
        "merged_cache_receipt_sha256": dense.MERGED_RECEIPT_SHA256,
        "whole_byte_base_proof_sha256": gate.BASE_PROOF_SHA256,
        "official_base_fresh_stats_equal_original": True,
        "official_encoder_provenance_equal_original": True,
        "release_sha256": binding["manifest_sha256"],
        "folds_sha256": supervision["folds_sha256"],
        "overlay_sha256": supervision["overlay_sha256"],
        "public_cohort": cohort,
        "supervision_sha256": gate.canonical(supervision),
        "full_fitting_studies": len(fitting),
        "full_outer_studies": len(outer),
        "engineering_studies": {role: len(part) for role, part in parts.items()},
        "training_16_sha256": original["training_16_sha256"],
        "outer_16_sha256": original["outer_16_sha256"],
        "public_tensor_sha256": proofs,
        "first_window_imagenet_sha256": original["first_window_imagenet_sha256"]["public"],
        "gradient_probe_study": probe_study,
        "matched_initialization_and_gradients": evidence,
        "future_obligations": future_obligations(
            configs, {name: states[name]["learned_parameters"] for name in NAMES}
        ),
        "GPU_wrapper_arguments": {
            "source": str(code),
            "source-manifest": str(recovery_manifest_path),
            "encoder-spec": str(spec_path),
            "artifact-root": inputs["artifact-root"],
            "cache": gate.PUBLIC_CACHE,
            "overlay": inputs["overlay"],
            "max-output-bytes": RECOVERY_OUTPUT_CAP_BYTES,
        },
        "optimizer_step_executed": False,
        "GPU_or_clinical_inference_executed": False,
        "merge_render_download_or_transfer_executed": False,
        "protected_labels_used_for_recipe_or_metrics": False,
        "new_labels_created": False,
        "accuracy_claimed": False,
        "GPU_acceptance_inferred": False,
        "diagnostic_only": True,
    }
    private = [str(uid) for frame_part in (fitting, outer) for uid in frame_part.study_uid]
    private += [str(group) for frame_part in (fitting, outer) for group in frame_part.leakage_group]
    dense.write_once(root / "CPU-accepted.json", value, private)
    print(json.dumps({"status": STATUS, "CPU_job": value["CPU_job"]}), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
