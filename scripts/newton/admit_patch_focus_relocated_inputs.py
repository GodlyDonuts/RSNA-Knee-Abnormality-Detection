"""Independent CPU replay on evc26 of the byte-exact relocated patch-focus source.

Prepared only. Source-node acceptance stays unchanged. No optimizer or GPU.
Numerical replay follows the reviewed evc40 admission; all imports use its frozen code.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

import torch

from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import admit_patch_focus_inputs as admit
from scripts.newton import admit_patch_focus_target_inputs as target
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import triplet_cache_relay as triplet

ROOT = Path("/tmp/rsna-patch-focus-evc26-admission-20261014-v1")
CACHE = Path(gate.PUBLIC_CACHE)
CONTROL = Path("/tmp/rsna-patch-focus-cache-relay-20261014-v1")
RUNTIME = "/tmp/rsna-environment-917129-_r45a3dc/environment"
SOURCE_CPU_SHA = "70d9c6722c6d6e4c9eb0be6653df0fccc0d159e7db2d33e6abb4de1e545509d5"
SOURCE_MANIFEST_SHA = "c46c6dec4ffc2b24648578cf35055bf2a4f6c0ba69edeffcf0c9fa9094b552ae"
SOURCE_GIT = "bf46b70eea1216673aafb1b5d34e513247b4ef1c"
OVERLAY = ROOT / "overlay.json"
OVERLAY_SHA256 = gate.OVERLAY_SHA256
OUTPUT = "relocated-CPU-accepted.json"
STATUS = "actual_relocated_patch_focus_source_inputs_models_RNG_gradients_CPU_accepted"


def check_placement(node: str, env: Mapping[str, str]) -> str:
    job = env.get("SLURM_JOB_ID", "")
    if (
        node != "evc26"
        or not job.isdigit()
        or job in {"908454", "917001", "916922", "917243", "917379"}
        or env.get("SLURM_CPUS_PER_TASK") != "1"
        or any(env.get(k) for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS"))
        or env.get("CUDA_VISIBLE_DEVICES") not in (None, "", "NoDevFiles")
    ):
        raise ValueError("Fresh standalone evc26 one-CPU zero-GPU allocation required")
    return job


def check_runtime(actual: Mapping, source: Mapping) -> None:
    if actual.get("prefix") != RUNTIME or {k: v for k, v in actual.items() if k != "prefix"} != {
        k: v for k, v in source.items() if k != "prefix"
    }:
        raise ValueError("Explicit relocated prefix/runtime differs from accepted source")


def check_source(code: Path, manifest: Mapping, cpu: Mapping) -> dict:
    if (
        manifest.get("source_git") != SOURCE_GIT
        or cpu.get("source_git") != SOURCE_GIT
        or cpu.get("node") != "evc40"
        or cpu.get("CPU_job") != "916922"
        or cpu.get("status") != target.STATUS
    ):
        raise ValueError("Original source-node CPU lineage differs")
    values = admit.verify_source(code, manifest)
    if values != cpu["code_sha256"]:
        raise ValueError("Relocated numerical source differs from accepted source")
    return values


def run(root: Path) -> dict:
    from rsna.competitive import backbones
    from rsna.competitive.data import collate_studies
    from rsna.competitive.folds import load_folds
    from rsna.competitive.patch_focus import classifier_model
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from rsna.competitive.source_slot import training_inputs
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development
    from rsna.schema import TARGETS

    job = check_placement(os.uname().nodename, os.environ)
    if root != ROOT or torch.cuda.is_available():
        raise ValueError("Exact private root/CPU placement required")
    torch.set_num_threads(1)
    target._no_links(root)
    output = root / OUTPUT
    if output.exists():
        raise ValueError("Existing destination: reconcile, never repeat")
    code = root / "code"
    manifest_path = root / "source.json"
    if (
        gate.sha(manifest_path) != SOURCE_MANIFEST_SHA
        or gate.sha(root / "source-CPU-accepted.json") != SOURCE_CPU_SHA
    ):
        raise ValueError("Source proof bytes changed")
    manifest = json.loads(manifest_path.read_text())
    cpu = json.loads((root / "source-CPU-accepted.json").read_text())
    code_sha256 = check_source(code, manifest, cpu)
    runtime = target.runtime_identity()
    check_runtime(runtime, cpu["runtime"])
    relay_path = CONTROL / "target-verified.json"
    relay = json.loads(relay_path.read_text())
    if (
        relay.get("status") != "distinct_source40_to_target26_complete_cache_full_size_SHA_verified"
        or relay.get("source_node") != "evc40"
        or relay.get("target_node") != "evc26"
        or relay.get("source_CPU_receipt_sha256") != SOURCE_CPU_SHA
        or relay.get("plan_sha256")
        != "067f30d66ed1171c09d910eb2c98fa30668192489e9b64a93069a386a96cbd74"
        or relay.get("index_sha256") != triplet.INDEX_SHA256
        or relay.get("all_bytes_size_SHA_count_EOF_verified") is not True
        or (relay.get("files"), relay.get("bytes")) != (3404, 27668082285)
    ):
        raise ValueError("Actual relocation proof differs")
    cache = CACHE
    m = triplet.manifest_from_index((cache / triplet.INDEX).read_bytes())
    cache_proof = target.check_cache(cache, m, relay_path.stat())
    if (
        cache_proof["signatures_sha256"] != relay["signatures_sha256"]
        or triplet.canonical_sha256(m) != relay["manifest_sha256"]
    ):
        raise ValueError("Copied cache changed after verification")
    target._regular(OVERLAY)
    if stat.S_IMODE(OVERLAY.stat().st_mode) != 0o400 or gate.sha(OVERLAY) != OVERLAY_SHA256:
        raise ValueError("Original overlay differs")
    used = target.root_usage(root) + target.root_usage(CONTROL)
    if used + 2**20 > 8 * 2**20:
        raise ValueError("Existing target control cap exceeded")
    # Remaining pre-funded cache/runtime/control obligations and 20GiB reserve remain.
    required = (
        356641161911
        + (8 * 2**30 - 6322691335)
        + (32 * 2**30 - 27668082285)
        + 32 * 2**20
        + (8 * 2**20 - used)
        + 20 * 2**30
    )
    if shutil.disk_usage(root).free < required:
        raise ValueError("Fresh outstanding obligations/reserve breached")
    recovery_manifest_path = root / "recovery-source-manifest.json"
    if json.loads(recovery_manifest_path.read_text()) != manifest["files"]:
        raise ValueError("GPU wrapper source manifest differs from the admitted source")
    if admit.recovery.verify_source(code, manifest["files"]) != code_sha256:
        raise ValueError("GPU wrapper would bind different numerical source")
    original = json.loads((code / dense.ORIGINAL_RECEIPT).read_text())
    inputs = dense.validate_original(original)
    if Path(inputs["overlay"]) != dense.ORIGINAL_OVERLAY:
        raise ValueError("Original overlay lineage differs")
    delta = admit.code_delta(original["code_sha256"], code_sha256)
    configs = admit.load_arms(code, original["configs"]["public"])
    spec_path = root / "encoder-spec.json"
    spec = admit.encoder_spec(configs, json.loads(spec_path.read_text()))

    config = configs["reference"]
    frame, binding = target.bounded_release(inputs, original)
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
        OVERLAY,
        config["fitting_supervision"],
        release_sha256=binding["manifest_sha256"],
        folds_sha256=gate.sha(folds_path / "fold.csv"),
        fold=config["fold"],
    )
    if supervision != original["supervision"]:
        raise ValueError("Fitting labels/masks/fold supervision differ from the original receipt")
    policy = config["public_input"]
    cohort = verify_public_cohort(cache, policy, fitting, outer, binding["source_revision"])
    if cohort != original["public_cohort"] or cohort["index_sha256"] != relay["index_sha256"]:
        raise ValueError("Complete public cohort membership differs from the original receipt")
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

    # Real fitting-only MRI study; development rows are never read for gradients.
    position, routing_mask, routing_fixture = admit.routing_probe(fitting)
    item = dataset(fitting.iloc[position : position + 1], True)[0]
    batch = collate_studies([item])
    batch["study_uid"] = list(fitting.study_uid.iloc[position : position + 1])
    seed = config["seed"]
    probe_inputs = training_inputs(
        batch, torch.device("cpu"), config["classifier_strategy"], seed=seed, epoch=0, update=0
    )
    labels, original_mask = batch["labels"].float(), batch["label_mask"].float()
    fitting_mask = torch.tensor(
        fitting.iloc[position : position + 1][[f"{t}__mask" for t in TARGETS]].to_numpy(),
        dtype=torch.float32,
    )
    if not torch.equal(original_mask, fitting_mask):
        raise ValueError("Dataset mask differs from the admitted fitting row")
    probe_study = {
        "fitting_position": position,
        "images_sha256": hashlib.sha256(item["images"].numpy().tobytes()).hexdigest(),
        "labels_and_mask_sha256": hashlib.sha256(
            labels.numpy().tobytes() + original_mask.numpy().tobytes()
        ).hexdigest(),
        "routing_fixture": routing_fixture,
        "routing_mask_sha256": hashlib.sha256(routing_mask.numpy().tobytes()).hexdigest(),
        "acquired_slots": int(batch["series_mask"].sum()),
    }

    proof_path = Path(gate.BASE_PROOF)
    if gate.sha(proof_path) != gate.BASE_PROOF_SHA256:
        raise ValueError("Original whole-byte scientific asset proof changed")
    asset_proof = json.loads(proof_path.read_text())
    checked = backbones._checked_file
    methods: dict[str, str] = {}

    def original_file(values, key):
        if key == "weights":
            path = Path(values["weights_path"])
            record = asset_proof["scientific_assets"][str(path)]
            if values["weights_sha256"] != record["sha256"]:
                raise ValueError("Official base SHA differs from the whole-byte proof")
            stats = original["official_base_fresh_stats"].get(str(path))
            methods[str(path)] = methods.get(str(path)) or target.base_binding(path, record, stats)
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
        evidence, states = admit.matched_initialization(
            build, probe_inputs, labels, routing_mask, seed, original
        )
    finally:
        backbones._checked_file = checked
    if set(normalized) != {original["first_window_imagenet_sha256"]["public"]}:
        raise ValueError("Public 336 RGB triplets no longer normalize exactly as admitted")
    if gate.sha(OVERLAY) != OVERLAY_SHA256 or output.exists():
        raise ValueError("Overlay changed or destination appeared during admission")

    expected = cpu["matched_initialization_and_gradients"]
    for key in (
        "initial_values_sha256",
        "initial_logits_sha256",
        "post_factory_RNG_sha256",
        "reference_encoder_gradient_norm",
        "candidate_query_gradient_norm",
    ):
        if evidence[key] != expected[key]:
            raise ValueError("Relocated actual model/RNG/gradient differs from original source")
    if proofs != cpu["public_tensor_sha256"]:
        raise ValueError("Relocated actual tensors differ from original source")
    result = {
        "status": STATUS,
        "CPU_job": job,
        "node": "evc26",
        "source_node": "evc40",
        "source_CPU_receipt_sha256": SOURCE_CPU_SHA,
        "source_git": SOURCE_GIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "relocation_receipt_sha256": gate.sha(relay_path),
        "runtime": runtime,
        "overlay_sha256": OVERLAY_SHA256,
        "public_cohort": cohort,
        "supervision_sha256": gate.canonical(supervision),
        "public_tensor_sha256": proofs,
        "matched_initialization_and_gradients": evidence,
        "code_delta": delta,
        "encoder_spec": spec,
        "backward_only_probe": probe_study,
        "source_files_and_masks_changed": False,
        "GPU_or_optimizer_or_clinical_metric": False,
        "fresh_required_free_bytes": required,
        "original_source_node_proofs_relabelled": False,
    }
    private = (
        list(fitting.study_uid)
        + list(outer.study_uid)
        + list(fitting.leakage_group)
        + list(outer.leakage_group)
    )
    dense.write_once(output, result, private)
    print(json.dumps({"status": STATUS, "CPU_job": job}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
