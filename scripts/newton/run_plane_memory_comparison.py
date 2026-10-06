"""Fixed-budget cross-sequence readout comparison on original owning-node MRI inputs."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/tmp/rsna-plane-memory-full-fit-20261006-v1")
INPUT = Path("/tmp/rsna-plane-memory-inputs-20261006-v1")
PREVIOUS = Path("/tmp/rsna-smooth-witness-full-fit-20261006-v2")
BASE = Path("/home/sa305415/rsna-knee")
SCIENCE = "b2bb3a9c19be45b63074ee792180fa16980830b0"
ROLES = ("control", "candidate")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def signature(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def verify_recipe(values, role, accepted):
    from rsna.competitive.train import TrainConfig

    if role not in ROLES or values != accepted:
        raise ValueError("Exact CPU-admitted memory arm required")
    TrainConfig(**values).validate()
    if (
        values.get("witness") is not None
        or values.get("relation") is not None
        or values.get("fitting_supervision") is not None
        or values.get("neighbor_channels") is not None
    ):
        raise ValueError("Only the isolated sequence-memory branch is admitted")
    if values.get("plane_memory") != {
        "seed": 20261006,
        "hidden_dim": 128,
        "weight": int(role == "candidate"),
    }:
        raise ValueError("Exact zero/one width128 readout policy required")
    fixed = {
        "epochs": 3,
        "diagnostic_study_limit": None,
        "fit_scope": "outer_fold",
        "fold": 0,
        "seed": 20260922,
        "side": 384,
        "slices": 16,
        "max_series": 6,
        "batch_size": 1,
        "accumulation_steps": 8,
        "warmup_updates": 141,
        "precision": "fp16",
        "amp_init_scale": 128,
        "encode_chunk_size": 8,
        "pooling": "attention",
        "slice_context": "none",
        "series_sampling": "selected",
        "slice_sampling": "uniform",
    }
    if any(values.get(k) != v for k, v in fixed.items()):
        raise ValueError("Predeclared full fixed3 native384 comparison required")
    return values


def expected_binding_config(values, provenance):
    from rsna.competitive.train import TrainConfig

    result = asdict(TrainConfig(**values))
    for name in (
        "relation",
        "witness",
        "fitting_supervision",
        "neighbor_channels",
        "patch_summary",
    ):
        if result[name] is None:
            result.pop(name)
    result["encoder"] = provenance
    return result


def original():
    if os.uname().nodename != "evc31" or ROOT.is_symlink() or ROOT.stat().st_uid != os.getuid():
        raise ValueError("Private owning-node original control required")
    files = json.loads((ROOT / "files.json").read_text())
    for name, row in files["files"].items():
        path = ROOT / name
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Immutable control source/config changed")
    source = json.loads((ROOT / "source.json").read_text())
    for name, h in source["files"].items():
        if sha(ROOT / "code" / name) != h:
            raise ValueError("Frozen scientific closure changed")
    if (
        source["source_git"] != SCIENCE
        or sha(INPUT / "CPU-accepted.json") != files["original_memory_CPU_sha256"]
    ):
        raise ValueError("Actual original memory source/CPU proof required")
    memory = json.loads((INPUT / "CPU-accepted.json").read_text())
    if (
        memory["status"] != "actual_original32_MRI_and_official_memory_initialization_CPU_accepted"
        or memory["source_git"] != SCIENCE
        or memory["initial_learned_state"]["control"]
        != memory["initial_learned_state"]["candidate"]
    ):
        raise ValueError("Actual common official initial graph/RNG admission required")
    previous = json.loads((PREVIOUS / "CPU-accepted.json").read_text())
    if (
        sha(PREVIOUS / "CPU-accepted.json")
        != "9ad690361fe22a7f88fd82c2b6e4c183ebe17f289f8a5f44718e1bb1de7e3d74"
    ):
        raise ValueError("Original full input/runtime receipt changed")
    if memory["trainer_inputs"] != previous["trainer_inputs"] or any(
        memory[k] != previous[k] for k in ("training_uids_sha256", "outer_uids_sha256")
    ):
        raise ValueError("Original complete fitting/development binding changed")
    header = Path("/tmp/rsna-neighbor-recovery-20261005-v3/header-CPU-accepted.json")
    if sha(header) != source["original_header_receipt_sha256"]:
        raise ValueError("Original independently accepted runtime header changed")
    config = json.loads(header.read_text())
    if config["status"] != "actual_original_runtime_Triton_helper_compile_import_accepted":
        raise ValueError("Actual runtime header acceptance required")
    return files, source, memory, previous, config


def stage():
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("Input/stat staging is CPU-only")
    files, source, memory, previous, header = original()
    proof_path = BASE / "production-launch-native384-a32-c9-v1/cpu-preflight.json"
    if sha(proof_path) != "fa9e8a52ce05e81c08c2c3ae3a5eec28999109be74af2d4c19b8ddef0e76e9b5":
        raise ValueError("Original whole-byte cache/asset proof differs")
    proof = json.loads(proof_path.read_text())
    for path, row in proof["scientific_assets"].items():
        if signature(path) != row["signature"]:
            raise ValueError("Original scientific asset stat differs")
    registry = BASE / "native384-assembly-c9-v1/registry.json"
    if sha(registry) != "73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770":
        raise ValueError("Original full registry differs")
    members = {
        n: r for sh in json.loads(registry.read_text())["shards"] for n, r in sh["members"].items()
    }
    restored = Path(memory["trainer_inputs"]["cache"]).parent
    if len(members) != 37812:
        raise ValueError("Complete original cache required")
    for name, row in members.items():
        path = restored / name
        if (
            path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or signature(path) != proof["signatures"][name]
        ):
            raise ValueError("Original complete cache stat differs")
    configs = {
        r: verify_recipe(
            json.loads((ROOT / (r + ".json")).read_text()),
            r,
            json.loads((INPUT / (r + ".json")).read_text()),
        )
        for r in ROLES
    }
    ep = previous["expected_bound_config"]["control"]["encoder"]
    if any(digest(ep) != memory["encoder_provenance_sha256"][r] for r in ROLES):
        raise ValueError("Actual identical official encoder provenance required")
    if shutil.disk_usage("/tmp").free < (20 + 16 + 10) * 2**30:
        raise ValueError("Existing live future16GiB plus new10GiB and reserve20GiB required")
    receipt = {
        "status": "actual_full_memory_original_inputs_and_initialization_reused",
        "job_id": os.environ["SLURM_JOB_ID"],
        "source_git": SCIENCE,
        "source_sha256": source["files"],
        "training_studies": 2247,
        "outer_studies": 1156,
        "initial_learned_state": memory["initial_learned_state"],
        "original_memory_CPU_receipt_sha256": sha(INPUT / "CPU-accepted.json"),
        "original_memory_CPU_job": memory["CPU_job"],
        "original_cache_member_stat_bindings": 37812,
        "training_uids_sha256": memory["training_uids_sha256"],
        "outer_uids_sha256": memory["outer_uids_sha256"],
        "folds_sha256": previous["folds_sha256"],
        "release_binding_sha256": previous["release_binding_sha256"],
        "expected_bound_config": {r: expected_binding_config(configs[r], ep) for r in ROLES},
        "config_sha256": {r: sha(ROOT / (r + ".json")) for r in ROLES},
        "runtime": source["runtime"],
        "trainer_inputs": memory["trainer_inputs"],
        "original_header_receipt_sha256": sha(
            Path("/tmp/rsna-neighbor-recovery-20261005-v3/header-CPU-accepted.json")
        ),
        "wrapper_sha256": sha(ROOT / "launch.py"),
        "publisher_sha256": sha(ROOT / "publish.py"),
        "shared_future_bytes": 0,
        "shared_future_files": 0,
        "node_future_bytes": 10 * 2**30,
        "checkpoint_future_bytes": 6 * 2**30,
        "download_scratch_future_bytes": 2 * 2**30,
        "GPU_or_optimizer_executed": False,
        "clinical_accuracy_evidence": False,
        "T4_recovery_required_before_GPU_dispatch": True,
    }
    save(ROOT / "CPU-accepted.json", receipt)
    print(json.dumps(receipt), flush=True)


def recover(inp, configs, cpu):
    import torch

    from scripts.newton.probe_patch_witness_trainer_recovery import (
        compare_recovery,
        numerical_updates,
    )

    results = {}
    for role in ROLES:
        values = dict(
            configs[role],
            epochs=2,
            diagnostic_study_limit=16,
            warmup_updates=2,
            submission_every_epochs=0,
        )
        config = ROOT / (role + "-engineering.json")
        save(config, values)
        full, replay = ROOT / (role + "-engineering-full"), ROOT / (role + "-engineering-replay")
        command = [
            sys.executable,
            "-B",
            str(ROOT / "code/scripts/newton/enforce_deterministic_runtime.py"),
            "--module",
            "rsna.competitive.train",
            "--config",
            str(config),
            "--device",
            "cuda",
        ]
        for k, v in inp.items():
            command += ["--" + k, v]

        def execute(extra, phase, expected):
            with (
                (ROOT / (role + "-engineering-" + phase + ".stdout")).open("x") as out,
                (ROOT / (role + "-engineering-" + phase + ".stderr")).open("x") as err,
            ):
                result = subprocess.run(
                    command + extra, cwd=ROOT / "code", stdout=out, stderr=err, timeout=1500
                )
            if result.returncode != expected:
                raise ValueError("Actual H100 recovery failed; retain originals")

        execute(["--output", str(full)], "full", 0)
        execute(["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        execute(["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0)
        whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        history = json.loads((full / "invocation.json").read_text())
        last = json.loads((replay / "invocation.json").read_text())
        state = compare_recovery(
            whole,
            resumed,
            partial,
            history,
            first,
            last,
            (full / "outer_predictions.csv").read_bytes(),
            (replay / "outer_predictions.csv").read_bytes(),
        )
        if (
            whole["binding"]["config"]["plane_memory"] != values["plane_memory"]
            or whole["amp_scaler"]["_growth_tracker"] != 4
            or whole["binding"]["code_sha256"]
            != {
                n.removeprefix("rsna/"): h
                for n, h in cpu["source_sha256"].items()
                if n.startswith("rsna/")
            }
        ):
            raise ValueError("Actual numerical policy/scaler/source differs")
        original = json.loads(
            Path("/tmp/rsna-patch-witness-H100-recovery-20261006-v2/cpu-accepted.json").read_text()
        )
        if (
            whole["binding"]["training_uids_sha256"] != original["training_16_sha256"]
            or whole["binding"]["outer_uids_sha256"] != original["outer_16_sha256"]
        ):
            raise ValueError("Original engineering memberships differ")
        results[role] = {
            "complete_state_sha256": state,
            "checkpoint_sha256": sha(full / "checkpoint.pt"),
            "prediction_sha256": sha(full / "outer_predictions.csv"),
            "finite_updates": numerical_updates(history),
            "amp_scaler": whole["amp_scaler"],
            "exact_complete_state_and_CSV_recovery": True,
        }
        del whole, resumed, partial
        gc.collect()
        torch.cuda.empty_cache()
    save(
        ROOT / "actual-H100-recovery.json",
        {
            "status": "actual_plane_memory_H100_trainer_recovery_accepted",
            "job_id": os.environ["SLURM_JOB_ID"],
            "source_git": SCIENCE,
            "CPU_receipt_sha256": sha(ROOT / "CPU-accepted.json"),
            "roles": results,
            "clinical_accuracy_evidence": False,
        },
    )


def training_only_review(inputs, configs):
    import pandas as pd
    import torch

    from rsna.competitive.adapters import configure_encoder_training
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
    from rsna.competitive.folds import load_folds
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.competitive.train import TrainConfig, _uid_hash, select_development
    from rsna.release import load_verified_release

    frame, binding = load_verified_release(Path(inputs["studies"]), Path(inputs["artifact-root"]))
    folds = load_folds(Path(inputs["folds"]), frame, binding)
    fitting, outer = select_development(frame, folds, 0)
    chosen = fitting.head(3)
    if set(chosen.study_uid) & set(outer.study_uid):
        raise ValueError("Readout stress must use fitting images only")
    config = TrainConfig(**configs["candidate"])
    torch.manual_seed(config.seed)
    encoder, dim, ep = create_encoder(config.encoder)
    configure_encoder_training(encoder, config.freeze_encoder)
    model = CompetitiveStudyNet(
        encoder,
        dim,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        plane_memory=config.plane_memory,
    )
    guard = FrozenBaseDelta(model, ep)
    checkpoint = torch.load(
        ROOT / "candidate-engineering-full/checkpoint.pt", map_location="cpu", weights_only=True
    )
    restore_checkpoint_model(model, checkpoint, guard)
    del checkpoint, guard
    model = model.to("cuda").eval()
    captured = {}

    def hook(module, arguments, output):
        captured["readout"] = output.detach().float().cpu()

    handle = model.plane_memory.register_forward_hook(hook)
    dataset = CompetitiveStudyDataset(
        chosen,
        pd.read_csv(inputs["series"]),
        Path(inputs["cache"]),
        slices=config.slices,
        side=config.side,
        max_series=config.max_series,
    )
    rows = []
    try:
        for index in range(len(dataset)):
            batch = collate_studies([dataset[index]])
            values = {
                k: batch[k].to("cuda")
                for k in ("images", "features", "series_mask", "slice_mask", "positions")
            }

            def forward(values, current_model=model):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    result = current_model(**values)
                if not torch.isfinite(result).all():
                    raise ValueError("Nonfinite readout stress")
                return result.float().cpu()

            normal = forward(values)
            readout = captured["readout"]
            padding = {k: v.clone() for k, v in values.items()}
            padding["images"][~padding["slice_mask"]] = float("nan")
            padding["positions"][~padding["slice_mask"]] = float("nan")
            padding["features"][~padding["series_mask"]] = float("nan")
            if not torch.equal(normal, forward(padding)):
                raise ValueError("Invalid padding changed actual MRI prediction")
            background = forward(dict(values, images=torch.zeros_like(values["images"])))
            missing = None
            if int(values["series_mask"].sum()) > 1:
                reduced = {k: v.clone() for k, v in values.items()}
                first = int(torch.nonzero(reduced["series_mask"][0], as_tuple=False)[0, 0])
                reduced["series_mask"][0, first] = False
                reduced["slice_mask"][0, first] = False
                missing = forward(reduced)
            rows.append(
                {
                    "fitting_engineering_index": index,
                    "max_abs_readout_logit": float(readout.abs().max()),
                    "max_abs_original_global_logit": float((normal - readout).abs().max()),
                    "NaN_padding_prediction_exact": True,
                    "black_background_max_probability_change": float(
                        (normal.sigmoid() - background.sigmoid()).abs().max()
                    ),
                    "missing_series_max_probability_change": float(
                        (normal.sigmoid() - missing.sigmoid()).abs().max()
                    )
                    if missing is not None
                    else None,
                }
            )
    finally:
        handle.remove()
    save(
        ROOT / "training-only-review.json",
        {
            "status": "actual_fitting_only_memory_contribution_and_finite_input_stresses_checked",
            "job_id": os.environ["SLURM_JOB_ID"],
            "training_roster_sha256": _uid_hash(chosen),
            "checkpoint_sha256": sha(ROOT / "candidate-engineering-full/checkpoint.pt"),
            "rows": rows,
            "diagnostic_AUC_used_to_choose_recipe": False,
            "anatomical_or_shortcut_validity_established": False,
            "clinical_accuracy_evidence": False,
        },
    )
    del model, encoder, dataset
    gc.collect()
    torch.cuda.empty_cache()


def fit():
    import torch

    files, source, memory, previous, header = original()
    cpu = json.loads((ROOT / "CPU-accepted.json").read_text())
    if (
        cpu["status"] != "actual_full_memory_original_inputs_and_initialization_reused"
        or cpu["source_sha256"] != source["files"]
        or cpu["initial_learned_state"] != memory["initial_learned_state"]
        or any(cpu["config_sha256"][r] != sha(ROOT / (r + ".json")) for r in ROLES)
    ):
        raise ValueError(
            "Actual complete CPU source/configuration/initialization admission required"
        )
    t4 = json.loads((ROOT / "T4-recovery-accepted.json").read_text())
    if (
        t4["status"] != "actual_single_T4_sequential_plane_memory_recovery_accepted"
        or t4["science_git"] != SCIENCE
        or t4["CPU_receipt_sha256"] != sha(INPUT / "CPU-accepted.json")
    ):
        raise ValueError("Actual new-source T4 recovery required before dispatch")
    if any(
        not t4["roles"][r]["exact_complete_state_scaler_RNG_and_CSV_recovery"]
        or len(t4["roles"][r]["finite_updates"]) != 4
        or t4["roles"][r]["amp_scaler"]["_growth_tracker"] != 4
        for r in ROLES
    ):
        raise ValueError("Complete actual finite T4 recovery required")
    if (
        os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
        or os.environ.get("CPATH") != header["CPATH"]
        or sys.executable != source["runtime"]["python"]
    ):
        raise ValueError("Actual assigned one-H100/two-CPU pinned runtime/header required")
    value = torch.ones(4, device="cuda")
    assert value.sum().item() == 4
    del value
    save(
        ROOT / "GPU-header.json",
        {
            "status": "actual_plane_memory_assigned_H100_context_accepted",
            "job_id": os.environ["SLURM_JOB_ID"],
            "CPU_receipt_sha256": sha(ROOT / "CPU-accepted.json"),
            "T4_receipt_sha256": sha(ROOT / "T4-recovery-accepted.json"),
            "source_git": SCIENCE,
            "device": torch.cuda.get_device_name(0),
        },
    )
    configs = {
        r: verify_recipe(
            json.loads((ROOT / (r + ".json")).read_text()),
            r,
            json.loads((INPUT / (r + ".json")).read_text()),
        )
        for r in ROLES
    }
    recover(cpu["trainer_inputs"], configs, cpu)
    training_only_review(cpu["trainer_inputs"], configs)
    for role in ROLES:
        out = ROOT / role
        if out.exists():
            raise ValueError("Original fitting output must never be repeated")
        save(
            ROOT / (role + "-trainer-intent.json"),
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "role": role,
                "source_git": SCIENCE,
                "config_sha256": sha(ROOT / (role + ".json")),
                "automatic_retry": False,
                "fixed_final": 3,
            },
        )
        command = [
            sys.executable,
            "-B",
            str(ROOT / "code/scripts/newton/enforce_deterministic_runtime.py"),
            "--module",
            "rsna.competitive.train",
            "--config",
            str(ROOT / (role + ".json")),
            "--device",
            "cuda",
            "--output",
            str(out),
        ]
        for k, v in cpu["trainer_inputs"].items():
            command += ["--" + k, v]
        with (
            (ROOT / (role + ".stdout")).open("x") as stdout,
            (ROOT / (role + ".stderr")).open("x") as stderr,
        ):
            result = subprocess.run(command, cwd=ROOT / "code", stdout=stdout, stderr=stderr)
        save(
            ROOT / (role + "-terminal.json"),
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "returncode": result.returncode,
                "role": role,
                "automatic_retry": False,
            },
        )
        if result.returncode:
            raise ValueError("Full fitting failed; preserve actual originals")
        terminal = json.loads((out / "result.json").read_text())
        run = json.loads((out / "run.json").read_text())
        if (
            terminal["optimizer_steps"],
            terminal["training_studies"],
            terminal["outer_studies"],
        ) != (843, 2247, 1156) or run["binding"]["config"] != cpu["expected_bound_config"][role]:
            raise ValueError("Actual fixed-final source/config/cohort differs")
        if (
            sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and not p.is_symlink())
            > 10 * 2**30
        ):
            raise ValueError("Ten GiB retained-output envelope exceeded")
    save(
        ROOT / "terminal.json",
        {
            "status": "two_original_full_plane_memory_arms_completed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "fixed_epochs": 3,
            "optimizer_updates_per_arm": 843,
            "endpoint_selection": False,
            "clinical_paired_analysis_completed": False,
        },
    )


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["stage", "fit"])
    args = parser.parse_args()
    stage() if args.phase == "stage" else fit()
