"""One owning-node, fixed-final smooth-witness two-arm fit; no retry or endpoint selection."""

import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/tmp/rsna-smooth-witness-full-fit-20261006-v1")
ORIGINAL = Path("/tmp/rsna-patch-witness-H100-recovery-20261006-v2")
BASE = Path("/home/sa305415/rsna-knee")
CPU_SHA = "1e9505f9c7ffded479ea7f4631d38f1ae49f3b431f8b866600f8924a353526ee"
REGISTRY_SHA = "73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770"
ORIGINAL_PROOF_SHA = "fa9e8a52ce05e81c08c2c3ae3a5eec28999109be74af2d4c19b8ddef0e76e9b5"
LIBRARY_GIT = "3b7e437431f23b1f6e89e76aacaecd42df1c3535"
MODES = ("control", "candidate")


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save(p, value):
    with Path(p).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def signature(p):
    s = Path(p).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify_config(config, mode, expected):
    from rsna.competitive.train import TrainConfig

    if mode not in MODES:
        raise ValueError("Unknown fixed arm")
    TrainConfig(**config).validate()
    required = dict(
        expected,
        epochs=3,
        diagnostic_study_limit=None,
        witness={
            "patches": 4,
            "slices": 2,
            "seed": 20261003,
            "weight": 0 if mode == "control" else 1,
            "pooling": "normalized_logmeanexp",
            "beta": 8,
        },
    )
    if (
        required["precision"] != "fp16"
        or required["encode_chunk_size"] != 8
        or required["amp_init_scale"] != 128
        or required.get("neighbor_channels") is not None
        or required.get("fitting_supervision") is not None
    ):
        raise ValueError("Only the actual accepted smooth FP16 recipe is admitted")
    if config != required:
        raise ValueError("Full recipe differs from fixed two-arm plan")
    return config


def original():
    if os.uname().nodename != "evc31" or ROOT.is_symlink() or ROOT.stat().st_uid != os.getuid():
        raise ValueError("Owner-only original node required")
    if shutil.disk_usage("/tmp").free < 36 * 2**30:
        raise ValueError("Sixteen GiB fit future plus twenty GiB reserve required")
    if sha(ORIGINAL / "cpu-accepted.json") != CPU_SHA:
        raise ValueError("Original CPU input admission changed")
    manifest = json.loads((ROOT / "files.json").read_text())
    if sha(ROOT / "T4-recovery-accepted.json") != manifest["T4_recovery_receipt_sha256"]:
        raise ValueError("Actual independent T4 acceptance changed")
    recovery = json.loads((ROOT / "T4-recovery-accepted.json").read_text())
    if (
        recovery["status"] != "COMPLETE"
        or recovery["kernel_id"] != 137276309
        or recovery["scientific_source_git"] != LIBRARY_GIT
    ):
        raise ValueError("Actual exact-source FP16 T4 recovery required")
    for role in MODES:
        accepted = recovery["roles"][role]
        if (
            accepted["optimizer_updates"] != 4
            or not accepted["exact_complete_model_base_AdamW_LR_cursors_all_RNG_and_CSV_recovery"]
            or not accepted["official_initial_graph_RNG_encoder_provenance_equal"]
            or accepted["precision"] != "fp16"
            or accepted["encode_chunk_size"] != 8
            or accepted["amp_scaler"]["_growth_tracker"] != 4
        ):
            raise ValueError("Actual complete finite state/scaler/CSV acceptance required")
    source = json.loads((ROOT / "source.json").read_text())
    if (
        source["source_git"] != LIBRARY_GIT
        or source["files"] != manifest["numerical_source_sha256"]
    ):
        raise ValueError("Actual T4 numerical source differs")
    for n, h in source["files"].items():
        if sha(ROOT / "code" / n) != h:
            raise ValueError("Original scientific module changed")
    manifest = json.loads((ROOT / "files.json").read_text())
    for n, row in manifest["files"].items():
        p = ROOT / n
        if p.is_symlink() or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("Immutable full-fit wrapper/config changed")
    proof_path = BASE / "production-launch-native384-a32-c9-v1/cpu-preflight.json"
    if sha(proof_path) != ORIGINAL_PROOF_SHA:
        raise ValueError("Original whole-byte proof changed")
    proof = json.loads(proof_path.read_text())
    for n, a in proof["scientific_assets"].items():
        if signature(n) != a["signature"]:
            raise ValueError("Original scientific asset stats changed")
    inp = json.loads((ORIGINAL / "cpu-accepted.json").read_text())["trainer_inputs"]
    return manifest, source, proof, inp


def configs():
    base = json.loads((ROOT / "base-config.json").read_text())
    return {
        m: verify_config(json.loads((ROOT / (m + ".json")).read_text()), m, base) for m in MODES
    }


def stage():
    import numpy as np
    import torch

    from rsna.competitive.adapters import configure_encoder_training
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.folds import load_folds
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.competitive.neighbor_input import neighbor_input_channels
    from rsna.competitive.train import TrainConfig, _hash, _uid_hash, select_development
    from rsna.release import load_verified_release

    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Original input/initialization admission is CPU-only")
    torch.set_num_threads(1)
    files, source, proof, inp = original()
    cfg = configs()
    registry = BASE / "native384-assembly-c9-v1/registry.json"
    if sha(registry) != REGISTRY_SHA:
        raise ValueError("Original registry changed")
    reg = json.loads(registry.read_text())
    restored = Path(inp["cache"]).parent
    members = {n: a for sh in reg["shards"] for n, a in sh["members"].items()}
    if len(members) != 37812:
        raise ValueError("Complete original cache required")
    for n, row in members.items():
        p = restored / n
        if (
            p.is_symlink()
            or not p.is_file()
            or p.stat().st_size != row["bytes"]
            or signature(p) != proof["signatures"][n]
        ):
            raise ValueError("Original byte-verified cache member changed")
    release, binding = load_verified_release(Path(inp["studies"]), Path(inp["artifact-root"]))
    folds = load_folds(Path(inp["folds"]), release, binding)
    train, outer = select_development(release, folds, 0)
    if len(train) != 2247 or len(outer) != 1156 or set(train.study_uid) & set(outer.study_uid):
        raise ValueError("Disjoint original full rosters required")
    if set(train.study_uid) != set(reg["training_uids"]) or set(outer.study_uid) != set(
        reg["outer_uids"]
    ):
        raise ValueError("Original registry membership differs")
    initial = {}
    bound = {}
    for mode in MODES:
        c = TrainConfig(**cfg[mode])
        random.seed(c.seed)
        np.random.seed(c.seed % 2**32)
        torch.manual_seed(c.seed)
        channels = neighbor_input_channels(c.neighbor_channels)
        encoder, dim, ep = create_encoder(
            c.encoder, **({"input_channels": 3} if channels == 3 else {})
        )
        configure_encoder_training(encoder, c.freeze_encoder)
        model = CompetitiveStudyNet(
            encoder,
            dim,
            pooling=c.pooling,
            hidden_dim=c.hidden_dim,
            encode_chunk_size=c.encode_chunk_size,
            slice_context=c.slice_context,
            witness=c.witness,
            **({"input_channels": 3} if channels == 3 else {}),
        )
        records = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                a = p.detach().contiguous().cpu().numpy()
                records[n] = {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                }
        initial[mode] = {
            "learned_graph_and_values_sha256": digest(records),
            "post_factory_torch_RNG_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        bc = asdict(c)
        for n in ("relation", "witness", "fitting_supervision", "neighbor_channels"):
            if bc[n] is None:
                bc.pop(n)
        bc["encoder"] = ep
        bound[mode] = bc
        del model, encoder, records
        gc.collect()
    if initial["control"] != initial["candidate"]:
        raise ValueError("Actual official seeded initial learned tensors/RNG differ")
    receipt = {
        "status": "actual_full_smooth_witness_original_inputs_and_seeded_initialization_accepted",
        "job_id": os.environ["SLURM_JOB_ID"],
        "source_git": LIBRARY_GIT,
        "source_sha256": source["files"],
        "training_studies": 2247,
        "outer_studies": 1156,
        "training_uids_sha256": _uid_hash(train),
        "outer_uids_sha256": _uid_hash(outer),
        "folds_sha256": _hash(folds.sort_values("study_uid").to_dict("records")),
        "release_binding_sha256": binding["binding_sha256"],
        "registry_sha256": REGISTRY_SHA,
        "original_whole_byte_proof_sha256": ORIGINAL_PROOF_SHA,
        "original_cache_member_stat_bindings": len(members),
        "original_recovery_receipt_sha256": files["T4_recovery_receipt_sha256"],
        "original_CPU_receipt_sha256": CPU_SHA,
        "initial_learned_state": initial,
        "expected_bound_config": bound,
        "runtime": source["runtime"],
        "trainer_inputs": inp,
        "config_sha256": {m: sha(ROOT / (m + ".json")) for m in MODES},
        "wrapper_sha256": sha(ROOT / "launch.py"),
        "publisher_sha256": sha(ROOT / "publish.py"),
        "shared_future_bytes": 0,
        "shared_future_files": 0,
        "node_future_bytes": 16 * 2**30,
        "checkpoint_future_bytes": 6 * 2**30,
        "download_scratch_future_bytes": 2 * 2**30,
        "node_free_bytes": shutil.disk_usage("/tmp").free,
        "gold_or_confirmation_used": False,
        "clinical_accuracy_evidence": False,
        "GPU_executed": False,
    }
    save(ROOT / "CPU-accepted.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


def fit():
    import torch

    files, source, proof, inp = original()
    cfg = configs()
    cpu = json.loads((ROOT / "CPU-accepted.json").read_text())
    if (
        cpu["status"]
        != "actual_full_smooth_witness_original_inputs_and_seeded_initialization_accepted"
        or cpu["source_sha256"] != source["files"]
    ):
        raise ValueError("Actual full-roster admission required")
    if (
        os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("One actual assigned BF16 H100 and two CPUs required")
    header = json.loads((ORIGINAL / "header-CPU-accepted.json").read_text())
    if os.environ.get("CPATH") != header["CPATH"]:
        raise ValueError("Original accepted CPATH required")
    if sys.executable != source["runtime"]["python"]:
        raise ValueError("Original accepted runtime required")
    x = torch.ones(4, device="cuda")
    assert x.sum().item() == 4
    del x
    save(
        ROOT / "GPU-header.json",
        {
            "status": "actual_full_smooth_witness_allocated_H100_accepted",
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": "evc31",
            "device": torch.cuda.get_device_name(0),
            "CPU_receipt_sha256": sha(ROOT / "CPU-accepted.json"),
            "recovery_receipt_sha256": files["T4_recovery_receipt_sha256"],
            "source_git": LIBRARY_GIT,
            "source_sha256": source["files"],
            "CPATH": os.environ["CPATH"],
            "runtime": source["runtime"],
            "fixed_epochs": 3,
            "updates_per_arm": 843,
            "training_studies": 2247,
            "outer_studies": 1156,
            "shared_future_bytes": 0,
            "node_future_bytes": 16 * 2**30,
            "checkpoint_future_bytes": 6 * 2**30,
        },
    )
    actual_hardware_recovery(inp, cfg, cpu)
    training_only_review(inp, cfg)
    for mode in MODES:
        if sha(ROOT / (mode + ".json")) != cpu["config_sha256"][mode]:
            raise ValueError("Accepted numerical config changed")
        out = ROOT / mode
        if out.exists():
            raise ValueError("Never repeat an original full-fit output")
        save(
            ROOT / (mode + "-trainer-intent.json"),
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "role": mode,
                "source_git": LIBRARY_GIT,
                "config_sha256": cpu["config_sha256"][mode],
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
            str(ROOT / (mode + ".json")),
            "--device",
            "cuda",
            "--output",
            str(out),
        ]
        for k, v in inp.items():
            command += ["--" + k, v]
        with (
            (ROOT / (mode + ".stdout")).open("x") as stdout,
            (ROOT / (mode + ".stderr")).open("x") as stderr,
        ):
            code = subprocess.run(
                command, cwd=ROOT / "code", stdout=stdout, stderr=stderr
            ).returncode
        save(
            ROOT / (mode + "-terminal.json"),
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "role": mode,
                "returncode": code,
                "automatic_retry": False,
            },
        )
        if code:
            raise ValueError("Original full-fit arm failed; preserve and inspect")
        result = json.loads((out / "result.json").read_text())
        run = json.loads((out / "run.json").read_text())
        if (
            result["optimizer_steps"] != 843
            or result["training_studies"] != 2247
            or result["outer_studies"] != 1156
            or run["binding"]["config"] != cpu["expected_bound_config"][mode]
        ):
            raise ValueError("Original fixed-final fitting result differs")
        if (
            sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and not p.is_symlink())
            > 16 * 2**30
        ):
            raise ValueError("Retained full-fit storage cap exceeded")
    save(
        ROOT / "terminal.json",
        {
            "status": "two_original_full_smooth_witness_arms_completed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "fixed_epochs": 3,
            "optimizer_updates_per_arm": 843,
            "endpoint_selection": False,
            "clinical_paired_analysis_completed": False,
        },
    )


def actual_hardware_recovery(inp, cfg, cpu):
    """Accept the exact new FP16 loop on this allocated H100 before the full fit."""
    import pandas as pd
    import torch

    from rsna.schema import TARGETS
    from scripts.newton.probe_patch_witness_trainer_recovery import (
        compare_recovery,
        numerical_updates,
    )

    results = {}
    for mode in MODES:
        c = dict(
            cfg[mode],
            epochs=2,
            diagnostic_study_limit=16,
            warmup_updates=2,
            submission_every_epochs=0,
        )
        path = ROOT / (mode + "-engineering.json")
        save(path, c)
        full = ROOT / (mode + "-engineering-full")
        replay = ROOT / (mode + "-engineering-replay")
        command = [
            sys.executable,
            "-B",
            str(ROOT / "code/scripts/newton/enforce_deterministic_runtime.py"),
            "--module",
            "rsna.competitive.train",
            "--config",
            str(path),
            "--device",
            "cuda",
        ]
        for k, v in inp.items():
            command += ["--" + k, v]

        def execute(arguments, phase, expected):
            with (
                (ROOT / (mode + "-engineering-" + phase + ".stdout")).open("x") as out,
                (ROOT / (mode + "-engineering-" + phase + ".stderr")).open("x") as err,
            ):
                result = subprocess.run(
                    command + arguments, cwd=ROOT / "code", stdout=out, stderr=err, timeout=1200
                )
            if result.returncode != expected:
                raise ValueError("Exact new H100 trainer recovery failed; retain originals")

        execute(["--output", str(full)], "full", 0)
        execute(["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        execute(["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0)
        whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        invocation = json.loads((full / "invocation.json").read_text())
        last = json.loads((replay / "invocation.json").read_text())
        state = compare_recovery(
            whole,
            resumed,
            partial,
            invocation,
            first,
            last,
            (full / "outer_predictions.csv").read_bytes(),
            (replay / "outer_predictions.csv").read_bytes(),
        )
        original = json.loads((ORIGINAL / "cpu-accepted.json").read_text())
        if (
            whole["binding"]["training_uids_sha256"] != original["training_16_sha256"]
            or whole["binding"]["outer_uids_sha256"] != original["outer_16_sha256"]
            or whole["binding"]["config"]["witness"] != c["witness"]
            or whole["binding"]["config"]["precision"] != "fp16"
            or whole["amp_scaler"]["_growth_tracker"] != 4
        ):
            raise ValueError("Original engineering roster/recipe/scaler differs")
        if whole["binding"]["code_sha256"] != {
            n.removeprefix("rsna/"): h
            for n, h in cpu["source_sha256"].items()
            if n.startswith("rsna/")
        }:
            raise ValueError("Accepted numerical source differs")
        predictions = pd.read_csv(full / "outer_predictions.csv")
        scores = predictions[list(TARGETS)].to_numpy()
        if (
            list(predictions.columns) != ["study_uid", "fold", "seed", *TARGETS]
            or len(predictions) != 16
            or digest(sorted(predictions.study_uid.tolist())) != original["outer_16_sha256"]
            or not __import__("numpy").isfinite(scores).all()
            or (scores < 0).any()
            or (scores > 1).any()
        ):
            raise ValueError("Exact finite research CSV required")
        results[mode] = {
            "complete_state_sha256": state,
            "checkpoint_sha256": sha(full / "checkpoint.pt"),
            "prediction_sha256": sha(full / "outer_predictions.csv"),
            "finite_updates": numerical_updates(invocation),
            "amp_scaler": whole["amp_scaler"],
            "exact_complete_state_and_CSV_recovery": True,
        }
        del whole, resumed, partial
        gc.collect()
        torch.cuda.empty_cache()
    save(
        ROOT / "actual-H100-recovery.json",
        {
            "status": "actual_new_smooth_FP16_H100_complete_trainer_recovery_accepted",
            "job_id": os.environ["SLURM_JOB_ID"],
            "source_git": LIBRARY_GIT,
            "CPU_receipt_sha256": sha(ROOT / "CPU-accepted.json"),
            "roles": results,
            "diagnostic_only": True,
            "clinical_accuracy_evidence": False,
        },
    )


def training_only_review(inp, cfg):
    """Training-only contribution/mask/background stress; never choose by tiny AUC."""
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

    frame, binding = load_verified_release(Path(inp["studies"]), Path(inp["artifact-root"]))
    folds = load_folds(Path(inp["folds"]), frame, binding)
    train, outer = select_development(frame, folds, 0)
    chosen = train.head(3)
    if set(chosen.study_uid) & set(outer.study_uid):
        raise ValueError("Training-only engineering review required")
    c = TrainConfig(**cfg["candidate"])
    torch.manual_seed(c.seed)
    encoder, dim, ep = create_encoder(c.encoder)
    configure_encoder_training(encoder, c.freeze_encoder)
    model = CompetitiveStudyNet(
        encoder,
        dim,
        pooling=c.pooling,
        hidden_dim=c.hidden_dim,
        encode_chunk_size=c.encode_chunk_size,
        slice_context=c.slice_context,
        witness=c.witness,
    )
    guard = FrozenBaseDelta(model, ep)
    cp = torch.load(
        ROOT / "candidate-engineering-full/checkpoint.pt", map_location="cpu", weights_only=True
    )
    original = json.loads((ORIGINAL / "cpu-accepted.json").read_text())
    if cp["binding"]["training_uids_sha256"] != original["training_16_sha256"]:
        raise ValueError("Original engineering weights required")
    restore_checkpoint_model(model, cp, guard)
    del cp, guard
    model = model.to("cuda").eval()
    captured = {}

    def hook(module, arguments, output):
        captured["local"] = output.detach().float().cpu()
        patches, mask = arguments
        raw = module.score(module.projection(patches)).detach().float()
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite genuine local evidence")
        captured["scores"] = {
            "genuine_slices": int(mask.sum()),
            "genuine_patch_cells": raw.numel(),
            "minimum": raw.min().item(),
            "maximum": raw.max().item(),
            "mean": raw.mean().item(),
            "std": raw.std().item(),
            "patch_grid": list(raw.shape[1:3]),
        }

    handle = model.witness.register_forward_hook(hook)
    series = pd.read_csv(inp["series"])
    data = CompetitiveStudyDataset(
        chosen, series, Path(inp["cache"]), slices=c.slices, side=c.side, max_series=c.max_series
    )
    rows = []
    try:
        for index in range(len(data)):
            batch = collate_studies([data[index]])
            values = {
                k: batch[k].to("cuda")
                for k in ("images", "features", "series_mask", "slice_mask", "positions")
            }

            def forward(v, current_model=model):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    out = current_model(**v)
                if not torch.isfinite(out).all():
                    raise ValueError("Nonfinite engineering stress output")
                return out.float().cpu()

            normal = forward(values)
            local = captured["local"]
            scores = dict(captured["scores"])
            context = normal - local
            padding = {k: v.clone() for k, v in values.items()}
            padding["images"][~padding["slice_mask"]] = float("nan")
            padding["positions"][~padding["slice_mask"]] = float("nan")
            padding["features"][~padding["series_mask"]] = float("nan")
            padded = forward(padding)
            if not torch.equal(normal, padded):
                raise ValueError("Invalid padding changed genuine MRI prediction")
            black = dict(values, images=torch.zeros_like(values["images"]))
            background = forward(black)
            missing = None
            if int(values["series_mask"].sum()) > 1:
                reduced = {k: v.clone() for k, v in values.items()}
                first = int(torch.nonzero(reduced["series_mask"][0], as_tuple=False)[0, 0])
                reduced["series_mask"][0, first] = False
                reduced["slice_mask"][0, first] = False
                missing = forward(reduced)
            rows.append(
                {
                    "engineering_training_index": index,
                    "genuine_local_scores": scores,
                    "max_abs_local_logit": local.abs().max().item(),
                    "max_abs_context_logit": context.abs().max().item(),
                    "NaN_padding_prediction_exactly_equal": True,
                    "black_background_max_probability_change": (
                        normal.sigmoid() - background.sigmoid()
                    )
                    .abs()
                    .max()
                    .item(),
                    "missing_one_series_max_probability_change": (
                        normal.sigmoid() - missing.sigmoid()
                    )
                    .abs()
                    .max()
                    .item()
                    if missing is not None
                    else None,
                }
            )
    finally:
        handle.remove()
    save(
        ROOT / "training-only-review.json",
        {
            "status": "actual_training_only_local_contributions_and_finite_mask_background_stresses_reviewed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "training_roster_sha256": _uid_hash(chosen),
            "checkpoint_sha256": sha(ROOT / "candidate-engineering-full/checkpoint.pt"),
            "rows": rows,
            "diagnostic_AUC_used_for_recipe_selection": False,
            "anatomical_or_shortcut_validity_established": False,
            "clinical_accuracy_evidence": False,
            "limitation": "Finite/local mask checks do not adjudicate lesion location; background/series sensitivity remains an experimental risk assessed at the frozen held-out endpoint.",
        },
    )
    del model, encoder, data
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    os.umask(0o077)
    if len(sys.argv) != 2 or sys.argv[1] not in ("stage", "fit"):
        raise ValueError("Exact independent stage or fit required")
    stage() if sys.argv[1] == "stage" else fit()
