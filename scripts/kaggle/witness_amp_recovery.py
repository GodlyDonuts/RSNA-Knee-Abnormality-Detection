"""Private two-T4 canonical-trainer recovery after actual original-MRI admission.

FP16/native384/chunk8 with checkpointed loss scaling is an explicit recipe. Completed diagnostics
are neither clinical checkpoints nor competition candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The generated notebook supplies immutable complete overlay/code hashes.
CODE_OVERRIDES = globals().get("CODE_OVERRIDES", {})
CODE_OVERRIDE_HASHES = globals().get("CODE_OVERRIDE_HASHES", {})
SCIENCE_GIT = globals().get("SCIENCE_GIT", "")
BOOTSTRAP_SOURCE = None
if __name__ == "__main__":
    import hashlib

    if (
        set(CODE_OVERRIDES) != {"rsna/competitive/train.py", "rsna/competitive/witness.py"}
        or set(CODE_OVERRIDE_HASHES) != set(CODE_OVERRIDES)
        or len(SCIENCE_GIT) != 40
    ):
        raise ValueError("Exact archived AMP/selector overlay required")
    control = Path(sys.argv[sys.argv.index("--control") + 1])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    role = sys.argv[sys.argv.index("--role") + 1]
    BOOTSTRAP_SOURCE = output.parent / (role + "-scientific-source")
    BOOTSTRAP_SOURCE.mkdir(mode=0o700)
    proof = json.loads((control / "original-input-proof.json").read_text())
    SOURCE_HASHES = dict(proof["source_sha256"], **CODE_OVERRIDE_HASHES)
    for name, sha in proof["source_sha256"].items():
        raw = (control / "code" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Original unchanged scientific source differs")
        if name in CODE_OVERRIDES:
            raw = CODE_OVERRIDES[name].encode()
        if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError("Immutable new numerical source differs")
        target = BOOTSTRAP_SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    sys.path.insert(0, str(BOOTSTRAP_SOURCE))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rsna.competitive.checkpoint import file_sha256, save_checkpoint  # noqa: E402
from rsna.competitive.train import TrainConfig  # noqa: E402
from rsna.schema import TARGETS  # noqa: E402
from scripts.newton.probe_patch_witness_trainer_recovery import (  # noqa: E402
    compare_recovery,
    digest,
    numerical_updates,
)


def validate_recipe(values, role, original):
    expected = dict(
        original, precision="fp16", encode_chunk_size=8, min_free_gib=3, amp_init_scale=128
    )
    expected["witness"] = dict(original["witness"], pooling="normalized_logmeanexp", beta=8)
    # Physical mount paths are resolved separately; no other policy change.
    ignored_paths = {"config_path", "weights_path"}
    if {k: v for k, v in values["encoder"].items() if k not in ignored_paths} != {
        k: v for k, v in original["encoder"].items() if k not in ignored_paths
    }:
        raise ValueError("Pinned official encoder/adapters differ from CPU admission")
    expected["encoder"] = values["encoder"]
    if values != expected or role not in {"control", "candidate"}:
        raise ValueError("Only admitted explicit FP16/chunk8 engineering recipe allowed")
    TrainConfig(**values).validate()
    if (
        values["epochs"] != 2
        or values["diagnostic_study_limit"] != 16
        or values["side"] != 384
        or values["witness"]
        != {
            "patches": 4,
            "slices": 2,
            "seed": 20261003,
            "weight": 0 if role == "control" else 1,
            "pooling": "normalized_logmeanexp",
            "beta": 8,
        }
    ):
        raise ValueError("Exact two-epoch/four-update signed-patch experiment required")


def validate_predictions(frame, roster_sha256, fold, seed):
    """The trainer's research CSV includes fold/seed provenance beside scores."""
    if (
        list(frame.columns) != ["study_uid", "fold", "seed", *TARGETS]
        or len(frame) != 16
        or frame.study_uid.duplicated().any()
        or digest(sorted(frame.study_uid.tolist())) != roster_sha256
        or not frame.fold.eq(fold).all()
        or not frame.seed.eq(seed).all()
    ):
        raise ValueError("Original16x12 MRI-only research output and provenance required")
    scores = torch.tensor(frame[list(TARGETS)].to_numpy())
    if not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Finite actual MRI probabilities required")


def run(args):
    started = time.monotonic()
    if (
        torch.cuda.device_count() != 2
        or any("T4" not in torch.cuda.get_device_name(i) for i in (0, 1))
        or args.role not in {"control", "candidate"}
        or args.device_index != (0 if args.role == "control" else 1)
        or not 0 < args.remaining_seconds <= 14400
    ):
        raise ValueError("Exact private two-T4 bounded engineering placement required")
    if file_sha256(args.cpu_acceptance) != args.cpu_acceptance_sha256:
        raise ValueError("Actual CPU input acceptance receipt changed")
    accepted = json.loads(args.cpu_acceptance.read_text())
    if (
        accepted["status"]
        != "actual_private_Kaggle_original32_MRI_full_release_and_official_initial_state_CPU_accepted"
        or accepted["input_tensors_equal_to_original"] is not True
        or accepted["labels_masks_equal_to_original"] is not True
        or accepted["GPU_or_optimizer_executed"] is not False
        or accepted["source_git"] != "d3cf58242455bf96e46b0c2071e8e9fe16cf4aaf"
        or accepted["initial_learned_state"]["control"]
        != accepted["initial_learned_state"]["candidate"]
    ):
        raise ValueError("Original complete MRI/mask/model/RNG acceptance required")
    if shutil.disk_usage(args.output.parent).free < 3 * 2**30 + 6 * 2**30:
        raise ValueError("Kaggle scratch must preserve3GiB plus6GiB diagnostic envelope")
    args.output.mkdir(mode=0o700)
    proof = json.loads((args.control / "original-input-proof.json").read_text())
    for name, sha in SOURCE_HASHES.items():
        if file_sha256(BOOTSTRAP_SOURCE / name) != sha:
            raise ValueError("Original admitted scientific source changed")
    values = json.loads((args.control / (args.role + ".json")).read_text())
    values.update(precision="fp16", encode_chunk_size=8, min_free_gib=3, amp_init_scale=128)
    values["witness"] = dict(values["witness"], pooling="normalized_logmeanexp", beta=8)
    original = proof["expected_config"][args.role]
    spec = json.loads(args.encoder_spec.read_text())
    values["encoder"] = spec
    validate_recipe(values, args.role, original)
    import gc
    import hashlib
    import random

    import numpy as np

    from rsna.competitive.adapters import configure_encoder_training
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet

    random.seed(values["seed"])
    np.random.seed(values["seed"] % 2**32)
    torch.manual_seed(values["seed"])
    encoder, dim, provenance = create_encoder(spec)
    configure_encoder_training(encoder, values["freeze_encoder"])
    model = CompetitiveStudyNet(
        encoder,
        dim,
        pooling=values["pooling"],
        hidden_dim=values["hidden_dim"],
        encode_chunk_size=values["encode_chunk_size"],
        slice_context=values["slice_context"],
        witness=values["witness"],
    )
    records = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            array = param.detach().contiguous().cpu().numpy()
            records[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    prior = accepted["initial_learned_state"][args.role]
    if (
        digest(records) != prior["learned_graph_and_values_sha256"]
        or hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        != prior["post_factory_torch_RNG_sha256"]
        or digest(provenance) != prior["encoder_provenance_sha256"]
    ):
        raise ValueError("Actual original official learned graph/values/RNG must remain equal")
    del model, encoder, records
    gc.collect()
    # Authenticate complete compatible base and source through canonical trainer;
    # the original all-development trained adapter checkpoint is never loaded.
    config = args.output / "config.json"
    config.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK=str(args.device_index),
        PYTHONPATH=str(BOOTSTRAP_SOURCE),
    )
    source = BOOTSTRAP_SOURCE
    # Original code is read-only and version-bound by the CPU source archive.
    from rsna.competitive import train

    if not Path(train.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("Recovery must import the original admitted scientific source")
    artifact = args.control / "artifacts"
    base = [
        sys.executable,
        "-B",
        str(source / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--device",
        "cuda",
        "--studies",
        str(artifact / "initiative/releases/report-supervision-v1/studies.parquet"),
        "--artifact-root",
        str(artifact),
        "--folds",
        str(artifact / "initiative/experiments/competitive-v1/folds"),
        "--series",
        str(
            artifact
            / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
        ),
        "--cache",
        str(args.cache),
    ]
    full, replay = args.output / "full", args.output / "replay"

    def execute(command, phase, code):
        if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 3 * 2**30:
            raise ValueError("One-arm3GiB diagnostic cap exceeded")
        remaining = args.remaining_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Private recovery whole-pipeline budget exhausted")
        print(
            json.dumps(
                {"role": args.role, "stage": phase, "elapsed_seconds": time.monotonic() - started}
            ),
            flush=True,
        )
        with (
            (args.output / (phase + ".stdout")).open("x") as out,
            (args.output / (phase + ".stderr")).open("x") as err,
        ):
            result = subprocess.run(
                command, env=env, cwd=source, stdout=out, stderr=err, timeout=remaining
            )
        if result.returncode != code:
            raise ValueError("Original canonical trainer failed; retain exact phase logs")

    execute(base + ["--output", str(full)], "full", 0)
    execute(base + ["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
    partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    first = json.loads((replay / "invocation.json").read_text())
    save_checkpoint(args.output / "interrupted-checkpoint.pt", partial, min_free_gib=3)
    (args.output / "interrupted-invocation.json").write_text(
        json.dumps(first, sort_keys=True, indent=2) + "\n"
    )
    execute(
        base + ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0
    )
    whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = json.loads((full / "invocation.json").read_text())
    last = json.loads((replay / "invocation.json").read_text())
    csv = (full / "outer_predictions.csv").read_bytes()
    state = compare_recovery(
        whole,
        resumed,
        partial,
        history,
        first,
        last,
        csv,
        (replay / "outer_predictions.csv").read_bytes(),
    )
    if whole["amp_scaler"]["_growth_tracker"] != 4 or whole["amp_scaler"]["scale"] < 1:
        raise ValueError("Actual four finite loss-scaled optimizer updates required")
    bound = whole["binding"]
    if (
        bound["training_uids_sha256"] != proof["training_16_sha256"]
        or bound["outer_uids_sha256"] != proof["outer_16_sha256"]
        or bound["release"] != accepted["release_binding"]
        or bound["config"]["precision"] != "fp16"
        or bound["config"]["encode_chunk_size"] != 8
        or bound["config"]["witness"] != values["witness"]
        or bound["config"]["amp_init_scale"] != 128
        or bound["code_sha256"]
        != {n.removeprefix("rsna/"): h for n, h in SOURCE_HASHES.items() if n.startswith("rsna/")}
    ):
        raise ValueError("Actual trainer source/input/recipe binding differs")
    frame = pd.read_csv(full / "outer_predictions.csv")
    validate_predictions(frame, proof["outer_16_sha256"], values["fold"], values["seed"])
    result = {
        "status": "actual_private_T4_FP16_smooth_witness_trainer_recovery_accepted",
        "role": args.role,
        "device_index": args.device_index,
        "CPU_receipt_sha256": args.cpu_acceptance_sha256,
        "complete_decoded_state_sha256": state,
        "checkpoint_sha256": file_sha256(full / "checkpoint.pt"),
        "predictions_sha256": file_sha256(full / "outer_predictions.csv"),
        "updates": numerical_updates(history),
        "optimizer_updates": 4,
        "exact_complete_model_base_AdamW_LR_cursors_all_RNG_and_CSV_recovery": True,
        "source_git": SCIENCE_GIT,
        "original_input_source_git": accepted["source_git"],
        "amp_scaler": whole["amp_scaler"],
        "official_initial_graph_RNG_encoder_provenance_equal": True,
        "source_sha256": SOURCE_HASHES,
        "precision": "fp16",
        "encode_chunk_size": 8,
        "elapsed_seconds": time.monotonic() - started,
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "competition_candidate": False,
    }
    (args.output / "recovery-accepted.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("control", "cache", "cpu-acceptance", "encoder-spec", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cpu-acceptance-sha256", required=True)
    p.add_argument("--role", choices=("control", "candidate"), required=True)
    p.add_argument("--device-index", type=int, required=True)
    p.add_argument("--remaining-seconds", type=float, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
