"""Exercise both admitted spatial-input roles in six fresh real CPU fit processes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.regional_context_train import ENGINE, ROLES
from rsna.competitive.released_context import tta_probabilities
from rsna.competitive.released_context_train import Recipe, Trainer, batch, digest
from scripts.newton.run_regional_context_fit import load_data


def same(a, b):
    """Compare all decoded learned/optimizer/counter/sampler/RNG/history state."""
    if type(a) is not type(b):
        raise ValueError("Decoded recovery state types differ")
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError("Decoded tensor state differs")
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError("Decoded NumPy RNG state differs")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Complete recovery state keys differ")
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Complete recovery state lengths differ")
        for x, y in zip(a, b, strict=True):
            same(x, y)
    elif a != b:
        raise ValueError("Decoded scalar state differs")


def model_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def zero_start_teacher(data, recipe):
    """Use the prediction path's exact batch shape, view layout and reduction.

    Scalar-versus-vector CPU sigmoid kernels can differ by one ULP. Rebatching
    the oracle removes that verification artifact without relaxing equality or
    altering the model, teacher values, fitting loop or precision policy.
    """
    result = []
    for start in range(0, len(data), recipe.batch_size):
        indices = list(range(start, min(start + recipe.batch_size, len(data))))
        x, _, _ = batch(data, indices)
        result.append(tta_probabilities(x[1].view(-1, 2, recipe.targets)).numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Fresh CPU-only recovery output required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = {r: load_data(args.inputs, args.inputs_sha256, r)[0] for r in ROLES}
    source = json.loads(args.inputs.read_text())
    if source["population_stage"] != "engineering32" or any(len(d) != 32 for d in data.values()):
        raise ValueError("Exact actual32 coherent engineering inputs required")
    recipe = Recipe(epochs=2, embedding_dim=8192)
    args.output.mkdir(mode=0o700, parents=True)
    recipe_path = args.output / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.__dict__, sort_keys=True))
    hashes = {role: hashlib.sha256() for role in ROLES}
    changed = 0
    for index in range(32):
        a, b = (data[role][index] for role in ROLES)
        for i in range(1, 6):
            same(a[i], b[i])
        if not torch.equal(a[6].isnan(), b[6].isnan()):
            raise ValueError("Source missing-value masks differ")
        same(a[6].nan_to_num(), b[6].nan_to_num())
        same(a[7], b[7])
        if a[0].shape != (2, 94, 8192) or not torch.equal(a[0][..., :1024], b[0][..., :1024]):
            raise ValueError("The exact coherent globals must be shared")
        changed += int(not torch.equal(a[0], b[0]))
        for role, sample in zip(ROLES, (a, b), strict=True):
            hashes[role].update(sample[0].numpy().tobytes())
    if changed != 32:
        raise ValueError("Both real input representations must actually be exercised")
    initial = {}
    initial_training_states = []
    for role in ROLES:
        trainer = Trainer(data[role], recipe)
        if sum(p.numel() for p in trainer.model.parameters()) != 1334796:
            raise ValueError("Same original matched parameter graph required")
        initial[role] = model_hash(trainer.model.state_dict())
        teacher = zero_start_teacher(data[role], recipe)
        if not np.array_equal(trainer.predictions(), teacher):
            raise ValueError("Zero-start must preserve the actual parent exactly")
        state = copy.deepcopy(trainer.state())
        state.pop("binding")  # Only the registered input-role binding differs.
        initial_training_states.append(state)
    if len(set(initial.values())) != 1:
        raise ValueError("Initial complete learned states differ")
    same(*initial_training_states)
    initial_receipt = {
        "input_tensor_sha256": {r: h.hexdigest() for r, h in hashes.items()},
        "initial_model_sha256": initial,
        "same_parent_globals_logits_metadata_labels_masks": True,
        "all32_input_tensors_differ": True,
        "zero_start_exact_actual_parent": True,
        "initial_non_binding_training_state_identical": True,
        "parameters_each_role": 1334796,
    }
    (args.output / "initial-admitted.json").write_text(
        json.dumps(initial_receipt, sort_keys=True, indent=2)
    )
    runner = Path(__file__).with_name("run_regional_context_fit.py")
    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    results = {}
    for role in ROLES:
        phases = []
        for phase in ("full", "stop", "resume"):
            output = args.output / role / phase
            command = [
                sys.executable,
                "-B",
                str(runner),
                "--inputs",
                str(args.inputs),
                "--inputs-sha256",
                args.inputs_sha256,
                "--role",
                role,
                "--recipe",
                str(recipe_path),
                "--output",
                str(output),
            ]
            if phase == "stop":
                command += ["--stop-after", "1"]
            if phase == "resume":
                checkpoint = args.output / role / "stop/checkpoint.pt"
                command += ["--resume", str(checkpoint), "--resume-sha256", digest(checkpoint)]
            log = args.output / f"{role}-{phase}.private.log"
            with log.open("x") as handle:
                code = subprocess.run(
                    command, env=env, stdout=handle, stderr=subprocess.STDOUT
                ).returncode
            phases.append({"phase": phase, "returncode": code})
            if code:
                raise RuntimeError("Original real CPU recovery phase failed; inspect without retry")
            terminal = json.loads((output / "terminal.json").read_text())
            if phase == "stop" and (
                terminal["status"] != "stopped_at_update_boundary"
                or terminal["completed_updates"] != 1
                or terminal["epochs"] != 0
            ):
                raise ValueError("Actual stop-after-one outcome is required")
        states = []
        for phase in ("full", "resume"):
            p = args.output / role / phase
            terminal = json.loads((p / "terminal.json").read_text())
            if (
                terminal["engine"] != ENGINE
                or terminal["role"] != role
                or terminal["status"] != "fixed_final_complete"
                or terminal["completed_updates"] != 4
                or terminal["diagnostic_only"] is not True
                or digest(p / "checkpoint.pt") != terminal["checkpoint_sha256"]
            ):
                raise ValueError("Actual completed original checkpoint and terminal required")
            states.append(torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=False))
        same(*states)
        state = states[0]
        if (
            state["completed"] != 4
            or state["epoch"] != 2
            or state["cursor"] != 0
            or state["order"] is not None
            or state["binding"] != data[role].binding
            or len(state["history"]) != 4
            or [r["update"] for r in state["history"]] != [1, 2, 3, 4]
            or any(
                not np.isfinite(
                    [r["loss"], r["gradient_norm"], r["lr"], r["max_abs_correction"]]
                ).all()
                or not 0 <= r["max_abs_correction"] <= 0.5
                for r in state["history"]
            )
        ):
            raise ValueError("Actual complete finite four-update terminal boundary required")
        full, resumed = (
            args.output / role / p / "predictions.private.npy" for p in ("full", "resume")
        )
        if full.read_bytes() != resumed.read_bytes() or np.load(full, allow_pickle=False).shape != (
            32,
            12,
        ):
            raise ValueError("Exact complete32x12 prediction bytes required")
        if not states[0]["model"]["output"].count_nonzero():
            raise ValueError("Actual nonzero learning required")
        results[role] = {
            "phases": phases,
            "complete_decoded_state_exact": True,
            "all32x12_prediction_bytes_exact": True,
            "completed_updates": 4,
            "actual_nonzero_learning": True,
        }
    retained = sum(p.stat().st_size for p in args.output.rglob("*.pt"))
    if any(p.is_dir() for p in args.output.rglob("epochs")):
        raise ValueError("Engineering recovery must not create duplicate epoch checkpoints")
    if retained > 112 * 2**20:
        raise ValueError("Engineering checkpoints exceed the admitted112MiB sub-bound")
    receipt = {
        "status": "actual32_both_regional_roles_complete_CPU_recovery_accepted",
        "engine": ENGINE,
        "CPU_job": os.environ.get("SLURM_JOB_ID"),
        "input_manifest_sha256": args.inputs_sha256,
        "initial": initial_receipt,
        "roles": results,
        "fresh_training_processes": 6,
        "engineering_checkpoint_bytes": retained,
        "two_production_roles_peak_bound_bytes": retained + (22 * 17 + 17 + 16) * 2**20,
        "two_production_roles_within_existing512MiB": retained + (22 * 17 + 17 + 16) * 2**20
        <= 512 * 2**20,
        "no_engineering_epoch_copies_created": True,
        "fitting_source_probabilities_used": True,
        "clinical_or_Kaggle_accuracy": False,
        "full3887_fit": False,
        "CNN_parameters_fitted": 0,
    }
    (args.output / "accepted.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
