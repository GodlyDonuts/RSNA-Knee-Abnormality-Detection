"""Read-only CPU head-objective measurement of the original retained C24 state."""

import ast
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import resource
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCIENCE = "39e315c65546aa5205518713bc27349ec63e7538"
ACCEPTED_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
ROOT = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
PORTABLE = Path("/tmp/rsna-predictive-portable-recovery-20261020-v1")
PINNED = {
    "worker.py": "78ba933ae78a7d6c3b7199e9f92d0f799c9d77dced13a341968e36e632eb641e",
    "rsna/competitive/predictive/model.py":
        "38c9fc7c2e7e79ea90972f1b14c6e47491aba2f1641496277534e7cd20856f08",
    "rsna/competitive/predictive/train.py":
        "066dc5eda8ca8a793548e6fde62c2c6a06480f052d9034aa3fe2e08a5004dc58",
    "rsna/competitive/predictive/density.py":
        "b5f56767dd277af65864983f3a5cd58d89d4cc8153990e9139e6f81b870e64a0",
    "rsna/competitive/predictive/context_cache.py":
        "bf9d59b26bd8a28f4533648d5b10d6c77bd5caf8d2f69de0b77968ce4d4ea3b5",
    "rsna/competitive/predictive/cache.py":
        "f82c478e0d4632ed31b491e4915066a47adf8275a7c2ba6fb42b9ca99871f944",
}
GAUSSIAN_CONSTANT = 0.5 * math.log(2 * math.pi)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def stat(path):
    value = path.stat()
    assert not path.is_symlink() and value.st_uid == os.getuid()
    return [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def gradient_balance(exam, weighted_density, total, named_tensors):
    """Actual vector decomposition; norm differences are not gradient estimates."""
    names, tensors = zip(*named_tensors)
    gradients = [
        torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
        for loss in (exam, weighted_density, total)
    ]
    groups = {"all_nonencoder_parameters": [i for i, n in enumerate(names) if not n.startswith("h.")]}
    for name in names:
        group = name.split(".", 1)[0]
        groups.setdefault(group, []).append(names.index(name))
    result = {}
    for group, indices in groups.items():
        squares = [0.0, 0.0, 0.0]
        dot = error2 = maximum_error = 0.0
        for i in indices:
            values = [
                torch.zeros_like(tensors[i], dtype=torch.float64)
                if g[i] is None else g[i].detach().double()
                for g in gradients
            ]
            assert all(torch.isfinite(x).all() for x in values)
            for j, value in enumerate(values):
                squares[j] += float(value.square().sum())
            dot += float((values[0] * values[1]).sum())
            error = values[2] - values[0] - values[1]
            error2 += float(error.square().sum())
            maximum_error = max(maximum_error, float(error.abs().max()))
        norms = [math.sqrt(x) for x in squares]
        result[group] = {
            "exam_gradient_L2": norms[0],
            "weighted_density_gradient_L2": norms[1],
            "total_gradient_L2": norms[2],
            "weighted_density_over_exam_L2": norms[1] / norms[0] if norms[0] else None,
            "cosine": dot / (norms[0] * norms[1]) if norms[0] and norms[1] else None,
            "sum_identity_max_abs_error": maximum_error,
            "sum_identity_relative_L2_error": math.sqrt(error2) / norms[2] if norms[2] else None,
        }
        # Independent backward traversals can differ by FP32 accumulation order.
        assert math.sqrt(error2) <= 2e-5 * max(norms[2], norms[0] + norms[1], 1e-12)
    return result


def supported_margin(log_bayes_factor, labels, mask, observed_count, width):
    """Wrong-label NLL minus supported-label NLL, per observed descriptor."""
    if observed_count == 0 or not mask.any():
        return None
    values = (2 * labels[mask] - 1) * log_bayes_factor[mask] / (observed_count * width)
    return float(values.detach().mean())


def prior_exam(context_logits, labels, mask):
    """Same BCE-of-mean rule as training, with evidence removed only at readout."""
    z = torch.stack(context_logits)[None]
    q = torch.where(mask, labels.detach(), 0)
    logp = torch.logsumexp(torch.nn.functional.logsigmoid(z), 1) - math.log(2)
    log1p = torch.logsumexp(torch.nn.functional.logsigmoid(-z), 1) - math.log(2)
    return (
        (-(q * logp + (1 - q) * log1p)).masked_fill(~mask, 0).sum()
        / mask.sum().clamp_min(1),
        logp.exp(),
    )


def tensor_summary(value):
    value = value.detach().double()
    assert value.numel() and torch.isfinite(value).all()
    return {
        "L2": float(value.norm()),
        "RMS": float(value.square().mean().sqrt()),
        "maximum_absolute": float(value.abs().max()),
    }


def diagnose_case(model, contexts, labels, mask, ordinal, two_mask_loss):
    """No .backward(), optimizer, parameter writes, or encoder-gradient claim."""
    assert len(contexts) == 2 and ordinal in (0, 1)
    h_values = []
    hook = model.negative.register_forward_pre_hook(lambda module, args: h_values.append(args[0]))
    try:
        outputs = [
            model(
                *(c[n] for n in ("images", "token_metadata", "region_metadata", "targets", "observed")),
                c["supports"],
            )
            for c in contexts
        ]
    finally:
        hook.remove()
    assert len(h_values) == 2
    evidences = [o["evidence"] for o in outputs]
    total, metrics = two_mask_loss(
        evidences, torch.stack([o["context_logits"] for o in outputs])[None],
        labels, mask, [c["observed"][None] for c in contexts], model.descriptor_width, 0.1,
    )
    named = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert named and all(p.requires_grad and p.dtype == torch.float32 for _, p in named)
    balance = gradient_balance(
        metrics["exam_loss"], 0.1 * metrics["density_loss"], total,
        named + [(f"h.{i}", h) for i, h in enumerate(h_values)],
    )
    prior_loss, prior_probability = prior_exam([o["context_logits"] for o in outputs], labels, mask)
    probability = metrics["probabilities"].detach()
    per_context = []
    for i, (context, output) in enumerate(zip(contexts, outputs)):
        observed = context["observed"]
        count = int(observed.sum())
        targets = context["targets"][observed].double()
        e = output["evidence"]
        q = torch.where(mask, labels, 0)
        active = mask & (count > 0)
        normalized_nll = -(q * e.log_p1 + (1 - q) * e.log_p0) / max(count * 64, 1)
        density = float(normalized_nll[active].detach().mean()) if active.any() else None
        images = context["images"]
        per_context.append({
            "context_ordinal": i, "observed_regions": count,
            "complete_regions": len(observed),
            "weak_label_weighted_supported_density_NLL": density,
            "density_minus_fixed_gaussian_constant": density - GAUSSIAN_CONSTANT
            if density is not None else None,
            "wrong_minus_supported_density_NLL": supported_margin(
                e.log_bayes_factor, labels, mask, count, model.descriptor_width,
            ),
            "log_bayes_factor": tensor_summary(e.log_bayes_factor),
            "context_prior_logits": tensor_summary(output["context_logits"]),
            "descriptor_RMS": float(targets.square().mean().sqrt()) if count else None,
            "descriptor_variance_over_all_observed_coordinates": float(targets.var(correction=0))
            if count else None,
            "image_stats_include_mask_fill_not_unmasked_contrast": {
                "mean": float(images.mean()), "std": float(images.std(correction=0)),
                "minimum": float(images.min()), "maximum": float(images.max()),
            },
        })
    return {
        "fitting_ordinal": ordinal, "observed_label_count": int(mask.sum()),
        "exam_loss": float(metrics["exam_loss"].detach()),
        "density_loss": float(metrics["density_loss"].detach()),
        "density_minus_fixed_gaussian_constant": float(metrics["density_loss"].detach())
        - GAUSSIAN_CONSTANT,
        "weighted_density_loss": float((0.1 * metrics["density_loss"]).detach()),
        "total_loss": float(total.detach()),
        "loss_sum_identity_abs_error": abs(float(
            (total - metrics["exam_loss"] - 0.1 * metrics["density_loss"]).detach()
        )),
        "context_only_exam_loss": float(prior_loss.detach()),
        "full_minus_context_only_exam_loss": float((metrics["exam_loss"] - prior_loss).detach()),
        "mean_absolute_posterior_change_all_targets": float(
            (probability - prior_probability.detach()).abs().mean()
        ),
        "posterior_saturation_fraction_all_targets_at_1e_minus4": float(
            ((probability <= 1e-4) | (probability >= 1 - 1e-4)).float().mean()
        ),
        "gradient_balance": balance, "contexts": per_context,
    }


def run(payload):
    started = time.monotonic()
    assert os.uname().nodename == "evc27"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not os.environ.get("SLURM_JOB_GPUS")
    assert os.environ["SLURM_CPUS_PER_TASK"] == "2" and sys.dont_write_bytecode
    assert shutil.disk_usage("/tmp").free >= 20 * 2**30
    assert payload["fitting_indices"] == [0, 1]
    assert payload["original_engineering_receipt_sha256"] == ACCEPTED_SHA
    for field, length in (("source_git", 40), ("diagnostic_sha256", 64)):
        assert len(payload[field]) == length and set(payload[field]) <= set("0123456789abcdef")
    accepted_body = (ROOT / "GPU-accepted.json").read_bytes()
    assert sha(accepted_body) == ACCEPTED_SHA
    accepted = json.loads(accepted_body)
    assert accepted == payload["original_engineering_receipt"]
    assert accepted["source_git"] == SCIENCE and accepted["GPU_job"] == "978709"
    assert accepted["status"] == "actual_zero_density_mean_production_warmup_B_C_C0_recovery_complete"
    original_c = next(a for a in accepted["arms"] if a["arm"] == "C")
    assert original_c["full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
    binding_body = (ROOT / "binding.private.json").read_bytes()
    binding = json.loads(binding_body)
    assert binding["source_git"] == SCIENCE and binding["zero_initial_density_means"] is True
    assert all(binding["code_sha256"][n] == h for n, h in PINNED.items())
    for name, expected in binding["code_sha256"].items():
        path = ROOT / "code" / name
        assert path.resolve().is_relative_to((ROOT / "code").resolve())
        assert sha(path.read_bytes()) == expected
    for field in ("targets16", "context_index"):
        assert sha(Path(binding[field + "_private_path"]).read_bytes()) == binding[field + "_sha256"]
    assert not any(n == "rsna" or n.startswith("rsna.") for n in sys.modules)
    sys.path[:0] = [str(ROOT / "code"), str(PORTABLE / "overlay")]
    import timm

    from rsna.competitive.predictive.context_cache import load
    from rsna.competitive.predictive.density import two_mask_loss
    from rsna.competitive.predictive.model import Predictive
    from rsna.competitive.predictive.train import Recipe

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 0
    source = (PORTABLE / "assets/parent_source.private.py").read_text()
    assert sha(source.encode()) == binding["parent_source_sha256"]
    weight = Path(binding["parent_weight_private_path"])
    wstat = stat(weight)
    assert {"bytes": wstat[1], "mtime_ns": wstat[2], "inode": wstat[0]} == binding["parent_weight_stat"]
    ns = dict(torch=torch, nn=torch.nn, timm=timm, gc=gc, hashlib=hashlib, INPUT_RES=320)
    selected = [
        n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in ("build_backbone", "RaptorClassifier", "load_model")
    ]
    assert len(selected) == 3
    exec(compile(ast.Module(body=selected, type_ignores=[]), "original_parent_AST", "exec"), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        parent, resolution = ns["load_model"](
            str(weight), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384, torch.device("cpu"),
        )
    assert resolution == 320 and sum(p.numel() for p in parent.parameters()) == 73136096
    encoder_ast = [
        n for n in ast.parse((ROOT / "code/worker.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "FixedEncoder"
    ]
    assert len(encoder_ast) == 1
    exec(compile(ast.Module(body=encoder_ast, type_ignores=[]), "original_FixedEncoder_AST", "exec"), ns)
    model = Predictive(ns["FixedEncoder"](parent.backbone), parent.backbone.num_features, hidden=128)
    del parent
    checkpoint = ROOT / "engineering/C/full/final.pt"
    before = stat(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recipe = Recipe("C", epochs=12, effective_batch=8, head_only_updates=3,
                    schedule_updates=2916, head_lr=3e-5, density_weight=0.1)
    assert saved["binding"] == binding and saved["recipe"] == asdict(recipe)
    assert saved["step"] == 24 and saved["cohort_size"] == 12 and saved["epoch_snapshots"] is False
    assert saved["history"] == original_c["real_optimizer_history"]
    assert [row["step"] for row in saved["history"]] == list(range(1, 25))
    terminal_body = (ROOT / "engineering/C/full/terminal.json").read_bytes()
    assert json.loads(terminal_body) == original_c["terminal"]
    original_state = saved["model"]
    del saved
    model.load_state_dict(original_state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == 73271124
    assert torch.count_nonzero(model.fixed_log_variance) == 0
    model.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    encoder_forward = model.encoder.forward

    def detached_encoder(*args, **kwargs):
        with torch.no_grad():
            return encoder_forward(*args, **kwargs).detach()

    model.encoder.forward = detached_encoder
    index = json.loads(Path(binding["context_index_private_path"]).read_text())
    assert len(index) == 16
    with np.load(binding["targets16_private_path"], allow_pickle=False) as arrays:
        assert arrays["labels"].shape == arrays["masks"].shape == (16, 12)
        labels = torch.from_numpy(arrays["labels"][:2].copy()).float()
        masks = torch.from_numpy(arrays["masks"][:2].copy()).bool()
    cases = []
    for ordinal in (0, 1):
        row = index[ordinal]
        contexts = load(row["path"], row["proof"], provenance=row["provenance"])
        cases.append(diagnose_case(
            model, contexts, labels[ordinal:ordinal + 1], masks[ordinal:ordinal + 1],
            ordinal, two_mask_loss,
        ))
        del contexts
        gc.collect()
        assert time.monotonic() - started < 1800
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, original_state[n]) for n, v in model.state_dict().items())
    assert stat(checkpoint) == before and stat(weight) == wstat
    assert (ROOT / "binding.private.json").read_bytes() == binding_body
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux owning node.
    assert peak_rss <= 12 * 2**30
    return {
        "status": "actual_retained_C24_CPU_objective_balance",
        "source_git": payload["source_git"], "diagnostic_sha256": payload["diagnostic_sha256"],
        "original_engineering_receipt_sha256": ACCEPTED_SHA,
        "original_science_git": SCIENCE, "original_GPU_job": "978709",
        "CPU_job": os.environ["SLURM_JOB_ID"], "checkpoint_step": 24,
        "checkpoint_stat_before_and_after": before, "checkpoint_model_state_unchanged": True,
        "binding_sha256": sha(binding_body), "terminal_sha256": sha(terminal_body),
        "original_module_sha256": PINNED, "recipe": asdict(recipe),
        "fitting_studies": 2, "contexts_per_study": 2, "remote_files_written": 0,
        "optimizer_steps": 0, "new_source_reads_or_rendering": False,
        "encoder_gradients_measured": False, "encoder_forward_detached": True,
        "measurement_precision": "CPU_FP32_eval_not_original_BF16_GPU_training",
        "gradient_scope": "nonencoder_parameters_and_context_h_before_any_clipping_or_Adam",
        "clinical_accuracy_or_first200_training_claim": False,
        "Gaussian_fixed_unit_variance_constant": GAUSSIAN_CONSTANT,
        "peak_process_RSS_bytes": peak_rss, "elapsed_seconds": time.monotonic() - started,
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
