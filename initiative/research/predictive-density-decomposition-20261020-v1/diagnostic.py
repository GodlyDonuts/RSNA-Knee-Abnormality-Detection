"""Read-only matched C24/C10024 decomposition on fitting ordinals zero and one."""

import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

LOADER_SHA = "36a87becf4a79aeda7f6c491c82ff58c69b779902348f231648b5f00bec3aac0"
PRIOR_SHA = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
PILOT = Path("/tmp/rsna-predictive-density100-pilot-20261020-v1")
CONSTANT = 0.5 * math.log(2 * math.pi)
STATUS = "actual_matched_C24_C10024_fitting_density_decomposition"
BASELINE_STATUS = "actual_C24_only_fitting_density_decomposition"


def sha(body):
    return hashlib.sha256(body.encode() if isinstance(body, str) else body).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def components(evidences, labels, mask, observations, width):
    """Match original active-label/context averaging, including zero-observation masks."""
    q = torch.where(mask, labels.detach(), 0)
    a = evidences[0].log_p0.new_zeros(())
    d = a.clone()
    valid = 0
    contexts = []
    for e, observed in zip(evidences, observations):
        count = observed.sum(1, keepdim=True) * width
        active = mask & (count > 0)
        aa = -e.log_p0 / count.clamp_min(1)
        dd = -q * e.log_bayes_factor / count.clamp_min(1)
        a = a + aa.masked_fill(~active, 0).sum()
        d = d + dd.masked_fill(~active, 0).sum()
        valid += int(active.sum())
        margin = (2 * q - 1) * e.log_bayes_factor
        contexts.append({
            "observed_regions": int(observed.sum()),
            "density_denominator": int(count.item()),
            "active_labels": int(active.sum()),
            "signed_BF_margin_unnormalized": float(
                margin.masked_fill(~active, 0).sum().detach() / active.sum().clamp_min(1)),
            "wrong_minus_supported_density_NLL": float(
                (margin / count.clamp_min(1)).masked_fill(~active, 0).sum().detach()
                / active.sum().clamp_min(1)),
        })
    return a / max(valid, 1), d / max(valid, 1), contexts


def gradient_components(exam, a, d, actual_density, named):
    """autograd.grad does not populate .grad or perform an optimizer update."""
    names, tensors = zip(*named)
    grads = [torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
             for loss in (exam, a, d, actual_density)]
    groups = {"all_nonencoder_parameters": list(range(len(names)))}
    for i, name in enumerate(names):
        groups.setdefault(name.split(".", 1)[0], []).append(i)
    result = {}
    for group, indices in groups.items():
        squares = [0.0] * 4
        dots = [0.0] * 3
        error2 = 0.0
        for i in indices:
            values = [torch.zeros_like(tensors[i], dtype=torch.float64) if g[i] is None
                      else g[i].detach().double() for g in grads]
            require(all(bool(torch.isfinite(v).all()) for v in values), "Nonfinite gradients")
            for k, value in enumerate(values):
                squares[k] += float(value.square().sum())
            dots[0] += float((values[0] * values[1]).sum())
            dots[1] += float((values[0] * values[2]).sum())
            dots[2] += float((values[1] * values[2]).sum())
            error2 += float((values[1] + values[2] - values[3]).square().sum())
        norms = [math.sqrt(x) for x in squares]

        def ratio(n, denominator):
            return n / denominator if denominator else None

        require(math.sqrt(error2) <= 2e-5 * max(norms[1] + norms[2], norms[3], 1e-12),
                "A+D gradient differs from original density")
        result[group] = {
            "exam_L2": norms[0], "common_A_L2": norms[1],
            "label_weighted_D_L2": norms[2], "actual_density_L2": norms[3],
            "A_over_exam": ratio(norms[1], norms[0]),
            "D_over_exam": ratio(norms[2], norms[0]),
            "D_over_A": ratio(norms[2], norms[1]),
            "cosine_A_exam": ratio(dots[0], norms[0] * norms[1]),
            "cosine_D_exam": ratio(dots[1], norms[0] * norms[2]),
            "cosine_A_D": ratio(dots[2], norms[1] * norms[2]),
            "gradient_sum_absolute_L2_error": math.sqrt(error2),
            "gradient_sum_relative_L2_error": ratio(
                math.sqrt(error2), max(norms[1] + norms[2], norms[3])),
        }
    return result


def readout(logits, labels, mask):
    z = torch.stack(logits).double()
    logp = torch.logsumexp(F.logsigmoid(z), 0)[None] - math.log(2)
    log1p = torch.logsumexp(F.logsigmoid(-z), 0)[None] - math.log(2)
    q = torch.where(mask, labels.double(), 0)
    bce = (-(q * logp + (1 - q) * log1p)).masked_fill(~mask, 0).sum()
    return logp.exp(), float(bce / mask.sum().clamp_min(1))


@torch.no_grad()
def cached_image_statistics(images):
    """Descriptive cached-tensor aggregates; exact-fill exclusion is not anatomy masking."""
    values = images.detach().float()
    require(values.numel() > 0 and bool(torch.isfinite(values).all()), "Finite cached images")
    nonfill = values[values != 0.5]
    details = None
    if nonfill.numel():
        quantiles = torch.quantile(nonfill, nonfill.new_tensor([0.01, 0.5, 0.99]))
        details = {
            "mean": float(nonfill.mean()), "std_population": float(nonfill.std(correction=0)),
            "minimum": float(nonfill.min()), "maximum": float(nonfill.max()),
            "quantile_0.01": float(quantiles[0]), "quantile_0.5": float(quantiles[1]),
            "quantile_0.99": float(quantiles[2]),
        }
    return {
        "fraction_elements_exact_fill_0.5": 1 - nonfill.numel() / values.numel(),
        "nonfill_elements": nonfill.numel(), "nonfill_summary": details,
        "interpretation": "Excludes exact fill only; interpolation mixing and background remain. "
                          "This is not an anatomical visibility mask.",
        "raw_image_values_exported": False,
    }


def diagnose_case(model, contexts, labels, mask, ordinal, two_mask_loss, density_weight):
    from rsna.competitive.predictive.density import evidence

    require(ordinal in (0, 1) and len(contexts) == 2, "Only two original fitting cases")
    require(density_weight in (0.1, 100.0) and not model.training,
            "Matched original evaluation recipes required")
    require(torch.count_nonzero(model.fixed_log_variance) == 0, "Original unit covariance")
    evidences, priors, changed, image_statistics = [], [], [], []
    for context in contexts:
        image_statistics.append(cached_image_statistics(context["images"]))
        mu, delta, prior = model.predict_hypotheses(
            *(context[n] for n in ("images", "token_metadata", "region_metadata")))
        target, observed, supports = (context[n] for n in ("targets", "observed", "supports"))
        e = evidence(target[None], mu[None], delta[None], model.fixed_log_variance,
                     supports, observed[None])
        evidences.append(e)
        priors.append(prior)
        with torch.no_grad():
            mean_target = target.clone()
            if observed.any():
                mean_target[observed] = target[observed].mean(0)
            mean_e = evidence(mean_target[None], mu.detach()[None], delta.detach()[None],
                              model.fixed_log_variance, supports, observed[None])
            changed.append(prior.detach() + mean_e.log_bayes_factor[0])
    total, metrics = two_mask_loss(
        evidences, torch.stack(priors)[None], labels, mask,
        [c["observed"][None] for c in contexts], model.descriptor_width, density_weight)
    a, d, per_context = components(
        evidences, labels, mask, [c["observed"][None] for c in contexts], model.descriptor_width)
    for row, summary in zip(per_context, image_statistics):
        row["cached_image_statistics"] = summary
    gap = abs(float((a + d - metrics["density_loss"]).detach()))
    require(gap <= 2e-6, "A+D differs from original density loss")
    require(torch.allclose(total, metrics["exam_loss"] + density_weight * (a + d),
                           rtol=2e-6, atol=2e-6), "Original weighted objective differs")
    named = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]
    balance = gradient_components(metrics["exam_loss"], a, d, metrics["density_loss"], named)
    with torch.no_grad():
        full_p, full_bce = readout(
            [p + e.log_bayes_factor[0] for p, e in zip(priors, evidences)], labels, mask)
        mean_p, mean_bce = readout(changed, labels, mask)
        prior_p, prior_bce = readout(priors, labels, mask)
        require(torch.allclose(full_p, metrics["probabilities"].double(), atol=2e-7, rtol=2e-7)
                and abs(full_bce - float(metrics["exam_loss"])) <= 2e-6,
                "Mean-posterior readout differs from original objective")
    require(all(p.grad is None for p in model.parameters()), "Diagnostic populated gradients")
    return {
        "density_weight": density_weight, "common_A": float(a.detach()),
        "common_A_minus_Gaussian_constant": float(a.detach()) -
            (CONSTANT if any(c["active_labels"] for c in per_context) else 0),
        "label_weighted_D": float(d.detach()), "density_loss": float(metrics["density_loss"].detach()),
        "A_plus_D_loss_absolute_error": gap, "exam_BCE": full_bce,
        "context_only_BCE": prior_bce, "full_minus_context_only_BCE": full_bce - prior_bce,
        "regional_mean_intervention_BCE_minus_full": mean_bce - full_bce,
        "regional_mean_intervention_probability_mean_absolute_change":
            float((mean_p - full_p).abs().mean()),
        "whole_BF_removal_probability_mean_absolute_change": float((prior_p - full_p).abs().mean()),
        "component_gradients_unweighted": balance, "contexts": per_context,
        "raw_labels_descriptors_predictions_exported": False,
    }


def validate_pilot(accepted, scheduler, binding, original_binding):
    require(accepted["status"] == "actual_C100_full_stop_resume_engineering_complete"
            and accepted["GPU_job"] == "981620" and accepted["source_git"] == PILOT_SOURCE
            and accepted["binding_sha256"] == PILOT_BINDING and accepted["branch"] == "C100"
            and accepted["full_stop_resume_complete_state_and_prediction_bytes_equal"] is True,
            "Actual completed original C100 pilot required")
    require(scheduler["GPU_job"] == "981620" and scheduler["state"] == "COMPLETED"
            and scheduler["exit_code"] == "0:0" and scheduler["node"] == "evc27",
            "Actual terminal scheduler evidence required")
    require(binding["source_git"] == PILOT_SOURCE and binding["branch"] == "C100"
            and binding["original_numerical_source_git"] == original_binding["source_git"]
            and binding["objective_measurement_sha256"] == PRIOR_SHA
            and accepted["objective_measurement_sha256"] == PRIOR_SHA,
            "Original numerical/input lineage required")
    for name in ("targets16", "context_index"):
        for suffix in ("_private_path", "_sha256"):
            require(binding[name + suffix] == original_binding[name + suffix],
                    "Matched original input receipt differs")
    require(binding["original_module_sha256"] == {
        n: h for n, h in original_binding["code_sha256"].items() if n.startswith("rsna/")
    }, "Original numerical modules differ")
    require(accepted["terminal"]["density_weight"] == 100
            and accepted["terminal"]["steps"] == 24
            and [r["step"] for r in accepted["real_optimizer_history"]] == list(range(1, 25)),
            "Exact C10024 terminal and history required")


def matched_case(model, contexts, labels, mask, ordinal, loss, c100_state):
    """Restore every original tensor even if the new diagnostic fails."""
    original = {n: value.detach().clone() for n, value in model.state_dict().items()}
    try:
        base = diagnose_case(model, contexts, labels, mask, ordinal, loss, 0.1)
        model.load_state_dict(c100_state, strict=True)
        changed = diagnose_case(model, contexts, labels, mask, ordinal, loss, 100.0)
        require(all(torch.equal(v, c100_state[n]) for n, v in model.state_dict().items()),
                "C100 model state changed")
        return {"fitting_ordinal": ordinal, "C24": base, "C10024": changed}
    finally:
        model.load_state_dict(original, strict=True)
        require(all(torch.equal(v, original[n]) for n, v in model.state_dict().items()),
                "Original C24 restoration differs")


def run(payload):
    started = time.monotonic()
    require(os.uname().nodename == "evc27" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not os.environ.get("SLURM_JOB_GPUS")
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1" and sys.dont_write_bytecode,
            "Owning-node one-CPU zero-GPU read-only invocation required")
    loader = payload["original_loader_source"]
    require(sha(loader) == LOADER_SHA and sha(payload["prior_result_text"]) == PRIOR_SHA,
            "Exact original CPU981494 loader and receipt required")
    for old, new in (('os.environ["SLURM_CPUS_PER_TASK"] == "2"',
                      'os.environ["SLURM_CPUS_PER_TASK"] == "1"'),
                     ("torch.set_num_threads(2)", "torch.set_num_threads(1)")):
        require(loader.count(old) == 1, "Original single-CPU loader anchor differs")
        loader = loader.replace(old, new)
    ns = {"__name__": "original_readonly_C24_loader"}
    exec(compile(loader, "original_readonly_C24_loader", "exec"), ns)
    stat, old = ns["stat"], ns["ROOT"]
    prior = json.loads(payload["prior_result_text"])
    require(stat(old / "engineering/C/full/final.pt") == prior["checkpoint_stat_before_and_after"]
            and sha((old / "binding.private.json").read_bytes()) == prior["binding_sha256"]
            and sha((old / "engineering/C/full/terminal.json").read_bytes()) == prior["terminal_sha256"],
            "Original retained C24 state differs")
    reference = payload["C100_reference"]
    accepted_body = (PILOT / "GPU-accepted.json").read_bytes()
    require(sha(accepted_body) == reference["acceptance_sha256"], "C100 acceptance differs")
    accepted = json.loads(accepted_body)
    binding_body = (PILOT / "binding.private.json").read_bytes()
    require(sha(binding_body) == PILOT_BINDING, "Original admitted C100 binding differs")
    binding = json.loads(binding_body)
    original_binding = json.loads((old / "binding.private.json").read_bytes())
    require(binding["original_binding_sha256"] == prior["binding_sha256"],
            "C100 did not inherit the original C24 inputs")
    validate_pilot(accepted, reference["scheduler"], binding, original_binding)
    for name, digest in binding["code_sha256"].items():
        p = PILOT / "code" / name
        require(p.resolve().is_relative_to((PILOT / "code").resolve())
                and sha(p.read_bytes()) == digest, "C100 frozen code differs")
    for name, proof in binding["runtime_files"].items():
        p = ns["PORTABLE"] / name if name != "resolved_python" else Path(proof["path"])
        s = p.stat()
        require([s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] == proof["stat"],
                "Admitted original runtime differs")
        if "sha256" in proof:
            require(sha(p.read_bytes()) == proof["sha256"], "Runtime receipt differs")
        else:
            require(s.st_uid == proof["owner_uid"] and s.st_mode & 0o777 == proof["mode"]
                    and not s.st_mode & 0o022 and s.st_mode & 0o111, "Runtime permissions differ")
    require(str(torch.__version__) == binding["runtime_versions"]["torch"] == "2.14.0+cu130"
            and torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90",
            "Actual admitted PyTorch runtime required")
    require(Path(sys.executable).resolve()
            == Path(binding["runtime_files"]["resolved_python"]["path"]),
            "Use the exact admitted runtime executable")
    final = PILOT / "engineering/C/full/final.pt"
    terminal = PILOT / "engineering/C/full/terminal.json"
    require(stat(final) == reference["checkpoint_stat"]
            and sha(terminal.read_bytes()) == reference["terminal_sha256"]
            and json.loads(terminal.read_bytes()) == accepted["terminal"], "C100 final reference differs")
    saved = torch.load(final, map_location="cpu", weights_only=False)
    expected_recipe = dict(prior["recipe"], density_weight=100.0)
    require(saved["binding"] == binding and saved["recipe"] == expected_recipe
            and saved["step"] == 24 and saved["cohort_size"] == 12
            and saved["epoch_snapshots"] is False
            and saved["history"] == accepted["real_optimizer_history"], "C100 checkpoint identity differs")
    state = saved["model"]
    del saved
    gc.collect()
    ns["diagnose_case"] = lambda model, contexts, labels, mask, ordinal, loss: matched_case(
        model, contexts, labels, mask, ordinal, loss, state)
    result = ns["run"](payload)
    require(stat(final) == reference["checkpoint_stat"]
            and (PILOT / "binding.private.json").read_bytes() == binding_body
            and (PILOT / "GPU-accepted.json").read_bytes() == accepted_body,
            "C100 retained artifacts changed")
    result.update(
        status=STATUS, C100_reference=reference, C100_source_git=PILOT_SOURCE,
        C100_binding_sha256=PILOT_BINDING, requested_CPUs=1,
        original_loader_sha256=LOADER_SHA, one_CPU_loader_sha256=sha(loader),
        previous_actual_receipt_sha256=PRIOR_SHA,
        both_checkpoint_models_unchanged=True, optimizer_steps=0, remote_files_written=0,
        actual_density_coefficients=[0.1, 100.0],
        gradient_scope="nonencoder_parameters_before_clipping_or_Adam",
        Gaussian_constant_has_zero_gradient=True,
        weak_labels_are_not_ground_truth=True, clinical_or_held_out_claim=False,
        diagnostic_cohort_four_used=False, automatic_training_or_recipe_decision=False,
        total_elapsed_seconds_including_C100_admission_and_load=time.monotonic() - started,
    )
    require(result["total_elapsed_seconds_including_C100_admission_and_load"] < 1800,
            "Bounded diagnostic elapsed time exceeded")
    return result


def baseline_case(model, contexts, labels, mask, ordinal, loss):
    """Use the already loaded original C24 model; no load_state_dict or new model."""
    return {"fitting_ordinal": ordinal,
            "C24": diagnose_case(model, contexts, labels, mask, ordinal, loss, 0.1)}


def run_baseline(payload):
    """Separate original-checkpoint-only entry; never reads the C100 root."""
    started = time.monotonic()
    require(payload.get("execution_mode") == "C24_only"
            and not any(str(k).startswith("C100") for k in payload),
            "Explicit C24-only scope without C100 references required")
    require(os.uname().nodename == "evc27" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not os.environ.get("SLURM_JOB_GPUS")
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1" and sys.dont_write_bytecode,
            "Owning-node one-CPU zero-GPU read-only invocation required")
    loader = payload["original_loader_source"]
    require(sha(loader) == LOADER_SHA and sha(payload["prior_result_text"]) == PRIOR_SHA,
            "Exact original CPU981494 loader and receipt required")
    for old, new in (('os.environ["SLURM_CPUS_PER_TASK"] == "2"',
                      'os.environ["SLURM_CPUS_PER_TASK"] == "1"'),
                     ("torch.set_num_threads(2)", "torch.set_num_threads(1)")):
        require(loader.count(old) == 1, "Original single-CPU loader anchor differs")
        loader = loader.replace(old, new)
    ns = {"__name__": "original_readonly_C24_only_loader"}
    exec(compile(loader, "original_readonly_C24_only_loader", "exec"), ns)
    prior = json.loads(payload["prior_result_text"])
    old, stat = ns["ROOT"], ns["stat"]
    require(stat(old / "engineering/C/full/final.pt") == prior["checkpoint_stat_before_and_after"]
            and sha((old / "binding.private.json").read_bytes()) == prior["binding_sha256"]
            and sha((old / "engineering/C/full/terminal.json").read_bytes()) == prior["terminal_sha256"],
            "Original retained C24 state differs")
    require(Path(sys.executable).resolve() == (ns["PORTABLE"] / "runtime/bin/python").resolve()
            and str(torch.__version__) == "2.14.0+cu130"
            and torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90",
            "Use the original admitted predictive runtime")
    ns["diagnose_case"] = baseline_case
    result = ns["run"](payload)
    result.update(
        status=BASELINE_STATUS, comparison_scope="C24_only_no_C100_model_loaded",
        requested_CPUs=1, original_loader_sha256=LOADER_SHA,
        one_CPU_loader_sha256=sha(loader), previous_actual_receipt_sha256=PRIOR_SHA,
        actual_density_coefficients=[0.1], optimizer_steps=0, remote_files_written=0,
        gradient_scope="nonencoder_parameters_before_clipping_or_Adam",
        Gaussian_constant_has_zero_gradient=True, weak_labels_are_not_ground_truth=True,
        clinical_or_held_out_claim=False, diagnostic_cohort_four_used=False,
        automatic_training_or_recipe_decision=False,
        total_elapsed_seconds=time.monotonic() - started,
    )
    require(result["total_elapsed_seconds"] < 1800
            and all(set(row) == {"fitting_ordinal", "C24"} for row in result["cases"]),
            "Baseline result exceeds scope or elapsed bound")
    return result


def execute(payload):
    mode = payload.get("execution_mode")
    require(mode in (None, "C24_only"), "Unknown diagnostic scope")
    return run_baseline(payload) if mode == "C24_only" else run(payload)


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
    except BaseException as error:
        tb = error.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(json.dumps({"error_type": type(error).__name__, "site_line": tb.tb_lineno}),
              file=sys.stderr, flush=True)
        raise SystemExit(1) from None
