"""Frozen C24 attribution on the same two fitting cases as CPU981494."""

import hashlib
import json
import math

import torch
from torch.nn import functional as F

LOADER_SHA = "36a87becf4a79aeda7f6c491c82ff58c69b779902348f231648b5f00bec3aac0"
PRIOR_RESULT_SHA = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
STATUS = "actual_frozen_C24_training_only_evidence_attribution"
BLOCKS = {"DCT32": (0, 32), "gradient16": (32, 48), "texture16": (48, 64)}


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def summary(value):
    value = value.detach().double()
    require(value.numel() > 0 and bool(torch.isfinite(value).all()), "Finite aggregate required")
    return dict(mean=float(value.mean()), RMS=float(value.square().mean().sqrt()),
                minimum=float(value.min()), maximum=float(value.max()),
                maximum_absolute=float(value.abs().max()))


def descriptor_statistics(targets, mu, delta, observed):
    """Per-coordinate summaries across regions, never raw descriptors/predictions."""
    t = targets[observed].double()
    require(t.ndim == 2 and t.shape[1] == 64 and len(t) > 0, "Observed 64D fitting regions")
    negative = t[None] - mu[:, observed].double()
    positive = negative - delta[:, observed].double()
    vectors = {
        "target_mean": t.mean(0),
        "target_variance_across_observed_regions": t.var(0, correction=0),
        "negative_residual_MSE_across_classes_and_regions": negative.square().mean((0, 1)),
        "positive_residual_MSE_across_classes_and_regions": positive.square().mean((0, 1)),
    }
    require(all(bool(torch.isfinite(v).all()) for v in vectors.values()), "Finite coordinate moments")
    return {
        "coordinates": {name: value.tolist() for name, value in vectors.items()},
        "blocks": {
            name: {key: float(value[start:end].mean()) for key, value in vectors.items()}
            for name, (start, end) in BLOCKS.items()
        },
        "negative_mean": summary(mu[:, observed]),
        "finding_shift": summary(delta[:, observed]),
        "residual_MSE_is_not_conditional_covariance_estimate": True,
        "regions_are_correlated_and_coordinates_have_different_units": True,
    }


def support_scores(local, supports, observed):
    """FP64 sparse aggregation; memory bounded by CSR edges and C×supports."""
    require(local.dtype == torch.float64 and local.ndim == 2, "FP64 class/region scores")
    owners = supports.validate(local.device)
    require(local.shape == (supports.log_prior.shape[0], supports.region_count)
            and observed.shape == (supports.region_count,) and observed.dtype == torch.bool,
            "Aligned observed regions and fixed supports")
    masked = local.masked_fill(~observed[None], 0)
    scores = local.new_zeros(local.shape[0], len(supports.indptr) - 1)
    touched = torch.zeros(scores.shape[1], dtype=torch.int64, device=local.device)
    for start in range(0, len(owners), 16384):
        region = supports.regions[start:start + 16384]
        owner = owners[start:start + 16384]
        scores.index_add_(1, owner, masked.index_select(1, region))
        touched.index_add_(0, owner, observed.index_select(0, region).long())
    return scores, touched


def mixture_at_unit_precision(scores, log_prior):
    """Exact derivative for logsumexp(logπ + βS) at β=1; no scale sweep."""
    require(scores.dtype == torch.float64 and scores.shape == log_prior.shape
            and bool(torch.isfinite(scores).all()), "Finite aligned support scores")
    prior = log_prior.double().log_softmax(-1)
    logbf = torch.logsumexp(prior + scores, -1)
    posterior = (prior + scores).softmax(-1)
    derivative = (posterior * scores).sum(-1)
    curvature = (posterior * (scores - derivative[:, None]).square()).sum(-1)
    return logbf, prior.exp(), posterior, derivative, curvature


def support_statistics(targets, mu, delta, observed, supports, original_evidence):
    t, m, d = targets.double(), mu.double(), delta.double()
    local = (d * (t[None] - m) - 0.5 * d.square()).sum(-1)
    scores, touched = support_scores(local, supports, observed)
    target_scores, _ = support_scores((d * t[None]).sum(-1), supports, observed)
    context_scores, _ = support_scores((-d * m - 0.5 * d.square()).sum(-1), supports, observed)
    decomposition_gap = float((scores - target_scores - context_scores).abs().max())
    require(torch.allclose(scores, target_scores + context_scores, atol=1e-10, rtol=1e-10),
            "Target/context support-score algebra differs")
    logbf, prior, posterior, derivative, curvature = mixture_at_unit_precision(
        scores, supports.log_prior,
    )
    context_only_bf = torch.logsumexp(supports.log_prior.double().log_softmax(-1)
                                     + context_scores, -1)
    gap = float((logbf - original_evidence.log_bayes_factor[0].double()).abs().max())
    require(gap <= 1e-4, "Independent FP64 support aggregation differs from original FP32")
    missing = touched == 0
    prior_missing = prior[:, missing].sum(-1)
    require(torch.allclose(prior_missing, original_evidence.no_observation_prior_mass[0].double(),
                           atol=2e-6, rtol=2e-6), "Unobserved support prior differs")
    bins = {
        "unobserved": missing, "one_observed_region": touched == 1,
        "two_to_four_observed_regions": (touched >= 2) & (touched <= 4),
        "five_to_sixteen_observed_regions": (touched >= 5) & (touched <= 16),
        "more_than_sixteen_observed_regions": touched > 16,
    }
    return {
        "supports": scores.shape[1], "complete_regions": supports.region_count,
        "observed_regions": int(observed.sum()), "support_log_score": summary(scores),
        "support_score_decomposition": {
            "target_dot_delta": summary(target_scores),
            "context_only_negative_delta_dot_mu_minus_half_delta_squared": summary(context_scores),
            "maximum_sum_identity_error": decomposition_gap,
            "logBF_without_target_dot_term": summary(context_only_bf),
            "logBF_change_when_target_dot_term_added": summary(logbf - context_only_bf),
            "support_terms_are_additive_but_mixture_logBF_terms_are_not": True,
        },
        "log_bayes_factor": summary(logbf),
        "independent_FP64_vs_original_FP32_maximum_logBF_gap": gap,
        "inverse_variance_beta_evaluation_point": 1.0,
        "d_logBF_d_beta_at_one": summary(derivative),
        "d2_logBF_d_beta2_at_one": summary(curvature),
        "observed_support_prior_mass": summary(prior[:, ~missing].sum(-1)),
        "unobserved_support_prior_mass": summary(prior_missing),
        "observed_support_posterior_mass": summary(posterior[:, ~missing].sum(-1)),
        "unobserved_support_posterior_mass": summary(posterior[:, missing].sum(-1)),
        "mass_by_observed_support_size": {
            name: {
                "supports": int(mask.sum()),
                "prior_mass": summary(prior[:, mask].sum(-1)),
                "posterior_mass": summary(posterior[:, mask].sum(-1)),
            }
            for name, mask in bins.items()
        },
        "covariance_or_temperature_sweep": False,
    }


def mean_intervention(targets, observed):
    """Remove within-context regional variation; retain its coordinate means."""
    require(bool(observed.any()), "Observed regions required for a mean intervention")
    changed = targets.clone()
    changed[observed] = targets[observed].mean(0)
    return changed


def readout(logits, labels, mask):
    z = torch.stack(logits).double()
    logp = torch.logsumexp(F.logsigmoid(z), 0) - math.log(2)
    log1p = torch.logsumexp(F.logsigmoid(-z), 0) - math.log(2)
    q = torch.where(mask[0], labels[0].double(), 0)
    loss = (-(q * logp + (1 - q) * log1p)).masked_fill(~mask[0], 0).sum()
    loss = loss / mask.sum().clamp_min(1)
    probabilities = logp.exp()
    return probabilities, {
        "posterior_summary": summary(probabilities),
        "training_only_masked_BCE_of_mean_posterior": float(loss),
    }


@torch.no_grad()
def diagnose_case(model, contexts, labels, mask, ordinal, two_mask_loss):
    # Import only after the original loader admits exact original rsna modules.
    from rsna.competitive.predictive.density import evidence

    require(ordinal in (0, 1) and len(contexts) == 2, "Only the original two fitting cases")
    require(torch.count_nonzero(model.fixed_log_variance) == 0, "Original unit covariance")
    result, original, mean_outputs, priors = [], [], [], []
    for i, context in enumerate(contexts):
        mu, delta, a = model.predict_hypotheses(
            *(context[n] for n in ("images", "token_metadata", "region_metadata")),
        )
        targets, observed, supports = (context[n] for n in ("targets", "observed", "supports"))
        e = evidence(targets[None], mu[None], delta[None], model.fixed_log_variance,
                     supports, observed[None])
        intervened = evidence(mean_intervention(targets, observed)[None],
                              mu[None], delta[None], model.fixed_log_variance,
                              supports, observed[None])
        count = int(observed.sum()) * model.descriptor_width
        q = torch.where(mask, labels, 0)
        original_nll = -(q * e.log_p1 + (1 - q) * e.log_p0) / count
        decomposed_nll = (-e.log_p0 - q * e.log_bayes_factor) / count
        require(torch.allclose(original_nll, decomposed_nll, atol=2e-6, rtol=2e-6),
                "Actual density normalization algebra differs")
        result.append({
            "context_ordinal": i, "density_normalization_N_times_64": count,
            "density_NLL_algebra_maximum_FP32_difference": float(
                (original_nll - decomposed_nll).abs().max()),
            "descriptor_statistics": descriptor_statistics(targets, mu, delta, observed),
            "support_statistics": support_statistics(targets, mu, delta, observed, supports, e),
            "context_prior_logits": summary(a),
            "mean_intervention_logBF_change": summary(
                intervened.log_bayes_factor - e.log_bayes_factor),
        })
        original.append(e)
        mean_outputs.append(a + intervened.log_bayes_factor[0])
        priors.append(a)
    _, metrics = two_mask_loss(
        original, torch.stack(priors)[None], labels, mask,
        [c["observed"][None] for c in contexts], model.descriptor_width, 0.1,
    )
    views = {
        "full": [a + e.log_bayes_factor[0] for a, e in zip(priors, original)],
        "prior_only": priors,
        "evidence_only": [e.log_bayes_factor[0] for e in original],
        "within_case_observed_coordinate_mean": mean_outputs,
    }
    posteriors, reports = {}, {}
    for name, logits in views.items():
        posteriors[name], reports[name] = readout(logits, labels, mask)
    require(torch.allclose(posteriors["full"], metrics["probabilities"][0].double(),
                           atol=2e-7, rtol=2e-7), "Original two-mask readout differs")
    for name in views:
        reports[name]["change_from_full"] = summary(posteriors[name] - posteriors["full"])
        reports[name]["mean_absolute_change_from_full"] = float(
            (posteriors[name] - posteriors["full"]).abs().mean())
        reports[name]["BCE_minus_full"] = (
            reports[name]["training_only_masked_BCE_of_mean_posterior"]
            - reports["full"]["training_only_masked_BCE_of_mean_posterior"])
    return {
        "fitting_ordinal": ordinal, "contexts": result, "readout_ablations": reports,
        "original_density_loss": float(metrics["density_loss"]),
        "original_exam_loss": float(metrics["exam_loss"]),
        "observed_label_count": int(mask.sum()),
        "intervention": "Each context replaces only its observed descriptors by their "
                        "within-study coordinate mean; original mu/delta/a and supports fixed. "
                        "This retains case-level means and removes regional variation.",
        "raw_labels_descriptors_or_prediction_rows_exported": False,
        "clinical_or_held_out_claim": False,
    }


def derive_loader(source):
    require(digest(source) == LOADER_SHA, "Exact previously executed CPU981494 loader required")
    for old, new in (
        ('os.environ["SLURM_CPUS_PER_TASK"] == "2"', 'os.environ["SLURM_CPUS_PER_TASK"] == "1"'),
        ("torch.set_num_threads(2)", "torch.set_num_threads(1)"),
    ):
        require(source.count(old) == 1, "Original CPU-only loader anchor differs")
        source = source.replace(old, new)
    return source


def run(payload):
    prior_text = payload["prior_result_text"]
    require(digest(prior_text) == PRIOR_RESULT_SHA, "Exact actual CPU981494 receipt required")
    prior = json.loads(prior_text)
    require(prior["CPU_job"] == "981494" and prior["checkpoint_step"] == 24
            and prior["diagnostic_sha256"] == LOADER_SHA, "Prior actual identity differs")
    loader = derive_loader(payload["original_loader_source"])
    namespace = {"__name__": "original_C24_readonly_loader"}
    exec(compile(loader, "original_C24_readonly_loader_one_CPU", "exec"), namespace)
    root = namespace["ROOT"]
    require(namespace["stat"](root / "engineering/C/full/final.pt")
            == prior["checkpoint_stat_before_and_after"]
            and digest((root / "binding.private.json").read_bytes()) == prior["binding_sha256"]
            and digest((root / "engineering/C/full/terminal.json").read_bytes())
            == prior["terminal_sha256"], "Original C24 checkpoint/binding/terminal changed")
    namespace["diagnose_case"] = diagnose_case
    result = namespace["run"](payload)
    require(result["checkpoint_stat_before_and_after"] == prior["checkpoint_stat_before_and_after"]
            and result["binding_sha256"] == prior["binding_sha256"], "Actual prior proof changed")
    result.update(
        status=STATUS, previous_actual_CPU_job="981494",
        previous_actual_receipt_sha256=PRIOR_RESULT_SHA,
        original_loader_sha256=LOADER_SHA, one_CPU_loader_sha256=digest(loader),
        gradient_scope="none; fixed-checkpoint forward attribution only",
        gradients_computed=False, optimizer_steps=0, requested_CPUs=1,
        covariance_or_temperature_sweep=False, model_or_inference_recipe_changed=False,
        conditional_covariance_estimated=False, clinical_or_held_out_claim=False,
        diagnostic_cohort_four_used=False,
        density_algebra="NLL=(-logp0-q*logBF)/(N_observed*64); "
                        "lambda multiplies this training loss only. Inference uses a+logBF.",
        beta_derivative_algebra="For fixed mu/delta, S=sum local log-ratios per support, "
                                "L(beta)=logsumexp(logpi+beta*S), L'(1)=E_posterior[S], "
                                "L''(1)=Var_posterior[S]. No beta other than1 is evaluated.",
    )
    return result


if __name__ == "__main__":
    import sys
    try:
        print(json.dumps(run(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
    except BaseException as error:
        tb = error.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(json.dumps({"error_type": type(error).__name__, "site_line": tb.tb_lineno}),
              file=sys.stderr, flush=True)
        raise SystemExit(1) from None
