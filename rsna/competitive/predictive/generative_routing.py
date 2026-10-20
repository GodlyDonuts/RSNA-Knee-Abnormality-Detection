"""Explicit experimental gradient routing; original training/inference are unchanged.

Only the exam loss treats the Bayes factor as a constant. The original proper
density loss retains both hypothesis graphs, including their shared encoder.
This module does not select a coefficient, alter a recipe, or launch training.
"""

from dataclasses import replace

from rsna.competitive.predictive.density import Evidence, two_mask_loss


def generative_two_mask_loss(
    evidences,
    context_logits,
    targets,
    label_mask,
    observations,
    descriptor_width=64,
    *,
    density_weight,
):
    """Keep exact loss/probability values, routing exam gradients only through priors.

    A new Evidence view detaches only ``log_bayes_factor``. Original ``log_p0``
    and ``log_p1`` stay attached: in particular log_p1's Bayes-factor gradient
    must not be detached or reconstructed from the detached view. Original
    evidence objects remain available for unchanged inference and diagnostics.
    The caller must explicitly choose this loss and its density coefficient.
    """
    if len(evidences) != 2 or any(not isinstance(e, Evidence) for e in evidences):
        raise ValueError("Two original Evidence objects required for explicit gradient routing")
    routed = [replace(e, log_bayes_factor=e.log_bayes_factor.detach()) for e in evidences]
    return two_mask_loss(
        routed,
        context_logits,
        targets,
        label_mask,
        observations,
        descriptor_width,
        density_weight,
    )
