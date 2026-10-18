"""Select the completed stronger sampler; keep the actual accepted label class."""

from rsna.competitive.licensed_importance import ProposalCases, Recipe
from rsna.competitive.licensed_importance_train import fit as original_fit


def fit_for(source):
    """Capture the exact loaded label-class identity, avoiding duplicate imports."""

    def fit(model, data, output, binding, recipe, **kwargs):
        if (
            not isinstance(recipe, Recipe)
            or recipe.role != "influence"
            or recipe.source_weight != 0.25
            or not isinstance(data, ProposalCases)
            or not isinstance(data.original, source.ConsensusLabels)
            or not isinstance(binding, dict)
            or "multisource_supervision" in binding
            or binding.get("diagnostic_only") is not data.original.binding["diagnostic_only"]
            or not binding["diagnostic_only"]
            and recipe != Recipe(role="influence")
        ):
            raise ValueError("Exact original influence recipe and accepted source class required")
        return original_fit(
            model,
            data,
            output,
            {**binding, "multisource_supervision": data.original.binding},
            recipe,
            **kwargs,
        )

    return fit
