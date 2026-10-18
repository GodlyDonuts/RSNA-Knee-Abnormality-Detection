"""An isolated sampler closure around the unchanged actual importance trainer."""

from __future__ import annotations

import hashlib
from types import FunctionType

import numpy as np
import torch
from sampling import draw

from rsna.competitive.licensed_importance import VERSION, ProposalCases, Recipe, validate_proposal
from rsna.competitive.licensed_importance import sample as original_sample
from rsna.competitive.licensed_importance_train import fit as original_fit


def fit(model, data, output, binding, recipe, *, components, allocation, **kwargs):
    if (
        not isinstance(data, ProposalCases)
        or not isinstance(recipe, Recipe)
        or recipe.role != "influence"
        or allocation not in ("iid", "finding_strata")
        or not isinstance(components, np.ndarray)
        or components.dtype != np.float64
        or components.shape != (len(data), 94, 12)
        or not isinstance(binding, dict)
        or "finding_allocation" in binding
    ):
        raise ValueError("Original influence engine, aligned components and explicit role required")
    values = components.copy()
    values.flags.writeable = False
    lookup = {uid: i for i, uid in enumerate(data.uids)}
    for i in range(len(data)):
        draw(values[i], data.proposal(i).numpy(), np.zeros((1, 16)))
    proof = {
        "version": "finding_stratified_12plus4_original_density_v1",
        "allocation": allocation,
        "components_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "mixture_density_unchanged": True,
        "images_per_study": 16,
    }

    def sample(uid, actual_recipe, epoch, proposal):
        if actual_recipe != recipe or uid not in lookup:
            raise ValueError("Exact bound recipe and original fitting identity required")
        if allocation == "iid":
            return original_sample(uid, actual_recipe, epoch, proposal)
        validate_proposal(proposal)
        if type(epoch) is not int or not 0 <= epoch < recipe.epochs:
            raise ValueError("Original zero-based epoch required")
        seed = hashlib.sha256(f"{VERSION}/{recipe.seed}/{epoch}/{uid}".encode()).digest()
        g = torch.Generator().manual_seed(int.from_bytes(seed[:8], "little") % 2**63)
        u = torch.rand(16, generator=g, dtype=torch.float64)
        mirror = bool(torch.randint(0, 2, (), generator=g).item())
        _, selected = draw(values[lookup[uid]], proposal.numpy(), u.numpy()[None])
        positions = selected[0].tolist()
        return positions, mirror, proposal[positions].log().float()

    # No mutation of original module globals, model graph, loss or RNG streams.
    engine = FunctionType(
        original_fit.__code__, {**original_fit.__globals__, "sample": sample}, "finding_allocation"
    )
    engine.__kwdefaults__ = original_fit.__kwdefaults__
    return engine(model, data, output, {**binding, "finding_allocation": proof}, recipe, **kwargs)
