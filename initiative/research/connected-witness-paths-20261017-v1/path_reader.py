"""Learn full-map evidence and variable source paths; no encoder is fitted here.

The canonical original training loop is reused explicitly with a new reader and
fresh optimizer. No numerical module or healthy original run is modified.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from variable_paths import REFERENCE_GAP_MM, witness_score
from variable_source_chains import qualify_variable_chains

from rsna.competitive.spatial_train import SpatialTrainer

VERSION = "N3_varying_source_witness_reader_v1"


@dataclass(frozen=True)
class PathPlan:
    source: object
    chains: tuple
    accounting: dict

    @property
    def groups(self):
        # The original trainer records relation counts; this is not the number
        # of independently confirming observations or a diagnostic measure.
        return self.source.groups


def prepare_plan(graphs, windows, source_plan):
    chains, report = qualify_variable_chains(graphs, windows, source_plan)
    if not chains or report["unsupported_windows"] or report["qualified_windows"] != len(windows):
        raise ValueError("The learned path reader requires every original source window")
    return PathPlan(source_plan, chains, report)


class PathCachedData:
    """Wrap the unchanged source-bound original cached provider, without new MRI IO."""

    def __init__(self, original, mechanism_binding: dict):
        if not mechanism_binding or not original.binding:
            raise ValueError("Original data and explicit new mathematical source bindings required")
        self.original = original
        self.binding = {
            "original": original.binding,
            "reader": VERSION,
            "mechanism": mechanism_binding,
        }
        self.plans = OrderedDict()

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        sample = self.original[index]
        row = self.original.records[index]
        key = (sample["study_uid"], sample["plan"].metadata_sha256)
        if key not in self.plans:
            windows = np.asarray(
                [(g["slot"], j) for g in row["source_graphs"] for j in range(len(g["supports"]))],
                dtype=np.int64,
            )
            self.plans[key] = prepare_plan(row["source_graphs"], windows, sample["plan"])
            if len(self.plans) > 8:
                self.plans.popitem(last=False)
        self.plans.move_to_end(key)
        return {**sample, "plan": self.plans[key]}


class PathReader(nn.Module):
    """Same images/projection/image summaries in both arms; only path coupling differs."""

    def __init__(self, channels=1024, width=64, *, seed=20261017):
        super().__init__()
        if type(channels) is not int or type(width) is not int or min(channels, width) < 1:
            raise ValueError("Positive full-map readout dimensions required")
        self.channels, self.width = channels, width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Sequential(
                nn.LayerNorm(channels), nn.Linear(channels, width), nn.GELU()
            )
            self.evidence = nn.Linear(width, 12)
            self.continuation = nn.Parameter(torch.zeros(12))
            self.output = nn.Parameter(torch.zeros(12, 9))
            self.bias = nn.Parameter(torch.zeros(12))

    def statistics(self, maps: Tensor, plan: PathPlan, *, arm: str):
        source = plan.source
        if (
            arm not in {"aligned", "null"}
            or maps.shape != (source.windows, source.channels, source.height, source.width)
            or source.channels != self.channels
            or maps.dtype != torch.float32
            or not torch.isfinite(maps).all()
            or plan.accounting["unsupported_windows"]
            or sum(len(p.indices) for p in plan.chains) != source.windows
        ):
            raise ValueError(
                "Complete finite original full maps and all-window source paths required"
            )
        evidence = self.evidence(self.project(maps.flatten(2).transpose(1, 2)))
        observed = torch.cat(
            [
                evidence[j, torch.as_tensor(tokens, device=maps.device)]
                for j, tokens in enumerate(source.valid_tokens)
            ]
        )
        summaries = [
            observed.mean(0),
            observed.std(0, unbiased=False),
            torch.logsumexp(observed, dim=0) - math.log(len(observed)),
            observed.amax(0),
        ]
        paths = []
        for path in plan.chains:
            indexes = torch.as_tensor(path.indices, device=maps.device)
            tokens = torch.as_tensor(path.tokens, device=maps.device)
            unary = evidence[indexes][:, tokens]
            score = witness_score(
                unary,
                path.joints.to(maps.device),
                path.starts.to(maps.device),
                path.gaps_mm.to(maps.device),
                self.continuation,
                arm="physical" if arm == "aligned" else "independent",
            )
            paths.append(score + math.log(len(path.indices)))
        summaries.append(torch.logsumexp(torch.stack(paths), dim=0) - math.log(source.windows))
        edges = sum(len(p.gaps_mm) for p in plan.chains)
        total_gap = sum(float(p.gaps_mm.sum()) for p in plan.chains)
        nuisance = maps.new_tensor(
            [
                math.log1p(source.windows),
                math.log1p(len(plan.chains)),
                plan.accounting["qualified_multistep_chains"] / len(plan.chains),
                total_gap / (max(1, edges) * REFERENCE_GAP_MM),
            ]
        )
        return torch.cat((torch.stack(summaries, dim=1), nuisance[None].expand(12, -1)), dim=1)

    def forward(self, maps, base_logits, plan, *, arm):
        if (
            base_logits.shape != (12,)
            or base_logits.dtype != torch.float32
            or not torch.isfinite(base_logits).all()
        ):
            raise ValueError("Complete unchanged twelve-target parent logits required")
        features = self.statistics(maps, plan, arm=arm)
        correction = (features * self.output).sum(1) + self.bias
        return base_logits.detach() + correction, correction


class PathTrainer(SpatialTrainer):
    """Explicit new reader/fresh AdamW; unchanged original schedule/state/RNG loop."""

    def __init__(self, data, recipe, *, arm, device, binding):
        if not binding.get("path_mechanism_source"):
            raise ValueError("Exact new source/fixture acceptance must be bound")
        super().__init__(data, recipe, arm=arm, device=device, binding=binding)
        self.model = PathReader(recipe.channels, recipe.width, seed=recipe.seed).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
            foreach=False,
        )
        self.binding["original_training_engine"] = self.binding["engine"]
        self.binding["engine"] = VERSION
        self.binding["reader_policy"] = {
            "encoder_fitted": False,
            "zero_start_unbounded_residual": True,
            "all_original_source_windows_required": True,
            "prior_starts_uniform_over_observed_tokens": True,
            "null": "same_single_position_prior_marginal_given_each_possible_start",
            "overlapping_chain_phases": "mixture_of_possible_starts_not_independent_votes",
        }
