"""Candidate serving arithmetic: same varying-path score, fewer Python launches.

No active training source is changed. Actual CUDA/T4 equivalence and speed are
not established by the synthetic CPU checks for this implementation.
"""

from __future__ import annotations

import math

import torch
from path_reader import PathReader
from variable_paths import REFERENCE_GAP_MM, inputs


SCORING_VERSION = "varying_path_serving_batched_starts_v1"


@torch.no_grad()
def marginal_table(transitions, starts):
    """Advance all existing starts in one matrix multiply per source step."""
    length, tokens = starts.shape
    if (
        transitions.requires_grad
        or starts.requires_grad
        or transitions.shape != (length - 1, tokens, tokens)
    ):
        raise ValueError("Complete fixed metadata transitions and starts required")
    front = starts[:1]
    columns = [torch.nn.functional.pad(front, (0, 0, 0, length - 1))]
    for t in range(1, length):
        front = torch.cat((front @ transitions[t - 1], starts[t : t + 1]), dim=0)
        columns.append(torch.nn.functional.pad(front, (0, 0, 0, length - t - 1)))
    return torch.stack(columns, dim=1)


def backward_all_starts(emission, log_continue, log_stop):
    length = emission.shape[0]
    value = emission[:, -1]
    values = [value[-1]]
    for t in range(length - 2, -1, -1):
        # Slice only valid starts. Invalid triangular slots never participate
        # in arithmetic or produce all-minus-infinity gradient singularities.
        value = emission[: t + 1, t] + torch.logaddexp(
            log_stop[t], log_continue[t] + value[: t + 1]
        )
        values.append(value[t])
    return torch.logsumexp(torch.stack(values), dim=0) - math.log(length)


def independent_pair(unary, transitions, starts, log_continue, log_stop, *, chunk_bytes):
    length, tokens, targets = unary.shape
    measures = marginal_table(transitions, starts)
    first, times = torch.triu_indices(length, length, device=unary.device)
    pairs = max(1, chunk_bytes // (tokens * targets * unary.element_size()))
    actual, zero = [], []
    tiny = torch.finfo(unary.dtype).tiny
    for offset in range(0, len(first), pairs):
        s, t = first[offset : offset + pairs], times[offset : offset + pairs]
        measure = measures[s, t]
        log_m = measure.clamp_min(tiny).log().masked_fill(measure == 0, -torch.inf)
        actual.append(torch.logsumexp(log_m[:, :, None] + unary[t], dim=1))
        zero.append(torch.logsumexp(log_m, dim=1))
    emission = unary.new_zeros(length, length, targets)
    emission = emission.index_put((first, times), torch.cat(actual))
    baseline = unary.new_zeros(length, length, targets)
    baseline = baseline.index_put((first, times), torch.cat(zero)[:, None].expand(-1, targets))
    return backward_all_starts(emission, log_continue, log_stop), backward_all_starts(
        baseline, log_continue, log_stop
    )


def physical_partition(unary, log_transitions, log_starts, log_continue, log_stop):
    # Same stable log-semiring recurrence as the existing scientific source.
    # Do not replace it with exp/matmul: that can underflow on sparse support.
    value = unary[-1]
    values = [torch.logsumexp(log_starts[-1, :, None] + value, dim=0)]
    for t in range(len(unary) - 2, -1, -1):
        future = torch.logsumexp(log_transitions[t, :, :, None] + value[None, :, :], dim=1)
        value = unary[t] + torch.logaddexp(log_stop[t], log_continue[t] + future)
        values.append(torch.logsumexp(log_starts[t, :, None] + value, dim=0))
    return torch.logsumexp(torch.stack(values), dim=0) - math.log(len(unary))


def witness_score(
    unary,
    joints,
    starts,
    gaps,
    kappa,
    *,
    arm,
    reference_gap_mm=REFERENCE_GAP_MM,
    chunk_bytes=16 * 2**20,
    max_work_bytes=256 * 2**20,
):
    """Separate candidate; preserve focal paths and start-conditioned marginal null."""
    if arm not in {"physical", "independent"}:
        raise ValueError("Explicit unchanged path arm required")
    if (
        type(chunk_bytes) is not int
        or type(max_work_bytes) is not int
        or min(chunk_bytes, max_work_bytes) < 1
    ):
        raise ValueError("Positive explicit work budgets required")
    transition, log_transition, log_start, log_continue, log_stop = inputs(
        unary, joints, starts, gaps, kappa, reference_gap_mm
    )
    length, tokens, targets = unary.shape
    if arm == "independent":
        # Conservative tensor work estimate, not a measured CUDA allocator peak.
        work = (
            3 * length * length * tokens + 4 * length * length * targets
        ) * unary.element_size() + 3 * chunk_bytes
        if work > max_work_bytes or tokens * targets * unary.element_size() > chunk_bytes:
            raise MemoryError("Candidate vectorized marginal work exceeds the declared budget")
        actual, baseline = independent_pair(
            unary, transition, starts, log_continue, log_stop, chunk_bytes=chunk_bytes
        )
    else:
        actual = physical_partition(unary, log_transition, log_start, log_continue, log_stop)
        baseline = physical_partition(
            torch.zeros_like(unary), log_transition, log_start, log_continue, log_stop
        )
    # Keep numerical zero subtraction; do not silently replace baseline by its
    # analytic zero and change rounding or continuation-parameter derivatives.
    return actual - baseline + kappa * 0


# Explicit serving-only subclass. Preserve the learned parameter graph and the
# five evidence/four geometry summaries; replace only the scoring arithmetic.


class FastPathReader(PathReader):
    def statistics(self, maps, plan, *, arm):
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
            raise ValueError("Complete original full maps and all-window source paths required")
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
