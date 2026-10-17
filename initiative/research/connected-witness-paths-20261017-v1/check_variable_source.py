"""Synthetic directed/changing source kernels; no clinical inputs."""

from dataclasses import replace

import torch
from check_source_chains import fixture
from variable_paths import log_partition
from variable_source_chains import qualify_variable_chains

torch.set_num_threads(1)
for shear in (0.0, 1.0):
    graph, windows, plan = fixture(shear=shear)
    paths, report = qualify_variable_chains([graph], windows, plan)
    assert report["qualified_windows"] == 7 and report["unsupported_windows"] == 0
    assert report["qualified_repeat_edges"] == 4
    assert [p.indices for p in paths] == [(0, 3, 6), (1, 4), (2, 5)]
    if shear:
        assert report["nonreversible_edge_matrices"] == 4
        assert report["old_fixed_joint_unsupported_windows"] == 7
    for path in paths:
        zero = torch.zeros(len(path.indices), len(path.tokens), 1)
        for arm in ("physical", "independent"):
            score = log_partition(
                zero, path.joints, path.starts, path.gaps_mm, torch.zeros(1), arm=arm
            )
            assert score.abs().max() < 1e-6

# A genuinely changing matrix remains present in the actual ordered sequence.
groups = list(plan.groups)
repeat = next(j for j, g in enumerate(groups) if g.kind == "repeat")
original = groups[repeat]
second_kernel = original.aligned.clone()
second_kernel[0, 0] += 0.001
second_kernel /= second_kernel.sum()
groups[repeat] = replace(original, first=original.first[:1], second=original.second[:1])
groups.append(
    replace(original, first=original.first[1:], second=original.second[1:], aligned=second_kernel)
)
changed = replace(plan, groups=tuple(groups))
paths, report = qualify_variable_chains([graph], windows, changed)
assert report["qualified_windows"] == 7 and report["varying_joint_chains"] > 0
assert any(len(p.indices) > 2 and not torch.equal(p.joints[0], p.joints[1]) for p in paths)

# Zero outgoing states remain unsupported; do not insert self-links.
bad_kernel = original.aligned.clone()
bad_kernel[0] = 0
bad_groups = tuple(replace(g, aligned=bad_kernel) if g.kind == "repeat" else g for g in plan.groups)
paths, report = qualify_variable_chains([graph], windows, replace(plan, groups=bad_groups))
assert not paths and report["unsupported_windows"] == 7
assert (
    report["unsupported_chain_reasons"]["zero_outgoing_observation_state_needs_explicit_policy"]
    == 3
)
print(
    "All-window variable/directed source retention, ordered changes, physical/control normalization and zero-outgoing refusal passed"
)
