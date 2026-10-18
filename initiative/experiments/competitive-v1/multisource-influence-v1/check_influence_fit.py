"""Exercise the actual unchanged optimizer with synthetic MRI-shaped inputs."""

import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from influence_fit import fit_for

from rsna.competitive.licensed_importance import ProposalCases, Recipe
from rsna.competitive.licensed_importance_train import fit as old_fit

sys.path.insert(0, str(Path(__file__).parent.parent / "multisource-supervision-v1"))
import sampled_overlay as h  # noqa: E402

h.fit = fit_for(h)

spec = importlib.util.spec_from_file_location(
    "existing_IID16_fixture", "tests/test_licensed_importance.py"
)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
torch.set_num_threads(1)
base = f.OriginalCases(9)
base.mask = torch.ones((9, 12), dtype=torch.bool)
eligible = torch.zeros_like(base.mask)
eligible[:, 11] = True
labels = base.labels.clone()
labels[:, 11] = (labels[:, 11] + 0.9) / 2


def wrap(role):
    value = base.labels if role == "unchanged_reference" else labels
    policy = {
        "version": h.VERSION,
        "role": role,
        "diagnostic_only": True,
        "parent_checkpoint_sha256": h.PARENT_SHA,
        "original_target_file_sha256": h.ORIGINAL_FILE,
        "population": 9,
        "population_sha256": hashlib.sha256(
            json.dumps(base.uids, separators=(",", ":")).encode()
        ).hexdigest(),
        "original_labels_sha256": h.sha(base.labels),
        "labels_sha256": h.sha(value),
        "mask_sha256": h.sha(base.mask),
        "teacher_sha256": h.sha(base.teacher),
        "eligible_sha256": h.sha(eligible),
        "target_file_sha256": h.ORIGINAL_FILE if role == "unchanged_reference" else "a" * 64,
        "annotation_file_sha256": "b" * 64,
    }
    return h.ConsensusLabels(base, value, eligible, policy, diagnostic_only=True)


control, candidate = [wrap(role) for role in ("unchanged_reference", "safe_family_mean")]
proposal = f.proposal().repeat(9, 1)
old_data = ProposalCases(base, proposal)
c, t = [ProposalCases(x, proposal) for x in (control, candidate)]
recipe = Recipe(epochs=2, role="influence")
binding = {"parent_checkpoint_sha256": h.PARENT_SHA, "diagnostic_only": True}


def train(data, path, *, engine=h.fit, resume=False, stop_after=None):
    torch.manual_seed(322)
    model = f.Model()
    engine(
        model,
        data,
        path,
        binding,
        recipe,
        device=torch.device("cpu"),
        resume=resume,
        stop_after=stop_after,
    )
    return torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)


def prediction(value):
    model = f.Model()
    model.load_state_dict(value["model"])
    model.eval()
    with torch.inference_mode():
        return (
            torch.stack(
                [
                    model(base.case(i, list(range(94)), mirror)["images"].unsqueeze(0)).sigmoid()
                    for i in range(9)
                    for mirror in (False, True)
                ]
            )
            .numpy()
            .tobytes()
        )


refusals = []


def refuse(name, call):
    try:
        call()
    except ValueError:
        refusals.append(name)
    else:
        raise AssertionError(name)


selected = [2, 5, 2, 5] * 4
row = t.case(0, selected, True)
assert torch.equal(row["images"], base.images[0, selected].flip(-1))
assert torch.equal(row["labels"], labels[0])
assert torch.equal(row["mask"], base.mask[0])
assert torch.equal(row["parent_logits"], base.teacher[0, 1])
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    old, same = train(old_data, root / "old", engine=old_fit), train(c, root / "control")
    # Experiment provenance intentionally differs. Every scientific/numerical
    # state field must equal the unchanged original control, not just predictions.
    f.equal(
        {k: v for k, v in old.items() if k != "binding"},
        {k: v for k, v in same.items() if k != "binding"},
    )
    assert prediction(old) == prediction(same)
    outcomes = {}
    for name, data in (("control", c), ("candidate", t)):
        full = train(data, root / (name + "-full"))
        stopped = train(data, root / (name + "-resume"), stop_after=1)
        assert stopped["completed"] == 1
        resumed = train(data, root / (name + "-resume"), resume=True)
        f.equal(full, resumed)
        assert prediction(full) == prediction(resumed)
        for epoch in (1, 2):
            f.equal(
                *[
                    torch.load(
                        root / (name + suffix) / "epoch-checkpoints" / f"epoch-{epoch:03d}.pt",
                        weights_only=True,
                    )
                    for suffix in ("-full", "-resume")
                ]
            )
        outcomes[name] = full
    assert any(
        not torch.equal(outcomes["control"]["model"][k], value)
        for k, value in outcomes["candidate"]["model"].items()
    )
    refuse("cross-source resume", lambda: train(c, root / "candidate-resume", resume=True))
    refuse(
        "changed anchor",
        lambda: h.fit(
            f.Model(),
            t,
            root / "invalid",
            binding,
            replace(recipe, source_weight=0.5),
            device=torch.device("cpu"),
        ),
    )

bad = labels.clone()
bad[0, 0] = 1 - bad[0, 0]
refuse(
    "change outside eligible",
    lambda: h.ConsensusLabels(base, bad, eligible, candidate.binding, diagnostic_only=True),
)
refuse(
    "unbound policy mutation",
    lambda: h.ConsensusLabels(
        base,
        labels,
        eligible,
        {**candidate.binding, "labels_sha256": "0" * 64},
        diagnostic_only=True,
    ),
)
refuse(
    "synthetic production scope",
    lambda: h.ConsensusLabels(
        base,
        labels,
        eligible,
        {**candidate.binding, "diagnostic_only": False},
        diagnostic_only=False,
    ),
)
saved = base.teacher.clone()
base.teacher[0, 0, 0] += 1
refuse("teacher changed after admission", lambda: candidate.case(0, [2, 5], False))
base.teacher.copy_(saved)
result = {
    "status": "actual_original_influence_optimizer_synthetic_family_label_integration_accepted",
    "class_source_sha256": hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(
        Path(__file__).with_name("influence_fit.py").read_bytes()
    ).hexdigest(),
    "checks": [
        "original control complete numerical state and full94 prediction equality, excluding explicit experiment binding",
        "IID duplicates order mirror mask and view-specific parent unchanged",
        "both roles full/stop1/resume complete state and epoch identity",
        "both roles prediction-byte recovery",
        "nonzero treatment effect on learned parameters",
        "final singleton effective batch",
    ],
    "refusals": refusals,
    "actual_MRI_GPU_training_or_accuracy": False,
}
Path(__file__).with_name("sampled-overlay-checked.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
