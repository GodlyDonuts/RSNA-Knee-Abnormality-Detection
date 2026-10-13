import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle import reviewed_infer as reviewed


def state():
    root = Path("initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final")
    binding = json.loads((root / "public-run.json").read_text())["binding"]
    result = json.loads((root / "public-result.json").read_text())
    checkpoint = dict(
        binding=binding,
        schema_version=2,
        optimizer_boundary=True,
        epoch=10,
        next_update=0,
        step=2810,
        scheduled_epoch=True,
    )
    admission = dict(
        schema_version=1,
        status="owner_authorized_fixed_final_kaggle_comparison",
        role="public_source_final10",
        scientific_source_git="a9902f3db0cf56995c074cddef1f0c2370e3ce47",
        checkpoint_sha256=result["checkpoint_sha256"],
        training_binding_sha256=reviewed.object_sha(binding),
        training_code_sha256=reviewed.object_sha(binding["code_sha256"]),
        seed=20260922,
        optimizer_steps=2810,
        training_studies=2247,
        outer_studies=1156,
        MRI_only_inference=True,
        original_checkpoint_and_numerical_source_unchanged=True,
    )
    return checkpoint, admission, result["checkpoint_sha256"]


def test_exact_public_final_admission_preserves_original_checkpoint_and_default_guard():
    checkpoint, admission, sha = state()
    before = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError, match="not admitted"):
        reviewed.infer.validate_inference_checkpoint(checkpoint)
    assert reviewed.validate_reviewed(checkpoint, admission, sha) is checkpoint["binding"]
    assert checkpoint == before


@pytest.mark.parametrize(
    "change", ["hash", "binding", "diagnostic", "external", "private", "epoch", "MRI"]
)
def test_public_final_admission_cannot_expand_scope_even_when_metadata_rebound(change):
    checkpoint, admission, sha = state()
    if change == "hash":
        sha = admission["checkpoint_sha256"] = "a" * 64
    elif change == "binding":
        checkpoint["binding"]["config"]["side"] = 224
    elif change == "diagnostic":
        checkpoint["binding"]["config"]["diagnostic_study_limit"] = 16
    elif change == "external":
        checkpoint["binding"]["external_initialization"] = {"source": "MRNet"}
    elif change == "private":
        checkpoint["binding"]["private_research_only"] = True
    elif change == "epoch":
        checkpoint["epoch"] = 9
    elif change == "MRI":
        admission["MRI_only_inference"] = False
    admission["training_binding_sha256"] = reviewed.object_sha(checkpoint["binding"])
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)
