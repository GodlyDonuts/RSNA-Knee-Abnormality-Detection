"""Private, final-source-checkpoint initialization of RSNA encoder adapters.

The source classifier, frozen base, optimizer and RNG never transfer. This is
an optional new experiment path; it cannot modify any frozen running source.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.mrnet import MRNET_TARGETS
from rsna.competitive.mrnet_training import MRNetRecipe

# Exact original source admission and numerical engine, preserved in the
# original-v2 dispatch archive. A different source recipe needs a new experiment.
SOURCE_BINDING = {
    "source_input_receipt_sha256": "146197f268471e167b2e16fb5e0724b24a58b714d3882567ec0bc09d9c95caf1",
    "acquisition_receipt_sha256": "f41c9f1d28f602ca0edcd1232a62da4584a065275f6b012693d983064f764259",
    "engine_sha256": "24714d28fa21abc143a7cd52cb59408db6284070d98654b6722a679e61189fba",
    "source_git": "f12a319aa9d36d2fecbac4f2df445c3c1d731146",
    "private_research_only": True,
    "external_weight_publication": False,
    "source_labels_never_mapped_to_RSNA_targets": True,
}


def _verified_bytes(path: Path, expected: str) -> bytes:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise ValueError("An explicit immutable checkpoint/receipt SHA256 is required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Source checkpoint or terminal receipt bytes differ")
    return raw


def initialize_mrnet_checkpoint(
    model: nn.Module,
    guard: FrozenBaseDelta,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    terminal: Path,
    terminal_sha256: str,
) -> dict[str, Any]:
    """Validate the complete final source envelope before changing any tensor.

    This accepts the full 1,129-exam source fit, never an engineering trajectory
    or an epoch chosen on source validation. It does not establish downstream
    benefit, cross-source independence or permission to publish derived weights.
    """
    saved = torch.load(
        io.BytesIO(_verified_bytes(checkpoint, checkpoint_sha256)),
        map_location="cpu",
        weights_only=True,
    )
    completed = json.loads(_verified_bytes(terminal, terminal_sha256))
    binding = saved.get("binding", {})
    recipe = asdict(MRNetRecipe())
    cases = binding.get("training_cases")
    if (
        saved.get("schema_version") != 2
        or saved.get("optimizer_boundary") is not True
        or any(binding.get(key) != value for key, value in SOURCE_BINDING.items())
        or binding.get("source_target_order") != list(MRNET_TARGETS)
        or binding.get("recipe") != recipe
        or not isinstance(cases, list)
        or any(not isinstance(case, str) for case in cases)
        or len(cases) != 1129
        or len(set(cases)) != 1129
        or binding.get("steps_per_epoch") != 142
        or type(saved.get("step")) is not int
        or saved["step"] != 1420
        or completed.get("status") != "private_full_Stanford_fitting_complete"
        or completed.get("source_training_exams") != 1129
        or completed.get("source_validation_exams") != 119
        or completed.get("step") != 1420
        or completed.get("checkpoint_sha256") != checkpoint_sha256
        or completed.get("no_competition_submission_or_weight_publication") is not True
        or completed.get("RSNA_or_Kaggle_accuracy_measured") is not False
        or not isinstance(completed.get("binding"), dict)
        or any(binding.get(key) != value for key, value in completed["binding"].items())
        or any(completed["binding"].get(key) != value for key, value in SOURCE_BINDING.items())
    ):
        raise ValueError("Exact completed full-source private Stanford fitting required")
    storage = saved.get("model_storage", {})
    expected_storage = guard.metadata()
    if (
        storage.get("format") != "frozen_base_delta_v1"
        or storage.get("encoder_provenance") != expected_storage["encoder_provenance"]
        or storage.get("frozen_tensors") != expected_storage["frozen_tensors"]
        or expected_storage["encoder_provenance"].get("architecture") != "dinov3_l16"
    ):
        raise ValueError("Identical complete frozen encoder bytes and adapter provenance required")
    learned, schema = saved.get("model"), storage.get("learned_tensors")
    if not isinstance(learned, dict) or not isinstance(schema, dict) or set(learned) != set(schema):
        raise ValueError("Complete original source learned-tensor schema required")
    for name, value in learned.items():
        if (
            not isinstance(value, torch.Tensor)
            or schema[name] != {"shape": list(value.shape), "dtype": str(value.dtype)}
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Invalid source learned tensor shape, dtype or numerical values")
    source_names = {name for name in learned if name.startswith("encoder.")}
    destination_names = {name for name in guard.trainable if name.startswith("encoder.")}
    if (
        not source_names
        or source_names != destination_names
        or any(not name.endswith((".lora_A", ".lora_B")) for name in source_names)
        or any(schema[name] != expected_storage["learned_tensors"][name] for name in source_names)
    ):
        raise ValueError("Exact complete encoder adapter graph, rank and dtype required")
    state = guard.checkpoint_model(model)["model"]  # Recheck actual frozen bytes before copying.
    with torch.no_grad():
        for name in sorted(source_names):
            state[name].copy_(learned[name])
    return {
        "method": "final_private_MRNet_encoder_adapters_only",
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_terminal_sha256": terminal_sha256,
        "source_binding": completed["binding"],
        "source_epochs": recipe["epochs"],
        "source_training_exams": 1129,
        "encoder_adapter_tensors": len(source_names),
        "source_heads_optimizer_counters_or_rng_transferred": False,
        "private_research_only": True,
        "external_weight_publication": False,
    }


def initialize_private_mrnet(
    args: Any, model: nn.Module, guard: FrozenBaseDelta | None
) -> dict | None:
    """CLI admission leaves an ordinary RSNA run completely unchanged."""
    fields = (
        "mrnet_checkpoint",
        "mrnet_checkpoint_sha256",
        "mrnet_terminal",
        "mrnet_terminal_sha256",
    )
    values = [getattr(args, name, None) for name in fields]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values) or guard is None:
        raise ValueError(
            "All source checkpoint/terminal paths, hashes and frozen-base storage required"
        )
    output = Path(args.output).resolve()
    if not any(
        output.is_relative_to(root) for root in (Path("/tmp").resolve(), Path("/private/tmp"))
    ):
        raise ValueError("Stanford-derived research runs require private node-local /tmp output")
    private_ancestors = [
        path
        for path in (output, *output.parents)
        if path.exists()
        and path.is_dir()
        and path.stat().st_uid == os.getuid()
        and stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    ]
    if not private_ancestors:
        raise ValueError("Source-derived outputs need an owner-only private scratch directory")
    source_path, source_sha, terminal_path, terminal_sha = values
    if (
        not isinstance(source_path, Path)
        or not isinstance(source_sha, str)
        or not isinstance(terminal_path, Path)
        or not isinstance(terminal_sha, str)
    ):
        raise ValueError("Explicit source Paths and SHA256 strings required")
    return initialize_mrnet_checkpoint(
        model,
        guard,
        checkpoint=source_path,
        checkpoint_sha256=source_sha,
        terminal=terminal_path,
        terminal_sha256=terminal_sha,
    )
