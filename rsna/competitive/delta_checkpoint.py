"""Exact adapter recovery against a checksummed, immutable pretrained base.

This stores the complete optimizer/counter/RNG envelope. Only frozen model
tensors are omitted; they must be reconstructed and verified before loading.
It is distinct from the earlier inference-only diagnostic delta.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
from torch import nn


def _tensor_record(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


class FrozenBaseDelta:
    """Bind the precise frozen/trainable partition of a verified base model."""

    def __init__(self, model: nn.Module, provenance: dict) -> None:
        self.provenance = dict(provenance)
        if provenance.get("pretrained") is not True or not all(
            isinstance(provenance.get(key), str)
            and len(provenance[key]) == 64
            and all(char in "0123456789abcdef" for char in provenance[key])
            for key in ("weights_sha256", "config_sha256")
        ):
            raise ValueError("Delta recovery needs a verified pretrained weight/config source")
        self.trainable = tuple(
            sorted(name for name, p in model.named_parameters() if p.requires_grad)
        )
        state = model.state_dict()
        self.frozen = tuple(sorted(set(state) - set(self.trainable)))
        if (
            not self.trainable
            or not self.frozen
            or any(not name.startswith("encoder.") for name in self.frozen)
        ):
            raise ValueError("Delta recovery supports only an immutable encoder base")
        self.initial_frozen = {name: _tensor_record(state[name]) for name in self.frozen}
        self.learned_schema = {
            name: {"shape": list(state[name].shape), "dtype": str(state[name].dtype)}
            for name in self.trainable
        }

    def _check_partition(self, model: nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        trainable = tuple(sorted(name for name, p in model.named_parameters() if p.requires_grad))
        if trainable != self.trainable or set(state) != set(self.frozen) | set(self.trainable):
            raise ValueError("Frozen/trainable model partition changed")
        if {name: _tensor_record(state[name]) for name in self.frozen} != self.initial_frozen:
            raise ValueError("A frozen pretrained tensor changed; refusing delta recovery")
        return dict(state)

    def metadata(self) -> dict:
        return {
            "format": "frozen_base_delta_v1",
            "encoder_provenance": self.provenance,
            "frozen_tensors": self.initial_frozen,
            "learned_tensors": self.learned_schema,
        }

    def checkpoint_model(self, model: nn.Module) -> dict:
        state = self._check_partition(model)
        return {
            "schema_version": 2,
            "model_storage": self.metadata(),
            "model": {name: state[name] for name in self.trainable},
        }

    def restore(self, model: nn.Module, checkpoint: Mapping) -> None:
        if (
            checkpoint.get("schema_version") != 2
            or checkpoint.get("model_storage") != self.metadata()
        ):
            raise ValueError("Delta frozen base, tensor schema or encoder provenance differs")
        state = self._check_partition(model)
        learned = checkpoint.get("model")
        if not isinstance(learned, dict) or set(learned) != set(self.trainable):
            raise ValueError("Delta learned tensor membership differs")
        for name, value in learned.items():
            expected = self.learned_schema[name]
            if (
                not isinstance(value, torch.Tensor)
                or list(value.shape) != expected["shape"]
                or str(value.dtype) != expected["dtype"]
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("Invalid delta learned tensor shape, dtype or values")
        state.update(learned)
        model.load_state_dict(state, strict=True)


def restore_checkpoint_model(
    model: nn.Module, checkpoint: dict, guard: FrozenBaseDelta | None
) -> None:
    if checkpoint.get("schema_version") == 2:
        if guard is None:
            raise ValueError("A delta checkpoint requires its verified frozen base")
        guard.restore(model, checkpoint)
    elif checkpoint.get("schema_version") == 1 and guard is None:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        raise ValueError("Checkpoint model storage differs from the configured format")
