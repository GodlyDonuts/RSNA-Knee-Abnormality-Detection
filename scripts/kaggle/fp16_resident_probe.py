"""Explicit CUDA FP16 throughput probe; original weights and MRI inputs remain fixed."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import torch

if __package__:
    from .resident_infer import ResidentInference
else:
    from resident_infer import ResidentInference

from rsna.schema import TARGETS


def predict_scores(model, loader, device, *, precision, seed, fold):
    if precision not in {"fp32", "fp16"}:
        raise ValueError("Exact FP32 reference or FP16 candidate compute required")
    identifiers, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            values = [
                batch[name].to(device)
                for name in ("images", "features", "series_mask", "slice_mask", "positions")
            ]
            with torch.autocast(device.type, dtype=torch.float16, enabled=precision == "fp16"):
                logits = model(*values)
            if (
                logits.shape != (len(batch["study_uid"]), len(TARGETS))
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("Nonfinite or malformed FP16 result; no numerical fallback")
            valid = batch.get("sample_valid", torch.ones(len(batch["study_uid"]), dtype=torch.bool))
            if valid.dtype != torch.bool or valid.shape != (len(batch["study_uid"]),):
                raise ValueError("Original genuine-study mask required")
            probabilities.append(logits.float().sigmoid().cpu().numpy()[valid.cpu().numpy()])
            identifiers.extend(uid for uid, keep in zip(batch["study_uid"], valid.tolist()) if keep)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Nonempty distinct original prediction studies required")
    frame = pd.DataFrame(np.concatenate(probabilities), columns=TARGETS)
    frame.insert(0, "seed", seed)
    frame.insert(0, "fold", fold)
    frame.insert(0, "study_uid", identifiers)
    return frame, {}


class FP16ResidentProbe(ResidentInference):
    def __init__(self, args, *, device_index=0, admit_performance_probe=False):
        if admit_performance_probe is not True or args.precision != "fp16" or args.device != "cuda":
            raise ValueError("Explicit CUDA-only FP16 performance probe admission required")
        reference = copy.copy(args)
        reference.precision = "fp32"
        # Reconstruct and verify the original checkpoint/base/source using the
        # unchanged reference path; only the ensuing compute context differs.
        super().__init__(reference, device_index=device_index)
        self.identity = self.options(args)
        self.inference_precision = "fp16"

    def predict(self, loader):
        return predict_scores(
            self.model,
            loader,
            self.device,
            precision="fp16",
            seed=self.config.seed,
            fold=self.config.fold,
        )

    def run(self, args):
        result = super().run(args)
        result.update(
            performance_probe_only=True,
            competition_submission_authorized=False,
            FP32_prediction_agreement_measured=False,
        )
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
        return result
