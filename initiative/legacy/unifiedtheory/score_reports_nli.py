#!/usr/bin/env python3
"""Score report findings with a multilingual NLI teacher.

This is an optional second weak-supervision channel.  It never replaces the
inspectable rules or expert labels in :mod:`report_teacher`; it only emits
``<target>__nli`` probabilities that can be calibrated out of fold there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .constants import TARGETS
    from .report_teacher import normalize_report
except ImportError:
    from constants import TARGETS
    from report_teacher import normalize_report


HYPOTHESES = {
    "ACL": "The anterior cruciate ligament is abnormal.",
    "MCL": "The medial collateral ligament is abnormal.",
    "Medial Meniscus": "The medial meniscus is abnormal.",
    "Lateral Meniscus": "The lateral meniscus is abnormal.",
    "Medial OA": "There is medial compartment osteoarthritis.",
    "Lateral OA": "There is lateral compartment osteoarthritis.",
    "PF OA": "There is patellofemoral osteoarthritis.",
    "Effusion": "There is a knee joint effusion.",
    "Synovitis": "There is synovitis.",
    "Baker's": "There is a Baker cyst.",
    "Contusion": "There is a bone contusion.",
    "Fracture": "There is a fracture.",
}


def _label_index(config: object, name: str, override: int | None) -> int:
    if override is not None:
        return override
    mappings = [
        getattr(config, "label2id", {}) or {},
        {str(value): int(key) for key, value in (getattr(config, "id2label", {}) or {}).items()},
    ]
    for mapping in mappings:
        for label, index in mapping.items():
            if name in str(label).lower():
                return int(index)
    raise ValueError(
        f"could not identify {name!r} class from model config; pass --{name}-index"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--entailment-index", type=int)
    parser.add_argument("--neutral-index", type=int)
    parser.add_argument("--neutral-baseline", type=float, default=0.22)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable CUDA float16 autocast only for models explicitly validated in fp16",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    frame = pd.read_csv(args.train_csv)
    if "StudyInstanceUID" not in frame or "Report" not in frame:
        raise ValueError("train CSV must contain StudyInstanceUID and Report")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    device = torch.device(args.device)
    model.to(device).eval()
    entailment = _label_index(model.config, "entail", args.entailment_index)
    neutral = _label_index(model.config, "neutral", args.neutral_index)
    reports = frame["Report"].map(normalize_report).tolist()
    output = pd.DataFrame({"StudyInstanceUID": frame["StudyInstanceUID"].astype(str)})
    # mDeBERTa's own model card warns that its disentangled attention is not a
    # generally supported fp16 path.  Keep the public default in fp32; AMP is
    # an explicit, model-specific optimization rather than a silent accuracy
    # and stability change.
    amp = device.type == "cuda" and args.amp
    for target in TARGETS:
        probabilities: list[np.ndarray] = []
        hypothesis = HYPOTHESES[target]
        for start in range(0, len(reports), args.batch_size):
            premise = reports[start : start + args.batch_size]
            encoded = tokenizer(
                premise,
                [hypothesis] * len(premise),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**encoded).logits
            class_probability = torch.softmax(logits.float(), dim=-1)
            finding_probability = (
                class_probability[:, entailment]
                + args.neutral_baseline * class_probability[:, neutral]
            )
            probabilities.append(finding_probability.cpu().numpy())
        output[f"{target}__nli"] = np.concatenate(probabilities)
        print(f"scored {target}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(output)} reports")


if __name__ == "__main__":
    main()
