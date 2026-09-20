#!/usr/bin/env python3
"""Sentence-level multilingual NLI scores for RSNA reports."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, "/home/ubuntu/rsna-unifiedtheory-code")

from constants import TARGETS
from report_teacher import normalize_report

from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

SENTENCE_SPLIT = re.compile(r"[\n\r]{2,}|(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--neutral-baseline", type=float, default=0.22)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.train_csv)
    if "StudyInstanceUID" not in frame or "Report" not in frame:
        raise ValueError("train CSV must contain StudyInstanceUID and Report")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).eval()
    device = torch.device("cpu")
    model.to(device)

    config = model.config
    def find_index(name: str) -> int:
        for mapping in [getattr(config, "label2id", {}) or {},
                        {str(value): int(key) for key, value in (getattr(config, "id2label", {}) or {}).items()}]:
            for label, idx in mapping.items():
                if name in str(label).lower():
                    return int(idx)
        raise ValueError(f"could not find {name} class")

    entailment = find_index("entailment")
    neutral = find_index("neutral")

    # Pre-split reports into normalized sentences
    study_sents = []
    for _, row in frame.iterrows():
        raw = "" if pd.isna(row["Report"]) else str(row["Report"])
        sents = [normalize_report(s) for s in split_sentences(raw)]
        study_sents.append(sents)

    output = pd.DataFrame({"StudyInstanceUID": frame["StudyInstanceUID"].astype(str)})

    for target in TARGETS:
        hypothesis = HYPOTHESES[target]
        all_premises = []
        study_idx = []
        for i, sents in enumerate(study_sents):
            for s in sents:
                all_premises.append(s)
                study_idx.append(i)

        if not all_premises:
            output[f"{target}__nli"] = 0.0
            continue

        probs = []
        for start in range(0, len(all_premises), args.batch_size):
            batch_premises = all_premises[start : start + args.batch_size]
            encoded = tokenizer(
                batch_premises,
                [hypothesis] * len(batch_premises),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            p = torch.softmax(logits.float(), dim=-1)
            finding_p = p[:, entailment] + args.neutral_baseline * p[:, neutral]
            probs.append(finding_p.cpu().numpy())

        all_probs = np.concatenate(probs)
        study_max = np.full(len(frame), 0.0, dtype=float)
        for idx, prob in zip(study_idx, all_probs):
            if prob > study_max[idx]:
                study_max[idx] = prob
        output[f"{target}__nli"] = study_max
        print(f"scored {target}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(output)} reports, {len(all_premises)} sentences")


if __name__ == "__main__":
    main()
