#!/usr/bin/env python3
"""Create frozen multilingual report embeddings for image distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    frame = pd.read_csv(args.train_csv, dtype={"StudyInstanceUID": str})
    required = {"StudyInstanceUID", "Report"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"training table is missing {sorted(missing)}")
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("training table contains duplicate studies")
    texts = frame["Report"].fillna("").astype(str).tolist()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    model = AutoModel.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    ).to(args.device).eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
            print(f"embedded {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    values = np.concatenate(embeddings).astype(np.float16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        uids=frame["StudyInstanceUID"].astype(str).to_numpy(dtype=str),
        embeddings=values,
    )
    metadata = {
        "schema_version": 1,
        "model": args.model_name,
        "rows": len(frame),
        "dimension": int(values.shape[1]),
        "max_length": args.max_length,
        "normalized": True,
        "source_sha256": file_sha256(args.train_csv),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
