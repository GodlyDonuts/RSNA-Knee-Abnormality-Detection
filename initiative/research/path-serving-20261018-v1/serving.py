"""Small fixed-final path-head export and streamed inference; no training or uploads.

Requires the unchanged path_reader/online_inputs modules beside this module or
on the explicit scientific bundle import path. Actual T4 execution is separate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from path_reader import VERSION, PathReader

from rsna.schema import TARGETS

PARENT = "5bec977964231fe67009d7106ad63ec71fdbd9cde5ddeb2cc2883ac59b0a7be5"
SCHEMA = "fixed_final_dense_path_head_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def export_final(snapshot: Path, terminal: Path, output: Path) -> dict:
    """Export only the completed new reader, never the optimizer or clinical rows."""
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(terminal.read_text())
    if (
        result.get("status") != "actual_fixed_final_dense_coherent_path3epoch_complete"
        or result.get("epochs") != 3
        or result.get("optimizer_updates") != 843
        or result.get("encoder_trained") is not False
        or result.get("base_checkpoint_sha256") != PARENT
        or result.get("checkpoint_sha256") != digest(snapshot)
    ):
        raise ValueError("Actual fixed-final full-path terminal/checkpoint required")
    state = torch.load(snapshot, map_location="cpu", weights_only=True)
    b = state["binding"]
    if (
        state["epoch"] != 3
        or state["completed"] != 843
        or state["cursor"] != 0
        or not state["optimizer_boundary"]
        or b.get("engine") != VERSION
        or b.get("diagnostic_only") is not False
        or b.get("arm") != result.get("arm")
        or b.get("arm") not in {"aligned", "null"}
        or b.get("source_git") != result.get("source_git")
        or b.get("source_manifest_sha256") != result.get("source_manifest_sha256")
        or b.get("GPU_acceptance_sha256") != result.get("GPU_acceptance_sha256")
        or b["data"]["original"]["parent_sha256"] != PARENT
        or b["data"].get("sampling") != "dense"
        or b["data"].get("encoder_precision") != "float32"
        or b["targets"] != list(TARGETS)
    ):
        raise ValueError("Complete matching production reader and FP32 dense policy required")
    reader = PathReader()
    reader.load_state_dict(state["model"], strict=True)
    if any(
        t.dtype != torch.float32 or not torch.isfinite(t).all()
        for t in reader.state_dict().values()
    ):
        raise ValueError("Finite FP32 final reader required")
    policy = {
        "schema": SCHEMA,
        "reader": VERSION,
        "arm": b["arm"],
        "epochs": 3,
        "updates": 843,
        "source_git": b["source_git"],
        "execution_wrapper_source_git": b["execution_wrapper_source_git"],
        "source_manifest_sha256": b["source_manifest_sha256"],
        "actual_integrated_GPU_receipt_sha256": b["GPU_acceptance_sha256"],
        "original_final_snapshot_sha256": result["checkpoint_sha256"],
        "original_final_terminal_sha256": digest(terminal),
        "parent_checkpoint_sha256": PARENT,
        "encoder_precision": "float32",
        "sampling": "dense",
        "chunk_size": 12,
        "targets": list(TARGETS),
        "channels": 1024,
        "width": 64,
        "new_head_parameters": sum(p.numel() for p in reader.parameters()),
        "no_optimizer_rows_predictions_or_pixels_in_export": True,
        "T4_runtime_qualified": False,
    }
    # O_EXCL prevents replacement of an existing immutable head.
    with output.open("xb") as stream:
        torch.save({"policy": policy, "model": reader.state_dict()}, stream)
    return {"policy": policy, "bytes": output.stat().st_size, "sha256": digest(output)}


class PathInference:
    def __init__(
        self,
        head: Path,
        expected_sha256: str,
        *,
        parent_sha256: str,
        device,
        vectorized_scoring: bool = False,
    ):
        if head.stat().st_size > 2**20 or digest(head) != expected_sha256:
            raise ValueError("Exact small immutable path-head export required")
        value = torch.load(head, map_location="cpu", weights_only=True)
        if set(value) != {"policy", "model"}:
            raise ValueError("Export contains unexpected data")
        p = value["policy"]
        if (
            p["schema"] != SCHEMA
            or p["reader"] != VERSION
            or p["arm"] not in {"aligned", "null"}
            or (p["epochs"], p["updates"]) != (3, 843)
            or p["parent_checkpoint_sha256"] != parent_sha256
            or parent_sha256 != PARENT
            or p["encoder_precision"] != "float32"
            or p["sampling"] != "dense"
            or p["targets"] != list(TARGETS)
            or (p["channels"], p["width"], p["chunk_size"]) != (1024, 64, 12)
        ):
            raise ValueError("Fixed-final same-parent path policy required")
        self.device = torch.device(device)
        if type(vectorized_scoring) is not bool:
            raise ValueError("Explicit candidate scoring policy required")
        reader_class = PathReader
        self.scoring_version = "reference:" + VERSION
        if vectorized_scoring:
            from path_vectorized import SCORING_VERSION, FastPathReader

            reader_class = FastPathReader
            self.scoring_version = SCORING_VERSION
        self.vectorized_scoring = vectorized_scoring
        self.reader = reader_class().to(self.device).eval()
        self.reader.load_state_dict(value["model"], strict=True)
        self.reader.requires_grad_(False)
        if any(
            t.dtype != torch.float32 or not torch.isfinite(t).all()
            for t in self.reader.state_dict().values()
        ):
            raise ValueError("Finite complete FP32 path state required")
        self.policy = p

    @torch.inference_mode()
    def predict_csv(self, provider, output: Path, *, deadline_seconds=8.5 * 3600):
        """Write complete CSV atomically; no labels, rounding, thresholding or fallback.

        The provider is the same StreamedFP32PathData over test-only study IDs.
        This routine does not decode competition DICOMs or qualify T4 throughput.
        """
        binding = provider.binding
        if (
            binding["original"].get("parent_sha256") != PARENT
            or binding.get("sampling") != "dense"
            or binding.get("encoder_precision") != "float32"
            or binding.get("encoder_chunk_size") != 12
            or type(deadline_seconds) not in {int, float}
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
            or len(provider) <= 0
        ):
            raise ValueError("Original dense FP32 provider and positive deadline required")
        partial = output.with_name(output.name + ".partial")
        if output.exists():
            raise FileExistsError(output)
        start = time.monotonic()
        seen = set()
        max_study_seconds = 0.0
        windows = 0
        with partial.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["StudyInstanceUID", *TARGETS])
            for i in range(len(provider)):
                before = time.monotonic()
                if before - start >= deadline_seconds:
                    raise TimeoutError(
                        "Inference deadline reached; incomplete CSV is not published"
                    )
                row = provider[i]
                uid = row["study_uid"]
                if not isinstance(uid, str) or not uid or uid in seen:
                    raise ValueError("Every test study must occur exactly once")
                logits, _ = self.reader(
                    row["maps"].to(self.device),
                    row["base_logits"].to(self.device),
                    row["plan"],
                    arm=self.policy["arm"],
                )
                probabilities = logits.sigmoid().float().cpu()
                if probabilities.shape != (12,) or not torch.isfinite(probabilities).all():
                    raise ValueError("Twelve finite MRI-only probabilities required")
                if time.monotonic() - start >= deadline_seconds:
                    raise TimeoutError(
                        "Inference deadline reached; incomplete CSV is not published"
                    )
                writer.writerow([uid, *probabilities.tolist()])
                seen.add(uid)
                windows += len(row["maps"])
                max_study_seconds = max(max_study_seconds, time.monotonic() - before)
                del row, logits, probabilities
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-overwrite publication on the same filesystem. Partial files
        # survive failures for diagnosis but are never named submission.csv.
        os.link(partial, output)
        partial.unlink()
        return {
            "rows": len(seen),
            "columns": 13,
            "windows": windows,
            "elapsed_seconds": time.monotonic() - start,
            "max_study_seconds": max_study_seconds,
            "csv_sha256": digest(output),
            "head_policy": self.policy,
            "actual_T4_qualification_claim": False,
            "candidate_vectorized_scoring": self.vectorized_scoring,
            "scoring_execution_version": self.scoring_version,
        }


class ShardedProvider:
    """Disjoint study lanes for two independent GPU processes; never DataParallel.

    Each process creates its own frozen parent/provider on cuda:rank. Original
    study order and per-study numerical operations stay unchanged.
    """

    def __init__(self, provider, *, rank: int, world_size: int = 2):
        if (
            type(rank) is not int
            or type(world_size) is not int
            or world_size not in {1, 2}
            or not 0 <= rank < world_size
        ):
            raise ValueError("One or two deterministic study lanes required")
        self.provider, self.rank, self.world_size = provider, rank, world_size
        self.indices = range(rank, len(provider), world_size)
        self.binding = provider.binding

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.provider[self.indices[i]]


def merge_shards(shards: list[Path], study_uids: list[str], output: Path) -> dict:
    """Combine completed GPU lanes into original test.csv order, requiring full coverage."""
    if output.exists():
        raise FileExistsError(output)
    if (
        not study_uids
        or any(not isinstance(u, str) or not u for u in study_uids)
        or len(set(study_uids)) != len(study_uids)
    ):
        raise ValueError("Unique complete original test roster required")
    if len(shards) not in {1, 2} or len(set(shards)) != len(shards):
        raise ValueError("One or two distinct complete lane CSVs required")
    header = ["StudyInstanceUID", *TARGETS]
    wanted, rows = set(study_uids), {}
    for path in shards:
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != header:
                raise ValueError("All lanes must use the exact competition schema")
            for row in reader:
                if len(row) != 13 or row[0] not in wanted or row[0] in rows:
                    raise ValueError("Unexpected, repeated or malformed study in lanes")
                probabilities = [float(x) for x in row[1:]]
                if not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities):
                    raise ValueError("Finite probabilities required in every lane")
                rows[row[0]] = row
    if set(rows) != wanted:
        raise ValueError("Incomplete MRI inference; do not publish a partial submission")
    partial = output.with_name(output.name + ".partial")
    with partial.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows[uid] for uid in study_uids)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, output)
    partial.unlink()
    return {
        "rows": len(rows),
        "lanes": len(shards),
        "columns": 13,
        "original_test_order_preserved": True,
        "csv_sha256": digest(output),
    }
