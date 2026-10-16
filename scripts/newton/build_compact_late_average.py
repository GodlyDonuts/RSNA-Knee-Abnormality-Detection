"""Build only the preregistered12/14/16 parameter average on CPU after original preservation.

One derived checkpoint, exact epoch16 buffers, no model inference, clinical metric,
training, credential, cloud upload or competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path


def verify_buffers(model, averaged, final, reconstructed_before):
    """Saved buffers retain E16; nonsaved graph caches retain blueprint values."""
    import torch

    saved = set(model.state_dict())
    parameters = set(dict(model.named_parameters()))
    buffers = dict(model.named_buffers())
    persistent = set(buffers) & saved
    reconstructed = set(buffers) - saved
    if saved != set(averaged) or saved != set(final) or saved != parameters | persistent:
        raise ValueError("Complete original parameter/persistent-buffer graph required")
    if set(reconstructed_before) != reconstructed:
        raise ValueError("Exact blueprint nonpersistent-buffer inventory required")
    if any(
        not torch.equal(averaged[name], final[name]) or not torch.equal(buffers[name], final[name])
        for name in persistent
    ) or any(not torch.equal(buffers[name], reconstructed_before[name]) for name in reconstructed):
        raise ValueError("Saved E16 buffers or reconstructed graph caches changed")
    return {"persistent_buffers": len(persistent), "reconstructed_buffers": len(reconstructed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.uname().nodename != "evc33" or os.environ.get("SLURM_JOB_GPUS") or args.output.exists():
        raise ValueError("Original owning-node CPU-only fresh derived output required")
    if shutil.disk_usage("/tmp").free <= 578871834563:
        raise OSError(
            "All original/new obligations, average1GiB cap and reserve must remain funded"
        )
    import torch

    from rsna.competitive.checkpoint import atomic_json, file_sha256, save_checkpoint
    from rsna.competitive.compact_average import EPOCHS, POLICY, derive_checkpoint
    from rsna.competitive.compact_bag import CompactBagClassifier
    from rsna.competitive.compact_infer import _blueprint

    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("No CUDA initialization permitted")
    checkpoints = []
    sources = []
    shas = []
    original_binding = json.loads((args.original / "production/run.json").read_text())["binding"]
    final = json.loads((args.original / "production/result.json").read_text())
    if final["status"] != "fixed_final_fit_complete" or final.get("diagnostic_only") is not False:
        raise ValueError("Actual complete original fixed-final fit required")
    for epoch in EPOCHS:
        path = args.original / f"production/epoch-checkpoints/epoch-{epoch:03d}.pt"
        side = json.loads(path.with_suffix(".json").read_text())
        proof_path = args.original / f"publication/epoch-{epoch:03d}/verified.json"
        proof = json.loads(proof_path.read_text())
        if (
            proof.get("status") != "private_epoch_fully_verified"
            or proof.get("epoch") != epoch
            or proof.get("files_verified_by_download") != 8
            or proof.get("checkpoint_sha256") != side["checkpoint_sha256"]
            or side["binding"] != original_binding
            or path.is_symlink()
        ):
            raise ValueError(
                "Each exact original late epoch must already be privately fully verified"
            )
        raw = path.read_bytes()
        if (
            len(raw) != side["checkpoint_bytes"]
            or not 250 * 2**20 <= len(raw) <= 512 * 2**20
            or hashlib.sha256(raw).hexdigest() != side["checkpoint_sha256"]
        ):
            raise ValueError("Actual immutable source checkpoint bytes differ")
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        checkpoints.append(value)
        shas.append(side["checkpoint_sha256"])
        sources.append(
            {
                "epoch": epoch,
                "checkpoint_sha256": side["checkpoint_sha256"],
                "checkpoint_bytes": len(raw),
                "private_revision": proof["revision"],
                "snapshot_sha256": file_sha256(path.with_suffix(".json")),
                "private_verified_receipt_sha256": file_sha256(proof_path),
            }
        )
        del raw
    model = CompactBagClassifier(_blueprint()).float()
    if sum(p.numel() for p in model.parameters()) != 73136096:
        raise ValueError("Actual original full graph differs")
    parameters = set(dict(model.named_parameters()))
    value = derive_checkpoint(checkpoints, parameters, shas)
    saved = set(model.state_dict())
    reconstructed = {n: b.clone() for n, b in model.named_buffers() if n not in saved}
    model.load_state_dict(value["model"], strict=True)
    buffer_counts = verify_buffers(model, value["model"], checkpoints[-1]["model"], reconstructed)
    args.output.mkdir(mode=0o700)
    path = args.output / "checkpoint.pt"
    save_checkpoint(path, value)
    path.chmod(0o600)
    owned = sum(
        p.stat().st_blocks * 512 for p in args.output.parent.rglob("*") if p.is_file() or p.is_dir()
    )
    if owned > 2**30:
        raise OSError("Derived average/source/output exceeded separately funded1GiB cap")
    if torch.cuda.is_initialized():
        raise ValueError("No CUDA initialized by CPU construction")
    receipt = {
        "status": "actual_fixed_late_parameter_average_CPU_created",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "node": "evc33",
        "policy": POLICY,
        "source_epochs": sources,
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_bytes": path.stat().st_size,
        "engine": value["binding"]["engine"],
        "parameter_count": 73136096,
        "all_final_serialized_buffers_exact": True,
        "nonpersistent_graph_buffers_unchanged_from_blueprint": True,
        **buffer_counts,
        "BN_counters_retained": sum(
            n.endswith("num_batches_tracked") for n, _ in model.named_buffers()
        ),
        "original_fixed_final_result_sha256": file_sha256(args.original / "production/result.json"),
        "actual_owned_physical_bytes": owned,
        "GPU_or_inference_or_training_executed": False,
        "target_runtime_qualified": False,
        "clinical_accuracy_claim": False,
        "competition_submission": False,
    }
    atomic_json(args.output / "accepted.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
