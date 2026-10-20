"""Build only a private16-study matched pilot cache; never modifies a live cache."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path("/tmp/rsna-predictive-visible-contrast16-20261020-v1")
RAW = Path("/tmp/rsna-predictive-raw16-cache-20261020-v1")
CONTEXT = Path("/tmp/rsna-predictive-context16-cache-20261020-v1")
SOURCE = Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code")
INDEX_SHA = "e3585fb3cafab3e0ebbca0fb03953e38bd879376956a7a142cc5b4781784008d"
GIB, MIB = 2**30, 2**20
CACHE_CAP, CONTROL_CAP = GIB, 8 * MIB
GENERATIONS = ("cache-20261020-v1", "alias-recovery-20261020-v3", "capacity-recovery-20261020-v1",
               "http-recovery-20261020-v1", "data-recovery-20261020-v1", "http499-recovery-20261020-v1")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def physical_bytes(paths):
    seen, total = set(), 0
    for p in paths:
        st = p.stat()
        assert p.is_file() and not p.is_symlink() and st.st_uid == os.getuid()
        key = st.st_dev, st.st_ino
        if key not in seen:
            seen.add(key)
            total += st.st_size
    return total


def budget(free, raw_paid, context_paid, shared_text):
    assert all(type(x) is int and x >= 0 for x in (free, raw_paid, context_paid))
    assert raw_paid < 320 * GIB and context_paid < 96 * GIB
    # Preserve the larger deployment other-node bound, all source scratch,
    # every unpaid original cache byte and the standing20GiB free reserve.
    old_required = 20 * GIB + 68 * GIB + 594 * MIB + 6 * GIB + 320 * GIB - raw_paid + 96 * GIB - context_paid
    required = old_required + CACHE_CAP + CONTROL_CAP
    row = next(line.split() for line in shared_text.splitlines() if line.strip().startswith("/home/sa305415"))
    shared_free = (int(row[2]) - int(row[1])) * 1024
    shared_files = int(row[6]) - int(row[5])
    assert free > required and shared_free > 20 * GIB + 123 * MIB and shared_files >= 5000 + 376
    return {"free_bytes_before": free, "old_required_bytes": old_required,
            "new_cache_cap_bytes": CACHE_CAP, "new_control_cap_bytes": CONTROL_CAP,
            "total_required_bytes": required, "margin_after_all_liabilities_bytes": free - required,
            "raw_paid_bytes": raw_paid, "context_paid_bytes": context_paid,
            "shared_free_after_old_liabilities_bytes": shared_free - 123 * MIB,
            "shared_files_after_old_liabilities": shared_files - 376,
            "new_shared_bytes_or_files": 0, "new_GPU_or_checkpoint_scope": 0}


def main(v):
    assert os.uname().nodename == "evc33" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ["SLURM_CPUS_PER_TASK"] == "1"
    assert sys.dont_write_bytecode and not ROOT.exists()
    os.umask(0o077)
    for name, expected in v["scientific_modules"].items():
        assert sha((SOURCE / name).read_bytes()) == expected
    assert sha(v["candidate_source"].encode()) == v["candidate_sha256"]
    assert sha(v["input_acceptance_body"].encode()) == v["input_acceptance_sha256"]
    acceptance = json.loads(v["input_acceptance_body"])
    assert acceptance["CPU_job"] == "984583" and acceptance["candidate_sha256"] == v["candidate_sha256"]
    assert acceptance["all_targets_masks_complete_priors_and_first_eight_metadata_columns_unchanged"]
    assert acceptance["hidden_source_intervention"]["context0_images_metadata_masks_and_scales_bitwise_equal"]
    raw_bytes = (RAW / "cache-index.private.json").read_bytes()
    assert sha(raw_bytes) == INDEX_SHA
    rows = json.loads(raw_bytes)
    ctx_bytes = (CONTEXT / "context-index.private.json").read_bytes()
    ctx_actual = json.loads((CONTEXT / "actual.json").read_bytes())
    assert sha(ctx_bytes) == ctx_actual["private_index_sha256"]
    original_context = json.loads(ctx_bytes)
    assert len(rows) == len(original_context) == 16
    assert [r["uid"] for r in rows] == [r["uid"] for r in original_context]
    raw_files, context_files = [Path(r["path"]) for r in rows], [Path(r["path"]) for r in original_context]
    for suffix in GENERATIONS:
        a = Path("/tmp/rsna-predictive-full-raw-" + suffix)
        b = Path("/tmp/rsna-predictive-full-context-" + suffix)
        assert a.is_dir() and b.is_dir() and not a.is_symlink() and not b.is_symlink()
        raw_files.extend((a / "cache").glob("*.npz"))
        context_files.extend(b.glob("context-*.npz"))
    raw_paid, context_paid = physical_bytes(raw_files), physical_bytes(context_files)
    funding = budget(shutil.disk_usage("/tmp").free, raw_paid, context_paid,
                     subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True))
    sys.path.insert(0, str(SOURCE))
    from rsna.competitive.predictive import cache, context_cache

    module_name = "rsna.competitive.predictive.visible_contrast"
    assert module_name not in sys.modules
    candidate = types.ModuleType(module_name)
    sys.modules[module_name] = candidate
    exec(compile(v["candidate_source"], "<visible_contrast>", "exec"), candidate.__dict__)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    ROOT.mkdir(mode=0o700)
    once(ROOT / "intent.json", {"CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
                               "funding": funding, "automatic_retry": False})
    with (ROOT / "candidate_source.py").open("x") as stream:
        stream.write(v["candidate_source"])
    once(ROOT / "input-acceptance.json", acceptance)
    rng, start, paid, index = torch.get_rng_state().clone(), time.monotonic(), 0, []
    for i, (row, old_row) in enumerate(zip(rows, original_context, strict=True)):
        groups, _ = cache.load(Path(row["path"]), row["proof"])
        reference = context_cache.load(old_row["path"], old_row["proof"], provenance=old_row["provenance"])
        variants = {}
        for role, normalize in (("control", False), ("normalized", True)):
            contexts, _ = candidate.assemble(groups, side=320, context_cap=24, classes=12,
                                             normalize_images=normalize)
            for old, new in zip(reference, contexts, strict=True):
                assert new["token_metadata"].shape[1] == new["region_metadata"].shape[1] == 10
                for key in ("targets", "observed"):
                    assert torch.equal(old[key], new[key])
                for key in ("token_metadata", "region_metadata"):
                    assert torch.equal(old[key], new[key][:, :8])
                for key in ("indptr", "regions", "log_prior"):
                    assert torch.equal(getattr(old["supports"], key), getattr(new["supports"], key))
                if role == "control":
                    assert torch.equal(old["images"], new["images"])
            provenance = {"original_raw_cache_sha256": row["proof"]["sha256"],
                          "original_context_sha256": old_row["proof"]["sha256"],
                          "original_source_revision": "52f995f4568e36bfa5fbbaf69a89af41226b4f5d",
                          "candidate_sha256": v["candidate_sha256"], "candidate_version": candidate.VERSION,
                          "role": role, "source_git": v["source_git"], "metadata_width": 10,
                          "side": 320, "halo": 8, "context_cap": 24, "classes": 12,
                          "original_scientific_modules": v["scientific_modules"]}
            path = ROOT / f"{role}-{i:03d}.private.npz"
            assert paid < CACHE_CAP
            proof = context_cache.write(path, contexts, provenance, cap_bytes=min(128 * MIB, CACHE_CAP - paid))
            restored = context_cache.load(path, proof, provenance=provenance)
            for left, right in zip(contexts, restored, strict=True):
                for key in context_cache.KEYS:
                    assert torch.equal(left[key], right[key])
                for key in ("indptr", "regions", "log_prior"):
                    assert torch.equal(getattr(left["supports"], key), getattr(right["supports"], key))
            paid += proof["bytes"]
            variants[role] = {"path": str(path), "proof": proof, "provenance": provenance}
            del contexts, restored
        index.append({"uid": row["uid"], "variants": variants})
        assert cache.signature(Path(row["path"])) == row["proof"]["stat"]
        assert torch.equal(rng, torch.get_rng_state())
        del groups, reference
    once(ROOT / "context-index.private.json", index)
    out = {"status": "actual_contrast_matched16_CPU_contexts_accepted",
           "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
           "candidate_sha256": v["candidate_sha256"], "input_acceptance_sha256": v["input_acceptance_sha256"],
           "fitting_training_studies": 12, "disjoint_engineering_prediction_studies": 4,
           "context_files": 32, "cache_bytes": paid, "funding": funding,
           "original_raw_index_sha256": INDEX_SHA, "original_context_index_sha256": sha(ctx_bytes),
           "private_index_sha256": sha((ROOT / "context-index.private.json").read_bytes()),
           "all_roundtrips_bitwise_equal": True, "global_RNG_unchanged": True,
           "original_control_images_targets_priors_and_metadata_preserved": True,
           "label_encoder_optimizer_GPU_reads_or_updates": 0,
           "clinical_accuracy_or_GPU_training_acceptance": False, "elapsed_seconds": time.monotonic() - start}
    once(ROOT / "actual.json", out)
    assert physical_bytes([p for p in ROOT.iterdir() if p.is_file() and p.suffix != ".npz"]) < CONTROL_CAP
    print(json.dumps(out, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
