"""Stage the already transferred matched inputs; no reacquisition or model loading."""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and sys.dont_write_bytecode
    assert re.fullmatch("[0-9a-f]{40}", v["source_git"])
    assert datetime.now(timezone.utc) < datetime(2026, 10, 21, 16, 15, tzinfo=timezone.utc)
    files, hashes = v["files"], v["sha256"]
    assert set(files) == set(hashes) and sum(len(x.encode()) for x in files.values()) < 2 * 2**20
    assert all(hashlib.sha256(body.encode()).hexdigest() == hashes[name] for name, body in files.items())
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files)
    for name, body in files.items():
        ast.parse(body, filename=name)
    builder, policy = {"__name__": "contrast_worker_builder"}, {"__name__": "contrast_controls"}
    exec(compile(files["worker_builder.py"], "contrast_builder", "exec"), builder)
    exec(compile(files["learning_controls.py"], "contrast_controls", "exec"), policy)
    root, old, portable = builder["ROOT"], builder["ORIGINAL"], policy["PORTABLE"]
    read, sha, signature = policy["read"], policy["sha"], policy["signature"]
    assert root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.getuid()
    assert not any((root / name).exists() for name in ("code", "learning-intent.json", "prepared.json", "engineering"))
    receipt, receipt_raw = read(root / "transfer-accepted.json")
    assert receipt == v["transfer_receipt"] and receipt["CPU_job"] == "984691"
    assert receipt["all_original_sizes_SHA_EOF_verified"] and receipt["original_target_row_order_equal"]
    prepared, prepared_raw = read(root / "source-context-acceptance.json")
    assert sha(prepared_raw) == receipt["source_actual_sha256"]
    assert prepared["CPU_job"] == "984645" and prepared["all_roundtrips_bitwise_equal"]
    assert prepared["candidate_sha256"] == builder["CANDIDATE_SHA"]
    assert prepared["input_acceptance_sha256"] == "17195330aa5b4d73fef5d8071dfd4065c4cbd89830ecd5e6480fbe0f92e3fc61"
    terminals = policy["terminal_jobs"](subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", "984645,984690,984691",
        "--format=JobIDRaw,State,ExitCode,NodeList"], text=True))
    binding, raw = read(old / "binding.private.json", policy["BINDING_SHA"])
    read(old / "GPU-accepted.json", policy["ENGINEERING_SHA"])
    assert binding["source_git"] == "39e315c65546aa5205518713bc27349ec63e7538"
    assert files["worker.py"] == builder["build"]((old / "code/worker.py").read_text())
    original_modules = {n: h for n, h in binding["code_sha256"].items() if n.startswith("rsna/")}
    assert {n for n in files if n.startswith("rsna/")} == set(original_modules)
    for name, expected in original_modules.items():
        assert hashes[name] == expected and (old / "code" / name).read_bytes() == files[name].encode()
    for name, expected in builder["NUMERICAL_SHA"].items():
        assert original_modules[name] == expected
    old_index, _ = read(Path(binding["context_index_private_path"]), binding["context_index_sha256"])
    index, _ = read(root / "context-index.private.json", policy["INDEX_SHA"])
    assert receipt["destination_index_sha256"] == policy["INDEX_SHA"]
    assert len(index) == 16 and [r["uid"] for r in index] == [r["uid"] for r in old_index]
    members, seen, total = 0, set(), 0
    for i, row in enumerate(index):
        assert set(row["variants"]) == {"control", "normalized"}
        for role, item in row["variants"].items():
            path = root / "contexts" / f"{role}-{i:03d}.private.npz"
            assert item["path"] == str(path) and signature(path) == item["proof"]["stat"]
            assert item["provenance"]["role"] == role and item["provenance"]["metadata_width"] == 10
            assert item["provenance"]["candidate_sha256"] == builder["CANDIDATE_SHA"]
            assert path.stat().st_ino not in seen and path.stat().st_size == item["proof"]["bytes"]
            seen.add(path.stat().st_ino)
            members += 1
            total += path.stat().st_size
    assert members == 32 and total == receipt["bytes"] == 538307806
    target = Path(binding["targets16_private_path"])
    signature(target)
    assert sha(target.read_bytes()) == binding["targets16_sha256"]
    parent = signature(Path(binding["parent_weight_private_path"]))
    assert {"inode": parent[0], "bytes": parent[1], "mtime_ns": parent[2]} == binding["parent_weight_stat"]
    assert sha((portable / "assets/parent_source.private.py").read_bytes()) == binding["parent_source_sha256"]
    runtime, runtime_raw = read(portable / "runtime-import.private.json")
    versions = json.loads(runtime["stdout"])
    assert runtime["returncode"] == 0 and versions["torch"] == "2.14.0+cu130"
    assert versions["new_predictive_graph_imported"] is True
    bound = 3 * signature(old / "engineering/C/full/final.pt")[1] + 64 * 2**20
    assert bound < 6 * 2**30
    capacity = policy["capacity"](root)
    binding.update(source_git=v["source_git"], original_numerical_source_git=binding["source_git"],
        original_binding_sha256=sha(raw), original_engineering_receipt_sha256=policy["ENGINEERING_SHA"],
        branch="matched_visible_only_contrast", candidate_sha256=builder["CANDIDATE_SHA"],
        metadata_width=10, recipe=builder["recipe"](), roles=["control", "normalized"],
        phases=[["control", "full"], ["normalized", "stop1"], ["normalized", "resume"]],
        actual_update_budget=400, code_sha256=hashes,
        original_module_sha256=original_modules, context_index_private_path=str(root / "context-index.private.json"),
        context_index_sha256=policy["INDEX_SHA"], input_rendering_receipt=prepared,
        transfer_receipt_sha256=sha(receipt_raw), runtime_versions=versions,
        runtime_receipt_sha256=sha(runtime_raw), private_engineering_only=True)

    def once(path, value):
        with path.open("x") as out:
            out.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    once(root / "learning-intent.json", {"CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
         "automatic_retry": False, "code_sha256": hashes, "capacity": capacity})
    for name, body in files.items():
        path = root / "code" / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("x") as out:
            out.write(body)
    once(root / "binding.private.json", binding)
    result = {"status": "actual_contrast_matched_learning_CPU_admitted", "source_git": v["source_git"],
        "CPU_job": os.environ["SLURM_JOB_ID"], "binding_sha256": sha((root / "binding.private.json").read_bytes()),
        "context_files": members, "context_bytes": total, "original_terminal_jobs": terminals,
        "training_studies": 12, "engineering_prediction_studies": 4, "metadata_width": 10,
        "actual_update_budget": 400, "three_checkpoint_bound_bytes": bound, "capacity": capacity,
        "original_module_sha256": original_modules, "runtime_versions": versions,
        "source_data_reacquisition": False, "new_shared_files": 0, "GPU_dispatched": False}
    once(root / "prepared.json", result)
    policy["capacity"](root)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
