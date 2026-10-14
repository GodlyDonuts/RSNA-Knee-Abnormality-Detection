"""Paired two-T4 checkpoint-continuous fixed-10 production session (issues #55/#60).

PREPARED ONLY; root alone reviews, funds, pushes and dispatches. One Kaggle "T4 x2" session
continues BOTH predeclared arms (plain-mean on device 0, disease-query on device 1) as two
independent single-GPU trainer processes over one shared read-only restored cache. Neither
recipe changes and it is not DDP. Every session resumes from the one bound previous paired
kernel output (a single extra kernel source, keeping the admitted 21-kernel/3-Dataset shape),
stops both arms at the same scheduled optimizer boundary and publishes each arm's exact
continuation state plus every newly completed immutable epoch write-once under
``/kaggle/working``. Only the final session reaches update 2810 and runs the trainer's one
1156-study evaluation, per arm. Any failure stops and reaps the sibling and publishes nothing.

The frozen engineering wrapper/plan (``window_bag_t4.py``/``window_bag_plan.py``) and the
immutable 971d7dac numerical package are reused byte-for-byte; this file is the only extra
bound source of the production package, so its manifest maps exactly onto the engineering
manifest that the accepted concurrent engineering receipt binds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

ARMS = t4.ARMS
PAIRED_MODULE = "scripts/kaggle/window_bag_paired.py"
# Bound current sources of the frozen engineering package (prepared source 6bd78096).
FROZEN_ENGINEERING_BOUND = {
    "scripts/kaggle/window_bag_t4.py": (
        "cd9f1fd9d271cea73262483353b29ef9050b61184a1d165931191222e8c43e85"
    ),
    "scripts/kaggle/window_bag_plan.py": (
        "b87a54355fccd009c5490c67ec48c1b3f8b35a1a8bde6ca71453cb64e7968503"
    ),
}
PAIRED_MODE = "paired_segment"
PAIRED_STATUS = "source_window_bag_fixed10_paired_two_T4_segment_completed"
PAIRED_RECEIPT = "window-bag-paired-segment-accepted.json"
PAIRED_DIR = "window-bag-paired"
# Combined scratch held by both arms' run directories (cache excluded); same 16 GiB ceiling
# as every saved-output cap.
SCRATCH_CAP_BYTES = plan.OUTPUT_CAP_CEILING_BYTES
PREVIOUS_KEYS = {
    "kernel_slug",
    "version",
    "source_sha256",
    "receipt_sha256",
    "publication_sha256",
    "checkpoint_sha256",
}
SEGMENT_KEYS = {
    "segment",
    "start_step",
    "end_step",
    "previous_output",
    "engineering_receipt_sha256",
    "engineering_package_manifest_sha256",
    "production_config_sha256",
    "output_cap_bytes",
}
BINDING_KEYS = (
    "engineering_receipt_sha256",
    "engineering_package_manifest_sha256",
    "production_config_sha256",
)


def manifest_text(manifest: dict) -> str:
    """Exact bytes ``window_bag_notebook.build`` hashes for a package manifest."""
    return json.dumps(manifest, sort_keys=True, indent=2) + "\n"


def engineering_manifest_sha256(manifest: dict) -> str:
    """The frozen engineering package this production package extends by one file only."""
    bound = manifest.get("bound_current") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"science_git", "files", "bound_current"}
        or manifest["science_git"] != t4.SCIENCE_GIT
        or not manifest["files"]
        or not isinstance(bound, dict)
        or set(bound) != {*FROZEN_ENGINEERING_BOUND, PAIRED_MODULE}
        or any(bound[k] != v for k, v in FROZEN_ENGINEERING_BOUND.items())
        or not t4._hex64(bound[PAIRED_MODULE])
        or PAIRED_MODULE in manifest["files"]
    ):
        raise ValueError("Production package must be the frozen engineering package plus runner")
    engineering = dict(manifest, bound_current=dict(FROZEN_ENGINEERING_BOUND))
    return hashlib.sha256(manifest_text(engineering).encode()).hexdigest()


def _epochs_completed(start: int, end: int) -> int:
    return end // plan.UPDATES_PER_EPOCH - start // plan.UPDATES_PER_EPOCH


def _segment_epochs_bound(updates: int) -> int:
    """Most epoch boundaries any contiguous segment of ``updates`` updates can cross."""
    return math.ceil(updates / plan.UPDATES_PER_EPOCH) + 1


def scratch_bytes(measured: dict, segment: tuple[int, int]) -> int:
    """Both arms' run directories: staged/rewritten continuation state (two copies during the
    trainer's atomic replace), every new epoch snapshot and the sidecars."""
    per_arm = (2 + _epochs_completed(*segment)) * measured["checkpoint_bytes"]
    return len(ARMS) * (per_arm + plan.PUBLICATION_SIDECAR_BYTES)


def output_cap(measured: dict, segment: tuple[int, int]) -> int:
    return len(ARMS) * plan.publication_bytes(measured, segment)


def paired_plan(measured: dict, gpu_quota: dict, free_scratch_bytes: int) -> dict:
    """Fixed 2810-update schedule for both arms per session, from ACTUAL concurrent timings.

    ``measured`` must be the ``measured`` record of the accepted concurrent two-T4
    engineering receipt (slowest arm while both ran at once). Each session is charged once
    for both provisioned T4s for its full timeout. Besides the trainer's own 25-update
    checkpoint rewrites, a session spends startup, final evaluation, staging both previous
    checkpoints and publishing both arms' continuation state plus every completed epoch.
    """
    keys = set(t4.MEASURED_KEYS)
    if set(measured) != keys or any(
        type(measured[k]) not in (int, float) or not measured[k] > 0 for k in keys
    ):
        raise ValueError("Actual measured concurrent T4 timings/bytes required")
    usable = plan.SESSION_LIMIT_SECONDS * plan.SESSION_MARGIN
    per_update = measured["seconds_per_update"] + measured["checkpoint_seconds"] / 25
    fixed = (
        measured["startup_seconds"]
        + measured["final_evaluation_seconds"]
        + len(ARMS) * measured["checkpoint_seconds"]
    )

    def seconds_for(updates: int) -> float:
        copies = len(ARMS) * (1 + _segment_epochs_bound(updates))
        return fixed + updates * per_update + copies * measured["checkpoint_seconds"]

    updates = min(int(max(usable - fixed, 0) // per_update), plan.TOTAL_UPDATES)
    while updates >= 1 and seconds_for(updates) > usable:
        updates -= 1
    if updates < 1:
        raise ValueError("Measured concurrent T4 performance cannot fit one update per session")
    segments = plan.segment_boundaries(updates)
    timeout = min(plan.SESSION_LIMIT_SECONDS, math.ceil(usable))
    charge = plan.session_charge(timeout)
    available = plan.conservative_available(gpu_quota)
    caps = [output_cap(measured, s) for s in segments]
    scratch = [scratch_bytes(measured, s) for s in segments]
    required = plan.EXACT_MEMBER_BYTES + max(scratch) + t4.SCRATCH_RESERVE_BYTES
    return {
        "updates_per_epoch": plan.UPDATES_PER_EPOCH,
        "total_updates_per_arm": plan.TOTAL_UPDATES,
        "arms": list(ARMS),
        "segments_per_arm": segments,
        "sessions": len(segments),
        "session_timeout_seconds": timeout,
        "projected_session_seconds": [seconds_for(b - a) for a, b in segments],
        "session_charge_seconds": charge,
        "conservative_charge_seconds": len(segments) * charge,
        "available_seconds": available,
        "next_session_fits_now": charge <= available,
        "quota_fits_now": len(segments) * charge <= available,
        "scratch_bytes_required": required,
        "scratch_fits": free_scratch_bytes >= required,
        "run_scratch_bytes_per_segment": scratch,
        "run_scratch_fits": all(s <= SCRATCH_CAP_BYTES for s in scratch),
        "output_cap_bytes_per_segment": caps,
        "output_caps_fit": all(c <= plan.OUTPUT_CAP_CEILING_BYTES for c in caps),
        "concurrent_arms": True,
        "provisioned_GPUs_charged_per_session": plan.PROVISIONED_T4,
        "only_final_epoch10_evaluated": True,
        "scientific_hyperparameters_changed": False,
        "intermediate_epoch_selection": False,
    }


def validate_segment(contract: dict) -> dict:
    """Exact paired boundary shared by both arms; validated before any GPU work."""
    if not isinstance(contract, dict) or set(contract) != SEGMENT_KEYS:
        raise ValueError("Exact paired segment contract required")
    start, end, previous = contract["start_step"], contract["end_step"], contract["previous_output"]
    if (
        any(type(v) is not int for v in (start, end, contract["segment"]))
        or not 0 <= start < end <= plan.TOTAL_UPDATES
        or (start == 0) != (contract["segment"] == 0)
        or (start == 0) != (previous is None)
    ):
        raise ValueError("Paired segment boundary differs from the fixed schedule")
    configs = contract["production_config_sha256"]
    if (
        not isinstance(configs, dict)
        or set(configs) != set(ARMS)
        or not all(map(t4._hex64, configs.values()))
        or configs["mean"] == configs["query"]
        or not t4._hex64(contract["engineering_receipt_sha256"])
        or not t4._hex64(contract["engineering_package_manifest_sha256"])
        or type(contract["output_cap_bytes"]) is not int
        or not 0 < contract["output_cap_bytes"] <= plan.OUTPUT_CAP_CEILING_BYTES
    ):
        raise ValueError("Engineering receipt/package, both configs and output cap required")
    if previous is not None and (
        not isinstance(previous, dict)
        or set(previous) != PREVIOUS_KEYS
        or not isinstance(previous["kernel_slug"], str)
        or previous["kernel_slug"].count("/") != 1
        or previous["version"] != 1
        or not t4._hex64(previous["source_sha256"])
        or not t4._hex64(previous["receipt_sha256"])
        or any(
            not isinstance(previous[k], dict)
            or set(previous[k]) != set(ARMS)
            or not all(map(t4._hex64, previous[k].values()))
            for k in ("publication_sha256", "checkpoint_sha256")
        )
    ):
        raise ValueError("Exact shared previous kernel/version/receipt/both-arm outputs required")
    return contract


def check_previous_receipt(receipt: dict, contract: dict) -> None:
    """The prior paired receipt is exactly the scheduled predecessor of ``contract``."""
    prior = receipt.get("contract") if isinstance(receipt, dict) else None
    arms = receipt.get("arms") if isinstance(receipt, dict) else None
    if (
        not isinstance(prior, dict)
        or not isinstance(arms, dict)
        or receipt.get("status") != PAIRED_STATUS
        or set(arms) != set(ARMS)
        or receipt.get("final_epoch10_evaluated") is not False
        or receipt.get("intermediate_selection") is not False
        or (prior.get("segment", -2) + 1, prior.get("end_step"))
        != (
            contract["segment"],
            contract["start_step"],
        )
        or any(prior.get(k) != contract[k] for k in BINDING_KEYS)
    ):
        raise ValueError("Previous paired receipt differs from the schedule/bindings")
    for arm in ARMS:
        record = arms[arm]
        if (
            record["publication"]["checkpoint_sha256"] != record["checkpoint_sha256"]
            or record.get("prediction_sha256") is not None
        ):
            raise ValueError("Previous paired arm output differs from its own publication")


def next_segment(
    previous_receipt: dict | None,
    segments: list,
    *,
    engineering_receipt_sha256: str,
    engineering_package_manifest_sha256: str,
    production_config_sha256: dict,
    measured: dict,
    previous_kernel: dict | None = None,
) -> dict:
    """Bind the next paired session to the exact prior paired kernel output, or start."""
    common = {
        "engineering_receipt_sha256": engineering_receipt_sha256,
        "engineering_package_manifest_sha256": engineering_package_manifest_sha256,
        "production_config_sha256": dict(production_config_sha256),
    }
    if previous_receipt is None:
        if previous_kernel is not None:
            raise ValueError("The first paired segment has no previous kernel")
        start, end = segments[0]
        return validate_segment(
            {
                "segment": 0,
                "start_step": start,
                "end_step": end,
                "previous_output": None,
                **common,
                "output_cap_bytes": output_cap(measured, (start, end)),
            }
        )
    prior = previous_receipt["contract"]
    index = prior["segment"] + 1
    if index >= len(segments) or tuple(segments[prior["segment"]]) != (
        prior["start_step"],
        prior["end_step"],
    ):
        raise ValueError("Previous paired segment differs from the schedule")
    start, end = segments[index]
    contract = {"segment": index, "start_step": start, "end_step": end, **common}
    check_previous_receipt(previous_receipt, contract)
    if (
        not isinstance(previous_kernel, dict)
        or set(previous_kernel) != {"kernel_slug", "version", "source_sha256"}
        or previous_kernel["version"] != 1
    ):
        raise ValueError("Exact previous kernel slug/version/source identity required")
    arms = previous_receipt["arms"]
    contract["previous_output"] = {
        **previous_kernel,
        "receipt_sha256": t4.canonical(previous_receipt),
        "publication_sha256": {a: arms[a]["publication"]["manifest_sha256"] for a in ARMS},
        "checkpoint_sha256": {a: arms[a]["checkpoint_sha256"] for a in ARMS},
    }
    contract["output_cap_bytes"] = output_cap(measured, (start, end))
    return validate_segment(contract)


# --- Inside the notebook -----------------------------------------------------------------


def _verify_publication(target: Path, publication_sha256: str, checkpoint_sha256: str) -> dict:
    path = target / t4.PUBLICATION_MANIFEST
    if path.is_symlink() or not path.is_file() or t4.sha(path) != publication_sha256:
        raise ValueError("Mounted previous arm publication differs from its bound manifest")
    manifest = json.loads(path.read_bytes())
    for name, record in manifest["files"].items():
        relative = Path(name)
        member = target / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or member.is_symlink()
            or not member.is_file()
            or member.stat().st_size != record["bytes"]
            or t4.sha(member) != record["sha256"]
        ):
            raise ValueError("Mounted previous arm member differs from its manifest")
    if (
        manifest["checkpoint_sha256"] != checkpoint_sha256
        or manifest["files"]["run/checkpoint.pt"]["sha256"] != checkpoint_sha256
    ):
        raise ValueError("Mounted continuation checkpoint differs from the bound boundary")
    return manifest


def restore_previous(input_root: Path, contract: dict, run_dirs: dict[str, Path]) -> dict:
    """Stage both arms' continuation checkpoints from the ONE bound previous paired output."""
    previous = contract["previous_output"]
    found = []

    def walk(directory: Path, depth: int) -> None:
        for entry in os.scandir(directory):
            if entry.is_symlink() or entry.name == "rsna-knee-abnormality-detection":
                continue
            if entry.is_file() and entry.name == PAIRED_RECEIPT:
                found.append(Path(entry.path))
            elif entry.is_dir() and depth > 0:
                walk(Path(entry.path), depth - 1)

    walk(Path(input_root), t4.SEARCH_DEPTH)
    matches = [
        p for p in found if t4.canonical(json.loads(p.read_bytes())) == previous["receipt_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("Exactly one bound previous paired receipt must be mounted")
    output = matches[0].parent
    receipt = json.loads(matches[0].read_bytes())
    check_previous_receipt(receipt, contract)
    staged = {}
    for arm in ARMS:
        record = receipt["arms"][arm]
        if (
            record["publication"]["manifest_sha256"] != previous["publication_sha256"][arm]
            or record["checkpoint_sha256"] != previous["checkpoint_sha256"][arm]
        ):
            raise ValueError("Previous paired receipt differs from the bound arm outputs")
        target = output / PAIRED_DIR / arm / t4.PUBLICATION_DIR
        manifest = _verify_publication(
            target, previous["publication_sha256"][arm], previous["checkpoint_sha256"][arm]
        )
        run_dirs[arm].mkdir(mode=0o700, parents=True, exist_ok=True)
        copy = run_dirs[arm] / "checkpoint.pt"
        t4._write_once(target / "run/checkpoint.pt", copy)
        copy.chmod(0o600)
        if t4.sha(copy) != previous["checkpoint_sha256"][arm]:
            raise ValueError("Staged checkpoint copy differs")
        staged[arm] = {
            "publication_sha256": previous["publication_sha256"][arm],
            "checkpoint_sha256": previous["checkpoint_sha256"][arm],
            "files": len(manifest["files"]),
        }
    return staged


def publish(arms: dict[str, tuple[Path, Path]], working: Path, cap: int, final: bool) -> dict:
    """Both arms' write-once publications; the combined cap is checked before any write."""
    files = {arm: t4.publication_files(root, run, final) for arm, (root, run) in arms.items()}
    total = sum(p.stat().st_size for members in files.values() for p in members.values())
    if set(arms) != set(ARMS) or total > cap:
        raise ValueError("Paired publication exceeds its explicit combined output cap")
    working.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(working).free < total + len(ARMS) * t4.PUBLICATION_RESERVE_BYTES:
        raise ValueError("Notebook output storage cannot hold both publications plus reserve")
    (working / PAIRED_DIR).mkdir(mode=0o700)
    return {
        arm: t4.publish_segment(root, run, working / PAIRED_DIR / arm, cap, final)
        for arm, (root, run) in arms.items()
    }


def continue_arms(
    contract: dict,
    arms: dict,
    supervisor,
    working: Path,
    hashes,
    cache=None,
    total_updates: int = plan.TOTAL_UPDATES,
):
    """Run both arms concurrently to the shared boundary, verify, then publish both.

    ``arms[arm]`` has ``root``, ``run_dir``, ``command`` (complete, including resume/stop)
    and ``env``. Nothing reaches ``working`` unless both arms reached the exact boundary.
    """
    import torch

    final = contract["end_step"] == total_updates
    expected = 0 if final else 75
    frozen = None if cache is None else t4.cache_digest(cache)
    timings: dict = {arm: {} for arm in ARMS}

    def job(arm):
        spec = arms[arm]
        return lambda: supervisor.run(
            arm, spec["command"], spec["root"], "segment", expected, spec["env"], timings[arm]
        )

    t4.run_concurrent({arm: job(arm) for arm in ARMS}, supervisor)
    if cache is not None and t4.cache_digest(cache) != frozen:
        raise ValueError("Shared admitted cache changed during the paired session")
    for arm in ARMS:
        run_dir = arms[arm]["run_dir"]
        state = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["step"] != contract["end_step"] or state["binding"]["code_sha256"] != hashes:
            raise ValueError("An arm did not stop at the shared scheduled boundary/source")
        del state
        if final != (run_dir / "outer_predictions.csv").is_file():
            raise ValueError("Only the final paired session evaluates the 1156 studies")
    publications = publish(
        {arm: (arms[arm]["root"], arms[arm]["run_dir"]) for arm in ARMS},
        working,
        contract["output_cap_bytes"],
        final,
    )
    results = {}
    for arm in ARMS:
        run_dir = arms[arm]["run_dir"]
        results[arm] = {
            "checkpoint_sha256": publications[arm]["checkpoint_sha256"],
            "publication": publications[arm],
            "epoch_snapshots": {
                p.name: t4.sha(p) for p in sorted((run_dir / "epoch-checkpoints").glob("epoch-*"))
            },
            "prediction_sha256": t4.sha(run_dir / "outer_predictions.csv") if final else None,
            "seconds": timings[arm],
        }
    return results


def write_receipt(working: Path, receipt: dict) -> None:
    with (working / PAIRED_RECEIPT).open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")


def segment_extra(contract: dict, run_dir: Path) -> list[str]:
    extra = ["--output", str(run_dir)]
    if contract["previous_output"] is not None:
        extra += ["--resume", str(run_dir / "checkpoint.pt")]
    if contract["end_step"] != plan.TOTAL_UPDATES:
        extra += ["--stop-after-updates", str(contract["end_step"] - contract["start_step"])]
    return extra


def run_paired_segment(args) -> dict:
    """One bounded paired production session on both provisioned T4s."""
    started = time.monotonic()
    deadline = started + min(args.remaining_seconds, t4.SESSION_SECONDS)
    contract = validate_segment(json.loads(Path(args.segment).read_text()))
    manifest = json.loads(Path(args.source_manifest).read_text())
    if engineering_manifest_sha256(manifest) != contract["engineering_package_manifest_sha256"]:
        raise ValueError("Production package differs from the accepted engineering package")
    observed = t4.check_runtime(engineering=False, visible=len(t4.PARALLEL_DEVICES))
    code, hashes, fixture, bag, configs, spec, policy, shards, manifests, index, paths, data = (
        t4.prepare(args)
    )
    fit, outer, _, _, _ = data
    values = {arm: t4.production_config(configs[arm], spec) for arm in ARMS}
    if {arm: t4.canonical(values[arm]) for arm in ARMS} != contract["production_config_sha256"]:
        raise ValueError("Production configs differ from the engineering-admitted configs")
    out = Path(args.output)
    out.mkdir(mode=0o700)
    scratch = out / "arms"
    scratch.mkdir(mode=0o700)
    roots = {arm: scratch / arm for arm in ARMS}
    for root in roots.values():
        root.mkdir(mode=0o700)
    run_dirs = {arm: roots[arm] / "run" for arm in ARMS}
    previous = None
    if contract["previous_output"] is not None:
        previous = restore_previous(Path(args.input_root), contract, run_dirs)
    for run_dir in run_dirs.values():
        run_dir.mkdir(mode=0o700, exist_ok=True)
    cache = out / "cache"
    restored = t4.restore(cache, shards, manifests, index, [*fit.study_uid, *outer.study_uid])
    frozen = t4.freeze_cache(cache)
    arms = {}
    for arm in ARMS:
        config = roots[arm] / "config.json"
        config.write_text(json.dumps(values[arm], sort_keys=True, indent=2) + "\n")
        arms[arm] = {
            "root": roots[arm],
            "run_dir": run_dirs[arm],
            "command": t4.trainer_command(code, config, paths, args.overlay, cache)
            + segment_extra(contract, run_dirs[arm]),
            "env": t4.trainer_env(code, t4.PARALLEL_DEVICES[arm]),
        }
    supervisor = t4.Supervisor(
        scratch,
        deadline,
        {arm: scratch / f"{arm}-journal.jsonl" for arm in ARMS},
        output_cap=SCRATCH_CAP_BYTES,
    )
    working = Path(args.publish_root)
    results = continue_arms(contract, arms, supervisor, working, hashes, cache)
    final = contract["end_step"] == plan.TOTAL_UPDATES
    receipt = {
        "status": PAIRED_STATUS,
        "contract": contract,
        "science_git": t4.SCIENCE_GIT,
        "package_manifest_sha256": t4.sha(args.source_manifest),
        "index_sha256": t4.INDEX_SHA256,
        "restored": restored,
        "shared_read_only_cache_sha256": frozen,
        "previous_restored": previous,
        "arms": results,
        "devices": t4.PARALLEL_DEVICES,
        "peak_combined_rss_bytes": supervisor.peak_rss,
        "peak_combined_run_scratch_bytes": supervisor.peak_output,
        "journal_sha256": {arm: t4.sha(scratch / f"{arm}-journal.jsonl") for arm in ARMS},
        "final_epoch10_evaluated": final,
        "elapsed_seconds": time.monotonic() - started,
        "resources_before": observed,
        "resources_after": t4.check_runtime(engineering=False, visible=len(t4.PARALLEL_DEVICES)),
        "intermediate_selection": False,
        "gold_or_confirmation_used": False,
        "competition_submission": False,
    }
    write_receipt(working, receipt)
    print(json.dumps({"status": PAIRED_STATUS, "end_step": contract["end_step"]}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "source-manifest",
        "fixture",
        "bag-cpu",
        "encoder-spec",
        "overlay",
        "pixel-group",
        "studies",
        "artifact-root",
        "folds",
        "series",
        "output",
        "segment",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--publish-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--remaining-seconds", type=float, required=True)
    parser.add_argument("--bootstrap-seconds", type=float, required=True)
    run_paired_segment(parser.parse_args())


if __name__ == "__main__":
    main()
