"""Paired two-T4 fixed-ten-epoch production continuation for the whole-window bag (issue #55).

PREPARED ONLY; root alone reviews, pushes and dispatches. One Kaggle "T4 x2" session runs
both original arms concurrently: the plain-mean arm in one fresh canonical trainer process on
``CUDA_VISIBLE_DEVICES=0`` and the disease-query arm in another on ``1``. Both read one
shared admitted cache, which is restored once from the 20 mounted original shards and then
made read-only. Each arm resumes only its own exact continuation checkpoint. Both
continuations come from the single mounted output of the bound previous paired session
(20 shards + 1 previous kernel, within the actual 21/3 input shape). Each arm stops at the
same scheduled optimizer boundary. Its continuation checkpoint and every newly completed
immutable epoch are published write-once under ``/kaggle/working``. Only the final segment
reaches epoch 10, the trainer's only 1156-study outer evaluation.

The numerical package and both scientific recipes are unchanged. ``window_bag_t4.py`` and
``window_bag_plan.py`` are bound byte-for-byte to the actual accepted concurrent engineering
package. This file is the only additional bound current source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

PAIRED_SOURCE = "scripts/kaggle/window_bag_paired.py"
PAIRED_SEGMENT_STATUS = "source_window_bag_fixed10_paired_two_T4_segment_completed"
PAIRED_DIR = "window-bag-paired"
PAIRED_RECEIPT = "window-bag-paired-segment-accepted.json"
ARM_PREVIOUS_KEYS = {"publication_sha256", "checkpoint_sha256"}
PREVIOUS_KEYS = {"kernel_slug", "version", "source_sha256", "receipt_sha256", "arms"}
PAIRED_KEYS = {
    "segment",
    "start_step",
    "end_step",
    "previous_output",
    "engineering_receipt_sha256",
    "engineering_package_manifest_sha256",
    "production_config_sha256",
    "arm_output_cap_bytes",
    "output_cap_bytes",
    "scratch_cap_bytes",
}


# --- Package identity -------------------------------------------------------------------


def engineering_manifest_text(manifest: dict) -> str:
    """Exact engineering package manifest text this paired package extends by one file."""
    bound = dict(manifest.get("bound_current", {}))
    if set(bound) != {*t4_bound_names(), PAIRED_SOURCE}:
        raise ValueError("Paired package must bind exactly the engineering files plus itself")
    del bound[PAIRED_SOURCE]
    engineering = {
        "science_git": manifest["science_git"],
        "files": manifest["files"],
        "bound_current": bound,
    }
    return json.dumps(engineering, sort_keys=True, indent=2) + "\n"


def t4_bound_names() -> tuple[str, ...]:
    return ("scripts/kaggle/window_bag_t4.py", "scripts/kaggle/window_bag_plan.py")


def verify_package_link(manifest: dict, expected_engineering_sha256: str) -> str:
    """The paired package's engineering part hashes to the accepted engineering manifest."""
    import hashlib

    digest = hashlib.sha256(engineering_manifest_text(manifest).encode()).hexdigest()
    if (
        manifest.get("engineering_package_manifest_sha256") != digest
        or digest != expected_engineering_sha256
    ):
        raise ValueError("Paired package differs from the accepted engineering package")
    return digest


# --- Plan and contracts ---------------------------------------------------------------


def scratch_cap_bytes(measured: dict, segment) -> int:
    """Both arms' run directories: published state plus one transient checkpoint rewrite."""
    return 2 * (plan.publication_bytes(measured, tuple(segment)) + measured["checkpoint_bytes"])


def paired_plan(measured: dict, gpu_quota: dict, free_scratch_bytes: int) -> dict:
    """Fixed 2810 updates per arm, both arms in every session; both T4s charged once each.

    The per-session wall time uses the slowest arm's actual concurrent seconds per update,
    startup and final evaluation, as measured by ``t4.measure`` in the concurrent
    engineering session. The recipe, batch and segment boundaries never change.
    """
    sized = plan.production_plan(measured, gpu_quota, free_scratch_bytes, concurrent=True)
    segments = sized["segments_per_arm"]
    per_arm = [plan.publication_bytes(measured, s) for s in segments]
    if sized["output_cap_bytes_per_segment"] != [2 * b for b in per_arm] or sized[
        "sessions"
    ] != len(segments):
        raise ValueError("Paired plan must hold one session per shared segment")
    timeout = sized["session_timeout_seconds"]
    return {
        **sized,
        "paired_segment_runner": True,
        "arm_output_cap_bytes_per_segment": per_arm,
        "scratch_cap_bytes_per_segment": [scratch_cap_bytes(measured, s) for s in segments],
        "session_charge_seconds": plan.session_charge(timeout),
        "session_within_9h_margin": timeout <= plan.SESSION_LIMIT_SECONDS * plan.SESSION_MARGIN,
        "next_session_quota_fits_now": plan.session_charge(timeout) <= sized["available_seconds"],
    }


def paired_contract(contract: dict) -> dict:
    """Exact scheduled boundary of one shared two-arm session; validated before any GPU work."""
    if not isinstance(contract, dict) or set(contract) != PAIRED_KEYS:
        raise ValueError("Exact paired segment contract required")
    start, end, previous = contract["start_step"], contract["end_step"], contract["previous_output"]
    if (
        any(type(v) is not int for v in (start, end, contract["segment"]))
        or not 0 <= start < end <= plan.TOTAL_UPDATES
        or (start == 0) != (contract["segment"] == 0)
        or (start == 0) != (previous is None)
    ):
        raise ValueError("Paired segment boundary differs from the fixed schedule")
    if previous is not None and (
        not isinstance(previous, dict)
        or set(previous) != PREVIOUS_KEYS
        or not isinstance(previous["kernel_slug"], str)
        or previous["kernel_slug"].count("/") != 1
        or previous["version"] != 1
        or not t4._hex64(previous["source_sha256"])
        or not t4._hex64(previous["receipt_sha256"])
        or not isinstance(previous["arms"], dict)
        or set(previous["arms"]) != set(t4.ARMS)
        or any(
            not isinstance(v, dict)
            or set(v) != ARM_PREVIOUS_KEYS
            or not all(t4._hex64(x) for x in v.values())
            for v in previous["arms"].values()
        )
    ):
        raise ValueError("One exact previous paired kernel output with both arms required")
    configs = contract["production_config_sha256"]
    cap, total = contract["arm_output_cap_bytes"], contract["output_cap_bytes"]
    scratch = contract["scratch_cap_bytes"]
    if (
        not t4._hex64(contract["engineering_receipt_sha256"])
        or not t4._hex64(contract["engineering_package_manifest_sha256"])
        or not isinstance(configs, dict)
        or set(configs) != set(t4.ARMS)
        or not all(t4._hex64(v) for v in configs.values())
        or configs["mean"] == configs["query"]
        or type(cap) is not int
        or type(total) is not int
        or type(scratch) is not int
        or not 0 < cap
        or scratch <= total
        or total != 2 * cap
        or total > plan.OUTPUT_CAP_CEILING_BYTES
    ):
        raise ValueError("Engineering, per-arm config and combined output cap bindings required")
    return contract


def next_paired_segment(
    previous_receipt: dict | None,
    segments: list,
    *,
    engineering_receipt_sha256: str,
    engineering_package_manifest_sha256: str,
    production_config_sha256: dict,
    measured: dict,
    previous_kernel: dict | None = None,
) -> dict:
    """Chain the next shared session to the exact single previous paired kernel output."""
    common = {
        "engineering_receipt_sha256": engineering_receipt_sha256,
        "engineering_package_manifest_sha256": engineering_package_manifest_sha256,
        "production_config_sha256": dict(production_config_sha256),
    }
    if previous_receipt is None:
        if previous_kernel is not None:
            raise ValueError("The first paired segment has no previous kernel")
        index, previous = 0, None
    else:
        contract = previous_receipt["contract"]
        index = contract["segment"] + 1
        arms = previous_receipt.get("arms", {})
        if (
            previous_receipt.get("status") != PAIRED_SEGMENT_STATUS
            or index >= len(segments)
            or list(segments[contract["segment"]]) != [contract["start_step"], contract["end_step"]]
            or previous_receipt.get("final_epoch10_evaluated") is not False
            or previous_receipt.get("intermediate_selection") is not False
            or {k: contract[k] for k in common} != common
            or set(arms) != set(t4.ARMS)
            or any(
                a["publication"]["checkpoint_sha256"] != a["checkpoint_sha256"]
                or a.get("step") != contract["end_step"]
                for a in arms.values()
            )
        ):
            raise ValueError("Previous paired receipt differs from the schedule")
        if (
            not isinstance(previous_kernel, dict)
            or set(previous_kernel) != {"kernel_slug", "version", "source_sha256"}
            or previous_kernel["version"] != 1
        ):
            raise ValueError("Exact previous kernel slug/version/source identity required")
        if segments[index][0] != contract["end_step"]:
            raise ValueError("Paired segments must be contiguous")
        previous = {
            **previous_kernel,
            "receipt_sha256": t4.canonical(previous_receipt),
            "arms": {
                arm: {
                    "publication_sha256": arms[arm]["publication"]["manifest_sha256"],
                    "checkpoint_sha256": arms[arm]["checkpoint_sha256"],
                }
                for arm in t4.ARMS
            },
        }
    start, end = segments[index]
    cap = plan.publication_bytes(measured, (start, end))
    return paired_contract(
        {
            "segment": index,
            "start_step": start,
            "end_step": end,
            "previous_output": previous,
            **common,
            "arm_output_cap_bytes": cap,
            "output_cap_bytes": 2 * cap,
            "scratch_cap_bytes": scratch_cap_bytes(measured, (start, end)),
        }
    )


# --- Session-boundary publication and restore -------------------------------------------


def publish_paired(roots: dict[str, Path], working: Path, contract: dict, final: bool) -> dict:
    """Both arms' write-once outputs in this one session output, within the combined cap."""
    sizes = {
        arm: sum(
            p.stat().st_size
            for p in t4.publication_files(roots[arm], roots[arm] / "run", final).values()
        )
        for arm in t4.ARMS
    }
    total = sum(sizes.values())
    if (
        total > contract["output_cap_bytes"]
        or max(sizes.values()) > contract["arm_output_cap_bytes"]
    ):
        raise ValueError("Paired publication exceeds its explicit output cap")
    working.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(working).free < total + 2 * t4.PUBLICATION_RESERVE_BYTES:
        raise ValueError("Notebook output storage cannot hold both arms plus reserve")
    (working / PAIRED_DIR).mkdir(mode=0o700)
    arms = {
        arm: t4.publish_segment(
            roots[arm],
            roots[arm] / "run",
            working / PAIRED_DIR / arm,
            contract["arm_output_cap_bytes"],
            final,
        )
        for arm in t4.ARMS
    }
    if sum(a["bytes"] for a in arms.values()) != total:
        raise ValueError("Published bytes changed during publication")
    return {"arms": arms, "bytes": total, "cap_bytes": contract["output_cap_bytes"]}


def write_receipt(working: Path, result: dict) -> str:
    with (working / PAIRED_RECEIPT).open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    (working / PAIRED_RECEIPT).chmod(0o444)
    return t4.canonical(result)


def _receipts(input_root: Path, expected: str) -> list[Path]:
    found = []

    def walk(directory: Path, depth: int) -> None:
        for entry in os.scandir(directory):
            if entry.is_symlink() or entry.name == "rsna-knee-abnormality-detection":
                continue
            if entry.is_file() and entry.name == PAIRED_RECEIPT:
                if t4.canonical(json.loads(Path(entry.path).read_bytes())) == expected:
                    found.append(Path(entry.path))
            elif entry.is_dir() and depth > 0:
                walk(Path(entry.path), depth - 1)

    walk(Path(input_root), t4.SEARCH_DEPTH)
    return found


def check_previous_receipt(receipt: dict, contract: dict) -> None:
    """The prior paired receipt is the contiguous non-evaluated session this one continues."""
    previous, prior = contract["previous_output"], receipt.get("contract", {})
    arms = receipt.get("arms", {})
    if (
        receipt.get("status") != PAIRED_SEGMENT_STATUS
        or receipt.get("final_epoch10_evaluated") is not False
        or receipt.get("intermediate_selection") is not False
        or (prior.get("segment", -2) + 1, prior.get("end_step"))
        != (contract["segment"], contract["start_step"])
        or any(
            prior.get(k) != contract[k]
            for k in (
                "engineering_receipt_sha256",
                "engineering_package_manifest_sha256",
                "production_config_sha256",
            )
        )
        or set(arms) != set(t4.ARMS)
        or any(
            arms[arm]["publication"]["manifest_sha256"]
            != previous["arms"][arm]["publication_sha256"]
            or arms[arm]["checkpoint_sha256"] != previous["arms"][arm]["checkpoint_sha256"]
            or arms[arm]["publication"]["checkpoint_sha256"] != arms[arm]["checkpoint_sha256"]
            or arms[arm].get("step") != contract["start_step"]
            for arm in t4.ARMS
        )
    ):
        raise ValueError("Previous paired receipt differs from the bound identity")


def restore_previous_paired(input_root: Path, contract: dict, run_dirs: dict[str, Path]) -> dict:
    """Stage each arm's exact continuation from the one mounted previous paired output."""
    previous = contract["previous_output"]
    found = _receipts(input_root, previous["receipt_sha256"])
    if len(found) != 1:
        raise ValueError("Exactly one bound previous paired output must be mounted")
    receipt = json.loads(found[0].read_bytes())
    check_previous_receipt(receipt, contract)
    restored = {}
    for arm in t4.ARMS:
        bound = previous["arms"][arm]
        target = found[0].parent / PAIRED_DIR / arm / t4.PUBLICATION_DIR
        manifest_path = target / t4.PUBLICATION_MANIFEST
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or t4.sha(manifest_path) != bound["publication_sha256"]
        ):
            raise ValueError("Mounted arm publication differs from the bound manifest")
        manifest = json.loads(manifest_path.read_bytes())
        for name, record in manifest["files"].items():
            relative = Path(name)
            path = target / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record["bytes"]
                or t4.sha(path) != record["sha256"]
            ):
                raise ValueError("Mounted arm member differs from its manifest")
        if manifest["checkpoint_sha256"] != bound["checkpoint_sha256"]:
            raise ValueError("Mounted arm continuation differs from the bound boundary")
        run_dirs[arm].mkdir(mode=0o700, parents=True, exist_ok=True)
        staged = run_dirs[arm] / "checkpoint.pt"
        t4._write_once(target / "run/checkpoint.pt", staged)
        staged.chmod(0o600)
        if t4.sha(staged) != bound["checkpoint_sha256"]:
            raise ValueError("Staged arm checkpoint copy differs")
        restored[arm] = {
            "publication_sha256": bound["publication_sha256"],
            "checkpoint_sha256": bound["checkpoint_sha256"],
            "files": len(manifest["files"]),
        }
    return {"receipt_sha256": previous["receipt_sha256"], "arms": restored}


# --- In-session runner -------------------------------------------------------------------


def run_arms(supervisor: t4.Supervisor, jobs: dict) -> dict:
    """One fresh trainer process per arm/T4; the first failure stops and reaps the sibling.

    ``jobs[arm] = (command, root, expected_returncode, env)``.
    """

    def job(arm):
        command, root, expected, env = jobs[arm]

        def run():
            timings: dict = {}
            supervisor.run(arm, command, root, "segment", expected, env, timings)
            return timings

        return run

    return t4.run_concurrent({arm: job(arm) for arm in t4.ARMS}, supervisor)


def run_paired_segment(args) -> dict:
    """One bounded shared session of both arms; the final segment alone evaluates epoch 10."""
    import torch

    started = time.monotonic()
    deadline = started + min(args.remaining_seconds, t4.SESSION_SECONDS)
    contract = paired_contract(json.loads(Path(args.segment).read_text()))
    torch.set_num_threads(2)
    observed = t4.check_runtime(engineering=False, visible=len(t4.PARALLEL_DEVICES))
    manifest = json.loads(Path(args.source_manifest).read_text())
    verify_package_link(manifest, contract["engineering_package_manifest_sha256"])
    code, hashes, fixture, bag, configs, spec, policy, shards, manifests, index, paths, data = (
        t4.prepare(args)
    )
    fit, outer, _, _, _ = data
    values = {arm: t4.production_config(configs[arm], spec) for arm in t4.ARMS}
    if {arm: t4.canonical(values[arm]) for arm in t4.ARMS} != contract["production_config_sha256"]:
        raise ValueError("Production configs differ from the engineering-admitted configs")
    out = Path(args.output)
    out.mkdir(mode=0o700)
    arms_root = out / "arms"
    arms_root.mkdir(mode=0o700)
    roots = {arm: arms_root / arm for arm in t4.ARMS}
    for root in roots.values():
        (root / "run").mkdir(mode=0o700, parents=True)
    previous = None
    if contract["previous_output"] is not None:
        previous = restore_previous_paired(
            Path(args.input_root), contract, {arm: roots[arm] / "run" for arm in t4.ARMS}
        )
    cache = out / "cache"
    restored = t4.restore(cache, shards, manifests, index, [*fit.study_uid, *outer.study_uid])
    frozen = t4.freeze_cache(cache)
    final = contract["end_step"] == plan.TOTAL_UPDATES
    jobs = {}
    for arm in t4.ARMS:
        config = roots[arm] / "config.json"
        config.write_text(json.dumps(values[arm], sort_keys=True, indent=2) + "\n")
        extra = ["--output", str(roots[arm] / "run")]
        if previous is not None:
            extra += ["--resume", str(roots[arm] / "run/checkpoint.pt")]
        if not final:
            extra += ["--stop-after-updates", str(contract["end_step"] - contract["start_step"])]
        command = t4.trainer_command(code, config, paths, args.overlay, cache) + extra
        jobs[arm] = (
            command,
            roots[arm],
            0 if final else 75,
            t4.trainer_env(code, t4.PARALLEL_DEVICES[arm]),
        )
    # Both arms' live run directories, including each trainer's transient checkpoint rewrite.
    supervisor = t4.Supervisor(
        arms_root,
        deadline,
        {arm: out / f"{arm}-journal.jsonl" for arm in t4.ARMS},
        output_cap=contract["scratch_cap_bytes"],
    )
    timings = run_arms(supervisor, jobs)
    if t4.cache_digest(cache) != frozen:
        raise ValueError("Shared admitted cache changed during the paired session")
    for arm in t4.ARMS:
        state = torch.load(roots[arm] / "run/checkpoint.pt", map_location="cpu", weights_only=True)
        if state["step"] != contract["end_step"] or state["binding"]["code_sha256"] != hashes:
            raise ValueError("An arm did not stop at its scheduled boundary/source")
        del state
    publication = publish_paired(roots, Path(args.publish_root), contract, final)
    result = {
        "status": PAIRED_SEGMENT_STATUS,
        "contract": contract,
        "science_git": t4.SCIENCE_GIT,
        "package_manifest_sha256": args.package_manifest_sha256,
        "engineering_package_manifest_sha256": contract["engineering_package_manifest_sha256"],
        "index_sha256": t4.INDEX_SHA256,
        "restored": restored,
        "previous_restored": previous,
        "arms": {
            arm: {
                "step": contract["end_step"],
                "device": t4.PARALLEL_DEVICES[arm],
                "checkpoint_sha256": publication["arms"][arm]["checkpoint_sha256"],
                "publication": publication["arms"][arm],
                "epoch_snapshots": {
                    p.name: t4.sha(p)
                    for p in sorted((roots[arm] / "run/epoch-checkpoints").glob("epoch-*.pt"))
                },
                "prediction_sha256": t4.sha(roots[arm] / "run/outer_predictions.csv")
                if final
                else None,
                "seconds": timings[arm],
            }
            for arm in t4.ARMS
        },
        "publication_bytes": publication["bytes"],
        "shared_read_only_cache_sha256": frozen,
        "peak_combined_rss_bytes": supervisor.peak_rss,
        "peak_combined_scratch_bytes": supervisor.peak_output,
        "journal_sha256": {arm: t4.sha(out / f"{arm}-journal.jsonl") for arm in t4.ARMS},
        "final_epoch10_evaluated": final,
        "elapsed_seconds": time.monotonic() - started,
        "resources_before": observed,
        "resources_after": t4.check_runtime(engineering=False, visible=len(t4.PARALLEL_DEVICES)),
        "GPUs_used_concurrently": 2,
        "provisioned_GPUs_charged": plan.PROVISIONED_T4,
        "intermediate_selection": False,
        "competition_submission": False,
    }
    write_receipt(Path(args.publish_root), result)
    print(
        json.dumps({"status": PAIRED_SEGMENT_STATUS, "end_step": contract["end_step"]}), flush=True
    )
    return result


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
