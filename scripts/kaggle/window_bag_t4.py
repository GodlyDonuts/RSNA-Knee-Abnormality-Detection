"""Private offline single-T4 path for the canonical whole-window DINOv3 bag (issue #55).

PREPARED ONLY; root alone reviews, pushes and dispatches. This wrapper runs inside the
notebook with the immutable 971d7dac numerical package on ``PYTHONPATH``. It never renders,
downloads or uploads MRI: it mounts the 20 original completed private CPU source-shard
kernel outputs, verifies every shard manifest, rebuilds the exact trainer index (it must hash
to the original finalized index) and decodes only the members a phase reads.

``engineering``: admits the actual 16 fitting + 16 outer tensors, full rosters, official
initialization and RNG against the original source-bound fixtures on CPU, then runs the
unchanged trainer for both plain-mean/disease-query arms in fresh full/stop1/resume processes
(2 epochs/4 updates) and requires complete state and MRI-only prediction-byte equality.

``segment``: one bounded production session of the fixed 10-epoch recipe. It resumes only
from the exact published output of the bound previous segment kernel (mounted read-only),
stops at a scheduled optimizer boundary and publishes its complete continuation checkpoint,
every newly completed immutable epoch and their sidecars write-once into private
``/kaggle/working``. Only the final segment reaches epoch 10, the only evaluation the
trainer performs.

Engineering metrics never select, score or submit a model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("mean", "query")
WEIGHTING = {"mean": "plain_mean", "query": "disease_query"}
# Original completed private CPU shard kernels (each pushed once, v1) and their exact
# shard-manifest SHA, as verified by the committed per-shard download receipts.
SHARD_MANIFEST_SHA256 = (
    "a41013dbc9dd6ed8b51db47dd4065f42f8cdf09af5029d3159235a482a46c82b",
    "6582b02afb0027cd991e219f9c315969f04938d154c2085a51603b4a460ee51f",
    "bc91729973dde04c2107616d35fa43a58947d8cc61f9973bf6b835e71d662d72",
    "2bde61566c2f4f487744e1dbb4df9b9235344405b5d417ef23cca7841b8a5cd4",
    "fc4a819f05c281da8d0c4cf7e5f2c427c5331fd837c8c8d9d104d389264c027d",
    "3b0a04818493d2390462054bb74b6ed371e81464af255947e492819880907c08",
    "13553dd2536b4707c0cd933a42658114ab742fe10e6a3c9ad23eac06002d9bb4",
    "67511364a3a7b594d57a1d70f62ccea7dd2422011ffc170474bef2485473750c",
    "5e9bcc94cc78e315b5fe63258b419e8a8efbb31fc66a79c249c72903958cc34e",
    "dfa75b182fb9da11bc7ec4c420dae01d56c10722deb409ac8df9459f760f196b",
    "cef9eb2dceb4e0b04b5b864249a495860eb7c881d4bc3c48996f5590dbf19184",
    "d20ecb7cbc27596a2842a3bc041291cfada0532c4e6c4bd894d9b2286b3fde42",
    "bc3fa1556274bd8b0f6375140f770362f4f6f753d1f8ef545b50f95913ec0cb3",
    "73d48cc44d274ef70b5cdccb8d1d5368c0dc6921cc6cce2b163e054075177d6f",
    "a4c78b5d51066ff6aed3c0312e0ef28950fb2720ed7a9904bfbae8e55176d99b",
    "88b06b0ee2f7fe9343de8e595a531f9b45a7fd711c5ed8e1cb73ab1fc394562f",
    "f46223958b23ce694bd17e641e5273b17d3e63a1d199a075b6c7790bffce3330",
    "2343660aa3a81293690bb5040d357d19220e8ff9a93677e42f70522a88649224",
    "97aa32015fb6c76bb4cc513e98c2e197a859ef74edd20aa11de40fe03d69d732",
    "f7f6f93ed0576a7ea47108cc533033dc92bfff23260d16edc42e3856cb769498",
)
SHARD_STUDIES = (175,) * 19 + (78,)
COHORT_MANIFEST_SHA256 = "3f265ddfb5b35857e75745b3476cf7761d6d41a93f3651839272c763ed5e7776"
# Original evc31 finalized complete cache (public-source-complete-cache-20261013-v1).
INDEX_SHA256 = "849855281837669efaad81596314f7770b8a936baa457e2a571418dbd87c1736"
EXACT_MEMBER_BYTES = 27667165884
FITTING_STUDIES, OUTER_STUDIES, ENGINEERING_STUDIES = 2247, 1156, 16
# Original CPU fixtures: public-input recipe admission (16+16 tensors, rosters, cohort) and
# the whole-window common official initialization admission.
FIXTURE_SHA256 = "adb931d591c3cbe4bac79ad8ad1050ece5c9cd2c37dc98e818716e731f050ce2"
FIXTURE_STATUS = (
    "public_input_recipe_native_and_public_triplet_inputs_and_official_initialization_CPU_accepted"
)
BAG_CPU_SHA256 = "7417e3637c4b4cb403a475deba043ce559656beb83454c7540493ffbf88818b9"
BAG_CPU_STATUS = (
    "source_window_bag_official_common_graph_values_RNG_and_original_inputs_CPU_accepted"
)
SCIENCE_GIT = "971d7dac7065681f0600064c9b3ffd959a1e8c45"
OVERLAY_SHA256 = "c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51"
GIB = 2**30
ENGINEERING_SECONDS = 7200
SESSION_SECONDS = 9 * 3600
RAM_CEILING_BYTES = 29 * GIB
CHECKPOINT_CAP_BYTES = 6 * GIB
SCRATCH_RESERVE_BYTES = 3 * GIB
ENGINEERING_STATUS = "source_window_bag_two_arm_actual_single_T4_canonical_recovery_accepted"
SEGMENT_STATUS = "source_window_bag_fixed10_single_T4_segment_completed"
SEARCH_DEPTH = 7
PUBLICATION_DIR = "window-bag-segment"
PUBLICATION_MANIFEST = "segment-publication.json"
SEGMENT_RECEIPT = "window-bag-segment-accepted.json"
ENGINEERING_RECEIPT = "window-bag-engineering-accepted.json"
PUBLICATION_SIDECARS = ("config.json", "segment.stdout", "segment.stderr")
PUBLICATION_RUN_FILES = ("checkpoint.pt", "run.json", "invocation.json")
PUBLICATION_FINAL_FILES = ("result.json", "outer_predictions.csv")
PUBLICATION_RESERVE_BYTES = GIB
MEASURED_KEYS = (
    "seconds_per_update",
    "startup_seconds",
    "checkpoint_seconds",
    "final_evaluation_seconds",
    "checkpoint_bytes",
)


def sha(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_bound(path: Path, expected: str) -> dict:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Bound receipt bytes differ from the original")
    return json.loads(payload)


def validate_fixtures(fixture: dict, bag: dict) -> None:
    """The original fixture and the whole-window CPU admission that reused it."""
    if (
        fixture.get("status") != FIXTURE_STATUS
        or (fixture.get("full_fitting_studies"), fixture.get("full_outer_studies"))
        != (FITTING_STUDIES, OUTER_STUDIES)
        or fixture.get("public_cohort", {}).get("index_sha256") != INDEX_SHA256
        or fixture.get("public_cohort", {}).get("members") != FITTING_STUDIES + OUTER_STUDIES
        or fixture.get("GPU_or_optimizer_or_clinical_inference_executed") is not False
        or fixture.get("protected_labels_used_for_recipe_or_metrics") is not False
        or bag.get("status") != BAG_CPU_STATUS
        or bag.get("source_git") != SCIENCE_GIT
        or bag.get("original_input_receipt_sha256") != FIXTURE_SHA256
        or bag.get("GPU_or_optimizer_executed") is not False
        or bag["initialization"]["mean"] != bag["initialization"]["query"]
        or bag["encoder_provenance"]["mean"] != bag["encoder_provenance"]["query"]
        or bag["encoder_provenance"]["mean"] != fixture["encoder_provenance"]["public"]
        or (bag.get("training_studies"), bag.get("outer_studies"))
        != (FITTING_STUDIES, OUTER_STUDIES)
    ):
        raise ValueError("Original fixture or whole-window CPU admission differs")


def bag_configs(bag: dict) -> dict[str, dict]:
    """Original fixed-10 production configs; only ``window_weighting`` may differ."""
    configs = bag["configs"]
    if set(configs) != set(ARMS):
        raise ValueError("Exactly the original mean/query pair is required")
    differ = {k for k in configs["mean"].keys() | configs["query"].keys()}
    differ = {k for k in differ if configs["mean"].get(k) != configs["query"].get(k)}
    strategies = [configs[a]["classifier_strategy"] for a in ARMS]
    if differ != {"classifier_strategy"} or {
        k for k in strategies[0] if strategies[0][k] != strategies[1][k]
    } != {"window_weighting"}:
        raise ValueError("Arms must differ only in their window weighting")
    for arm in ARMS:
        values = configs[arm]
        if (
            values["classifier_strategy"]["window_weighting"] != WEIGHTING[arm]
            or values["classifier_strategy"]["kind"] != "source_window_bag_v1"
            or values["fitting_supervision"]["sha256"] != OVERLAY_SHA256
            or values["fitting_supervision"]["enabled"] is not True
            or (values["epochs"], values["warmup_updates"]) != (10, 281)
            or (values["batch_size"], values["accumulation_steps"]) != (1, 8)
            or (values["precision"], values["amp_init_scale"], values["encode_chunk_size"])
            != ("fp16", 128, 8)
            or (values["side"], values["slices"]) != (336, 10)
            or values["encoder"]["adapter"]["rank"] != 32
            or values["fit_scope"] != "outer_fold"
            or values["diagnostic_study_limit"] is not None
        ):
            raise ValueError("Original whole-window effective-8 fixed-10 recipe required")
    return configs


def mounted_encoder(base: dict, spec: dict) -> dict:
    """Replace only the encoder file locations with the mounted pinned official base."""
    for key in set(base["encoder"]) | set(spec):
        if key not in {"weights_path", "config_path"} and base["encoder"].get(key) != spec.get(key):
            raise ValueError("Mounted encoder differs from the official pinned base")
    return dict(base, encoder=spec)


def engineering_config(base: dict, spec: dict) -> dict:
    """The canonical diagnostic shortening used by the original H100 recovery."""
    return dict(
        mounted_encoder(base, spec),
        epochs=2,
        diagnostic_study_limit=ENGINEERING_STUDIES,
        warmup_updates=2,
        submission_every_epochs=0,
        min_free_gib=3,
    )


def production_config(base: dict, spec: dict) -> dict:
    """Unchanged scientific recipe; only mounted paths and Kaggle scratch floor differ."""
    return dict(mounted_encoder(base, spec), min_free_gib=3)


def _walk(root: Path, skip: set[str], depth: int = SEARCH_DEPTH):
    for entry in os.scandir(root):
        if entry.is_symlink() or entry.name in skip:
            continue
        if entry.is_file() and entry.name == "shard-manifest.json":
            yield Path(entry.path)
        elif entry.is_dir() and depth > 0:
            yield from _walk(Path(entry.path), skip, depth - 1)


def locate_shards(input_root: Path, skip=("rsna-knee-abnormality-detection",)):
    """Find the 20 mounted original shard outputs by exact manifest SHA, never by name."""
    found: dict[str, list[Path]] = {}
    for path in _walk(Path(input_root), set(skip)):
        found.setdefault(sha(path), []).append(path.parent)
    shards = []
    for index, expected in enumerate(SHARD_MANIFEST_SHA256):
        if len(found.get(expected, [])) != 1:
            raise ValueError("Each original shard output must be mounted exactly once")
        shards.append(found[expected][0])
    cohorts = {(s.parent / "cohort-manifest.json") for s in shards}
    if not all(p.is_file() for p in cohorts):
        raise ValueError("Original shard cohort manifest absent")
    cohort = json.loads(shards[0].parent.joinpath("cohort-manifest.json").read_bytes())
    for path in cohorts:
        if json.loads(path.read_bytes()) != cohort:
            raise ValueError("Mounted shards disagree on the complete cohort")
    return shards, cohort


def verify_shards(shards: list[Path], cohort: dict) -> list[dict]:
    from rsna.competitive import kaggle_source_render as ksr

    if canonical(cohort) != COHORT_MANIFEST_SHA256 or len(shards) != len(cohort["shards"]):
        raise ValueError("Complete-cohort manifest differs from the original")
    manifests, seen = [], set()
    for index, (path, expected) in enumerate(zip(shards, SHARD_MANIFEST_SHA256, strict=True)):
        manifest = ksr.verify_shard(path, cohort, expected)
        uids = {m["study_uid"] for m in manifest["members"]}
        if (
            manifest["shard_index"] != index
            or len(uids) != SHARD_STUDIES[index]
            or uids & seen
            or manifest["labels_stored"] is not False
        ):
            raise ValueError("Original shard membership differs")
        seen |= uids
        manifests.append(manifest)
    if len(seen) != FITTING_STUDIES + OUTER_STUDIES:
        raise ValueError("Mounted shards do not cover the complete cohort")
    return manifests


def reconstruct_index(manifests: list[dict], cohort: dict, policy: dict, pixel_group) -> bytes:
    """Exact bytes of ``public_cache_producer.finalize`` for the original complete cache."""
    from rsna.competitive import public_input as pi

    if pi.canonical_sha256(pixel_group) != policy["pixel_group_sha256"]:
        raise ValueError("Pixel group differs from the bound policy")
    members = {
        record["study_uid"]: dict(record["member"])
        for manifest in manifests
        for record in manifest["members"]
    }
    members = dict(sorted(members.items()))
    if (
        sum(m["bytes"] for m in members.values()) != EXACT_MEMBER_BYTES
        or pi.roster_sha256(members) != cohort["cohort_sha256"]
    ):
        raise ValueError("Shard members differ from the complete cohort bytes")
    index = {
        "schema_version": 1,
        **{k: policy[k] for k in ("kind", "renderer", "source_sha256", "node_text_sha256")},
        "pixel_group": pixel_group,
        "source_revision": cohort["source_revision"],
        "fitting_uids_sha256": cohort["fitting_uids_sha256"],
        "evaluation_uids_sha256": cohort["evaluation_uids_sha256"],
        "cohort_sha256": pi.roster_sha256(members),
        "members": members,
    }
    payload = json.dumps(index, sort_keys=True, indent=1).encode()
    if hashlib.sha256(payload).hexdigest() != INDEX_SHA256:
        raise ValueError("Reconstructed index differs from the original finalized cache")
    return payload


def restore(cache: Path, shards: list[Path], manifests: list[dict], index: bytes, uids) -> dict:
    """Decode only ``uids`` from mounted outputs into a new private trainer cache."""
    from rsna.competitive import kaggle_source_render as ksr
    from rsna.competitive import public_input as pi

    wanted = set(uids)
    records = {
        r["study_uid"]: (path, r)
        for path, manifest in zip(shards, manifests, strict=True)
        for r in manifest["members"]
        if r["study_uid"] in wanted
    }
    if set(records) != wanted:
        raise ValueError("Requested members are absent from the original shards")
    need = sum(r["member"]["bytes"] for _, r in records.values())
    cache.mkdir(mode=0o700)
    if shutil.disk_usage(cache).free < need + SCRATCH_RESERVE_BYTES:
        raise ValueError("Scratch cannot hold the restored members plus reserve")
    root = cache / pi.CACHE_DIRECTORY
    (root / "members").mkdir(mode=0o700, parents=True)
    for uid in sorted(records):
        path, record = records[uid]
        blob = (path / record["encoded"]["path"]).read_bytes()
        if (
            len(blob) != record["encoded"]["bytes"]
            or hashlib.sha256(blob).hexdigest() != record["encoded"]["sha256"]
            or sha(path / ksr.RECEIPTS / f"{uid}.json") != record["receipt_sha256"]
            or sha(path / ksr.ORDERS / f"{uid}.json") != record["order_sha256"]
        ):
            raise ValueError("Mounted shard member bytes differ from the manifest")
        payload = ksr.decode_member(
            blob, size=record["member"]["bytes"], sha256=record["member"]["sha256"]
        )
        with (root / record["member"]["path"]).open("xb") as handle:
            handle.write(payload)
    with (root / "index.json").open("xb") as handle:
        handle.write(index)
    return {"members": len(records), "bytes": need, "index_sha256": INDEX_SHA256}


def development_parts(paths: dict, overlay: Path, policy: dict, fixture: dict):
    """Full release/fold membership and the exact 16+16 engineering rosters."""
    import pandas as pd

    from rsna.competitive.folds import load_folds
    from rsna.competitive.public_input import roster_sha256
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development
    from rsna.release import load_verified_release

    if sha(overlay) != OVERLAY_SHA256:
        raise ValueError("Exact fitting-only overlay required")
    frame, binding = load_verified_release(paths["studies"], paths["artifact-root"])
    folds = load_folds(paths["folds"], frame, binding)
    fitting, outer = select_development(frame, folds, 0, "outer_fold")
    if (len(fitting), len(outer)) != (FITTING_STUDIES, OUTER_STUDIES) or set(
        fitting.leakage_group
    ) & set(outer.leakage_group):
        raise ValueError("Original fitting/evaluation population or leakage boundary changed")
    fitting, supervision = apply_fitting_supervision(
        fitting,
        outer,
        overlay,
        {"enabled": True, "kind": "label_source_v1", "sha256": OVERLAY_SHA256},
        release_sha256=binding["manifest_sha256"],
        folds_sha256=sha(Path(paths["folds"]) / "fold.csv"),
        fold=0,
    )
    cohort = fixture["public_cohort"]
    parts = {
        "fitting": fitting.head(ENGINEERING_STUDIES),
        "development": outer.head(ENGINEERING_STUDIES),
    }
    if (
        roster_sha256(map(str, fitting.study_uid)) != cohort["fitting_uids_sha256"]
        or roster_sha256(map(str, outer.study_uid)) != cohort["evaluation_uids_sha256"]
        or supervision != fixture["supervision"]
        or _uid_hash(parts["fitting"]) != fixture["training_16_sha256"]
        or _uid_hash(parts["development"]) != fixture["outer_16_sha256"]
        or set(parts["fitting"].leakage_group) & set(parts["development"].leakage_group)
    ):
        raise ValueError("Release/fold/overlay membership differs from the original fixture")
    series_bytes = Path(paths["series"]).read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != binding["series_sha256"]:
        raise ValueError("Series sampling table differs from the accepted release")
    import io

    return fitting, outer, parts, pd.read_csv(io.BytesIO(series_bytes)), binding


def admit_tensors(cache, policy, fixture, parts, outer_full, fit_full, series, revision):
    """CPU-only actual input admission before any GPU optimizer update."""
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from scripts.newton.probe_public_input_recipe_inputs import tensor_proofs

    if (
        verify_public_cohort(cache, policy, fit_full, outer_full, revision)
        != fixture["public_cohort"]
    ):
        raise ValueError("Restored cache cohort binding differs from the original fixture")
    proofs = {}
    for role, part in parts.items():
        soft = {"soft_targets": True} if role == "fitting" else {}
        dataset = PublicTripletStudyDataset(
            part,
            series,
            cache,
            public_input=policy,
            source_revision=revision,
            slices=10,
            side=336,
            max_series=6,
            slice_sampling="uniform",
            series_sampling="selected",
            **soft,
        )
        proofs[role], _ = tensor_proofs(part, dataset, "public")
        if proofs[role] != fixture["tensor_sha256"][f"public/{role}"]:
            raise ValueError("Actual engineering tensors differ from the original fixture")
        del dataset
        gc.collect()
    return proofs


def admit_initialization(configs: dict, spec: dict, bag: dict) -> None:
    """Re-observe common official graph/values/RNG from the mounted base on CPU."""
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.window_bag import WindowBagSlotStudyNet
    from scripts.newton.probe_public_input_recipe_inputs import initialize

    for arm in ARMS:
        values = mounted_encoder(configs[arm], spec)

        def factory(encoder, dim, **_):
            return WindowBagSlotStudyNet(
                encoder,
                dim,
                values["classifier_strategy"],
                encode_chunk_size=values["encode_chunk_size"],
                input_channels=3,
            )

        model, provenance, state = initialize(values, "public", create_encoder, factory)
        if (
            state != bag["initialization"][arm]
            or provenance != bag["encoder_provenance"][arm]
            or model.window_query.count_nonzero() != 0
        ):
            raise ValueError("Mounted official initialization/RNG differs from CPU admission")
        del model
        gc.collect()


def check_runtime(*, engineering: bool) -> dict:
    import torch

    from rsna.competitive import kaggle_source_render as ksr

    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Exactly one visible T4 is required")
    observed = ksr.resources([Path("/kaggle/working"), Path("/tmp")])
    limit = observed["cgroup_memory_limit_bytes"] or observed["mem_total_bytes"]
    if limit is None or limit > 64 * GIB:
        raise ValueError("Measured session memory envelope required")
    return observed | {"engineering": engineering, "GPUs_visible": 1}


def trainer_command(code: Path, config: Path, paths: dict, overlay: Path, cache: Path):
    return [
        sys.executable,
        "-B",
        str(code / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--device",
        "cuda",
        "--fitting-supervision",
        str(overlay),
        "--studies",
        str(paths["studies"]),
        "--artifact-root",
        str(paths["artifact-root"]),
        "--folds",
        str(paths["folds"]),
        "--series",
        str(paths["series"]),
        "--cache",
        str(cache),
    ]


def trainer_env(code: Path) -> dict:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(code),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    return env


def output_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def peak_rss(pid_rss: int) -> None:
    if pid_rss > RAM_CEILING_BYTES:
        raise ValueError("Session RSS exceeded the explicit RAM ceiling")


def execute(command, extra, root: Path, phase: str, expected: int, deadline, env, timings, cap):
    """One fresh trainer process; logs are exclusive and retained."""
    import resource

    if output_bytes(cap) > CHECKPOINT_CAP_BYTES:
        raise ValueError("Diagnostic checkpoint/output cap exceeded")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Bounded session deadline exhausted")
    print(json.dumps({"phase": phase, "root": root.name}), flush=True)
    started = time.monotonic()
    with (root / f"{phase}.stdout").open("x") as out, (root / f"{phase}.stderr").open("x") as err:
        result = subprocess.run(
            command + extra,
            env=env,
            cwd=env["PYTHONPATH"],
            stdout=out,
            stderr=err,
            timeout=remaining,
        )
    timings[phase] = time.monotonic() - started
    peak_rss(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
    if output_bytes(cap) > CHECKPOINT_CAP_BYTES or result.returncode != expected:
        raise ValueError("Canonical trainer phase failed or exceeded its cap; logs retained")


def recover_arm(code, out, arm, values, paths, overlay, cache, fixture, hashes, deadline):
    import torch

    from scripts.newton.probe_patch_witness_trainer_recovery import (
        compare_recovery,
        numerical_updates,
    )
    from scripts.newton.probe_public_input_recipe_recovery import (
        validate_predictions,
        validate_state,
    )

    root = out / arm
    root.mkdir(mode=0o700)
    config = root / "config.json"
    config.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")
    command = trainer_command(code, config, paths, overlay, cache)
    env, timings = trainer_env(code), {}
    full, replay = root / "full", root / "replay"

    def run(extra, phase, expected):
        execute(command, extra, root, phase, expected, deadline, env, timings, out)

    run(["--output", str(full)], "full", 0)
    run(["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
    # Keep the actual interrupted checkpoint/history before resume replaces them.
    copy_started = time.monotonic()
    shutil.copyfile(replay / "checkpoint.pt", root / "actual-stop-checkpoint.pt")
    copy_seconds = time.monotonic() - copy_started
    shutil.copyfile(replay / "invocation.json", root / "actual-stop-invocation.json")
    run(["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)  # noqa: E731
    whole, resumed = load(full / "checkpoint.pt"), load(replay / "checkpoint.pt")
    partial = load(root / "actual-stop-checkpoint.pt")
    history = json.loads((full / "invocation.json").read_text())
    first = json.loads((root / "actual-stop-invocation.json").read_text())
    last = json.loads((replay / "invocation.json").read_text())
    state = compare_recovery(
        whole,
        resumed,
        partial,
        history,
        first,
        last,
        (full / "outer_predictions.csv").read_bytes(),
        (replay / "outer_predictions.csv").read_bytes(),
    )
    validate_state(whole, values, "public", fixture, hashes)
    validate_state(resumed, values, "public", fixture, hashes)
    result = {
        "complete_state_sha256": state,
        "checkpoint_sha256": sha(full / "checkpoint.pt"),
        "prediction_sha256": validate_predictions(
            full / "outer_predictions.csv", fixture["outer_16_sha256"]
        ),
        "finite_updates": numerical_updates(history),
        "actual_stop_checkpoint_sha256": sha(root / "actual-stop-checkpoint.pt"),
        "amp_scaler": whole["amp_scaler"],
        "phase_seconds": timings,
        "update_elapsed_seconds": [row["elapsed_seconds"] for row in history["updates"]],
        "checkpoint_bytes": (full / "checkpoint.pt").stat().st_size,
        "checkpoint_copy_seconds": copy_seconds,
        "complete_actual_partial_final_RNG_optimizer_scaler_and_CSV_recovery": True,
    }
    del whole, resumed, partial
    gc.collect()
    torch.cuda.empty_cache()
    return result


def code_hashes(code: Path, manifest: dict) -> dict:
    """Immutable numerical package plus explicitly bound current wrapper sources."""
    if manifest.get("science_git") != SCIENCE_GIT or not manifest.get("files"):
        raise ValueError("Exact immutable numerical package manifest required")
    for name, value in {**manifest["files"], **manifest.get("bound_current", {})}.items():
        if sha(code / name) != value:
            raise ValueError("Numerical package or bound wrapper bytes differ")
    return {
        n.removeprefix("rsna/"): h for n, h in manifest["files"].items() if n.startswith("rsna/")
    }


def prepare(args):
    """Shared admission: package, fixtures, mounted shards, index, release membership."""
    code = Path(args.source).resolve()
    hashes = code_hashes(code, json.loads(Path(args.source_manifest).read_text()))
    args.package_manifest_sha256 = sha(args.source_manifest)
    fixture = load_bound(args.fixture, FIXTURE_SHA256)
    bag = load_bound(args.bag_cpu, BAG_CPU_SHA256)
    validate_fixtures(fixture, bag)
    if hashes != bag["code_sha256"]:
        raise ValueError("Numerical package differs from the whole-window CPU admission")
    configs = bag_configs(bag)
    spec = json.loads(Path(args.encoder_spec).read_text())
    policy = configs["mean"]["public_input"]
    shards, cohort = locate_shards(Path(args.input_root))
    manifests = verify_shards(shards, cohort)
    index = reconstruct_index(manifests, cohort, policy, json.loads(args.pixel_group.read_text()))
    paths = {
        k: Path(getattr(args, k.replace("-", "_")))
        for k in ("studies", "artifact-root", "folds", "series")
    }
    fit, outer, parts, series, binding = development_parts(paths, args.overlay, policy, fixture)
    if binding["source_revision"] != cohort["source_revision"]:
        raise ValueError("Release source revision differs from the shard cohort")
    return (
        code,
        hashes,
        fixture,
        bag,
        configs,
        spec,
        policy,
        shards,
        manifests,
        index,
        paths,
        (fit, outer, parts, series, binding["source_revision"]),
    )


def measure(
    arms: dict,
    restored: dict,
    restore_seconds: float,
    bootstrap_seconds: float,
    admission_seconds: float,
) -> dict:
    """Conservative production timings derived only from this actual engineering session.

    Per update: the slowest observed update of either arm (effective 8, includes the
    trainer's own checkpoint writes). Final evaluation: the full-process tail after its last
    update, scaled from 16 to 1156 outer studies. Startup: notebook bootstrap plus
    admission, with the member restore scaled to all 27.7 GB of the complete cohort.
    """
    per_update, tail, copy, size = [], [], [], []
    for result in arms.values():
        elapsed = result["update_elapsed_seconds"]
        if len(elapsed) < 2 or any(b <= a for a, b in zip(elapsed, elapsed[1:])):
            raise ValueError("Actual contiguous per-update timings required")
        per_update.append(max([elapsed[0], *(b - a for a, b in zip(elapsed, elapsed[1:]))]))
        tail.append(result["phase_seconds"]["full"] - elapsed[-1])
        copy.append(result["checkpoint_copy_seconds"])
        size.append(result["checkpoint_bytes"])
    if restored["bytes"] <= 0 or min(tail) <= 0:
        raise ValueError("Actual restore/evaluation timings required")
    restore_full = restore_seconds * EXACT_MEMBER_BYTES / restored["bytes"]
    measured = {
        "seconds_per_update": max(per_update),
        "startup_seconds": bootstrap_seconds + admission_seconds - restore_seconds + restore_full,
        "checkpoint_seconds": max(copy),
        "final_evaluation_seconds": max(tail) * OUTER_STUDIES / ENGINEERING_STUDIES,
        "checkpoint_bytes": max(size),
    }
    if any(not value > 0 for value in measured.values()):
        raise ValueError("Measured timings/bytes must be positive")
    return measured


def run_engineering(args) -> dict:
    started = time.monotonic()
    deadline = started + min(args.remaining_seconds, ENGINEERING_SECONDS)
    import torch

    torch.set_num_threads(2)
    observed = check_runtime(engineering=True)
    code, hashes, fixture, bag, configs, spec, policy, shards, manifests, index, paths, data = (
        prepare(args)
    )
    fit, outer, parts, series, revision = data
    uids = sorted(map(str, list(parts["fitting"].study_uid) + list(parts["development"].study_uid)))
    out = Path(args.output)
    out.mkdir(mode=0o700)
    cache = out / "cache"
    restore_started = time.monotonic()
    restored = restore(cache, shards, manifests, index, uids)
    restore_seconds = time.monotonic() - restore_started
    tensors = admit_tensors(cache, policy, fixture, parts, outer, fit, series, revision)
    admit_initialization(configs, spec, bag)
    admitted = time.monotonic() - started
    results = {}
    for arm in ARMS:
        values = engineering_config(configs[arm], spec)
        results[arm] = recover_arm(
            code, out, arm, values, paths, args.overlay, cache, fixture, hashes, deadline
        )
    if (
        results["mean"]["finite_updates"][0]["loss"]
        != results["query"]["finite_updates"][0]["loss"]
    ):
        raise ValueError("Arms disagree on the original first-update loss")
    result = {
        "status": ENGINEERING_STATUS,
        "science_git": SCIENCE_GIT,
        "package_manifest_sha256": args.package_manifest_sha256,
        "fixture_sha256": FIXTURE_SHA256,
        "bag_CPU_sha256": BAG_CPU_SHA256,
        "index_sha256": INDEX_SHA256,
        "shard_manifest_sha256": list(SHARD_MANIFEST_SHA256),
        "encoder_spec_sha256": canonical(spec),
        # The exact fixed-10 configs a production segment must reproduce byte-for-byte.
        "production_config_sha256": {
            arm: canonical(production_config(configs[arm], spec)) for arm in ARMS
        },
        "restored": restored,
        "restore_seconds": restore_seconds,
        "bootstrap_seconds": args.bootstrap_seconds,
        "tensor_sha256": tensors,
        "arms": results,
        "admission_seconds": admitted,
        "measured": measure(results, restored, restore_seconds, args.bootstrap_seconds, admitted),
        "elapsed_seconds": time.monotonic() - started,
        "resources_before": observed,
        "resources_after": check_runtime(engineering=True),
        "GPUs_used_concurrently": 1,
        "gold_or_confirmation_used": False,
        "production_fitting": False,
        "clinical_accuracy_evidence": False,
        "competition_submission": False,
    }
    with (Path(args.receipt)).open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": ENGINEERING_STATUS}), flush=True)
    return result


PREVIOUS_OUTPUT_KEYS = {
    "kernel_slug",
    "version",
    "source_sha256",
    "receipt_sha256",
    "publication_sha256",
    "checkpoint_sha256",
}
SEGMENT_KEYS = {
    "arm",
    "segment",
    "start_step",
    "end_step",
    "previous_output",
    "engineering_receipt_sha256",
    "production_config_sha256",
    "output_cap_bytes",
}


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def segment_contract(contract: dict) -> dict:
    """Exact scheduled boundary of one production session; validated before any GPU work."""
    from scripts.kaggle.window_bag_plan import TOTAL_UPDATES

    if (
        not isinstance(contract, dict)
        or set(contract) != SEGMENT_KEYS
        or contract["arm"] not in ARMS
    ):
        raise ValueError("Exact segment contract required")
    start, end, previous = contract["start_step"], contract["end_step"], contract["previous_output"]
    if (
        any(type(v) is not int for v in (start, end, contract["segment"]))
        or not 0 <= start < end <= TOTAL_UPDATES
        or (start == 0) != (contract["segment"] == 0)
        or (start == 0) != (previous is None)
    ):
        raise ValueError("Segment boundary differs from the fixed schedule")
    if previous is not None and (
        set(previous) != PREVIOUS_OUTPUT_KEYS
        or not isinstance(previous["kernel_slug"], str)
        or previous["kernel_slug"].count("/") != 1
        or previous["version"] != 1
        or not all(_hex64(previous[k]) for k in PREVIOUS_OUTPUT_KEYS - {"kernel_slug", "version"})
    ):
        raise ValueError("Exact previous kernel/version/output identity required")
    if (
        not _hex64(contract["engineering_receipt_sha256"])
        or not _hex64(contract["production_config_sha256"])
        or type(contract["output_cap_bytes"]) is not int
        or contract["output_cap_bytes"] <= 0
    ):
        raise ValueError("Engineering receipt, config and output cap bindings required")
    return contract


def _write_once(source: Path, target: Path) -> dict:
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("xb") as writer:
        while block := reader.read(1 << 24):
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    target.chmod(0o444)
    if sha(source) != digest.hexdigest():
        raise ValueError("Source changed while publishing")
    return {"bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def publication_files(root: Path, run_dir: Path, final: bool) -> dict[str, Path]:
    """Explicit allowlist: continuation state, new immutable epochs, sidecars. Never cache."""
    files = {name: root / name for name in PUBLICATION_SIDECARS}
    files |= {"run/" + name: run_dir / name for name in PUBLICATION_RUN_FILES}
    if final:
        files |= {"run/" + name: run_dir / name for name in PUBLICATION_FINAL_FILES}
    epochs = run_dir / "epoch-checkpoints"
    if epochs.exists():
        for path in sorted(epochs.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix not in {".pt", ".json"}:
                raise ValueError("Unexpected epoch-checkpoint entry")
            files["run/epoch-checkpoints/" + path.name] = path
    for name, path in files.items():
        if path.is_symlink() or not path.is_file() or "cache" in Path(name).parts:
            raise ValueError("Publication member absent or outside the allowlist")
    return files


def publish_segment(root: Path, run_dir: Path, working: Path, cap: int, final: bool) -> dict:
    """Write-once private notebook output of one segment, within its explicit byte cap."""
    files = publication_files(root, run_dir, final)
    total = sum(path.stat().st_size for path in files.values())
    if total > cap:
        raise ValueError("Segment publication exceeds its explicit output cap")
    target = working / PUBLICATION_DIR
    working.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(working).free < total + PUBLICATION_RESERVE_BYTES:
        raise ValueError("Notebook output storage cannot hold the publication plus reserve")
    target.mkdir(mode=0o700)
    entries = {}
    for name, source in files.items():
        path = target / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        entries[name] = _write_once(source, path)
    for path in run_dir.glob("epoch-checkpoints/epoch-*.json"):
        record = json.loads(path.read_text())
        published = entries["run/epoch-checkpoints/" + record["checkpoint_file"]]
        if published["sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Epoch snapshot differs from its sidecar")
    manifest = {
        "schema_version": 1,
        "files": entries,
        "bytes": total,
        "cap_bytes": cap,
        "checkpoint_sha256": entries["run/checkpoint.pt"]["sha256"],
        "MRI_cache_or_reports_included": False,
    }
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    with (target / PUBLICATION_MANIFEST).open("xb") as handle:
        handle.write(payload)
    (target / PUBLICATION_MANIFEST).chmod(0o444)
    return {
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "files": len(entries),
        "bytes": total,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }


def _find(input_root: Path, name: str, expected: str) -> list[Path]:
    found = []

    def walk(directory: Path, depth: int) -> None:
        for entry in os.scandir(directory):
            if entry.is_symlink() or entry.name == "rsna-knee-abnormality-detection":
                continue
            if entry.is_file() and entry.name == name and sha(entry.path) == expected:
                found.append(Path(entry.path))
            elif entry.is_dir() and depth > 0:
                walk(Path(entry.path), depth - 1)

    walk(Path(input_root), SEARCH_DEPTH)
    return found


def restore_previous(input_root: Path, contract: dict, run_dir: Path) -> dict:
    """Stage the bound previous kernel output (a mounted read-only input) for resume."""
    previous = contract["previous_output"]
    manifests = _find(input_root, PUBLICATION_MANIFEST, previous["publication_sha256"])
    if len(manifests) != 1:
        raise ValueError("Exactly one bound previous segment publication must be mounted")
    target = manifests[0].parent
    receipt_path = target.parent / SEGMENT_RECEIPT
    if not receipt_path.is_file():
        raise ValueError("Previous segment receipt absent from its mounted output")
    receipt = json.loads(receipt_path.read_bytes())
    prior = receipt["contract"]
    if (
        canonical(receipt) != previous["receipt_sha256"]
        or receipt.get("status") != SEGMENT_STATUS
        or receipt["publication"]["manifest_sha256"] != previous["publication_sha256"]
        or receipt["checkpoint_sha256"] != previous["checkpoint_sha256"]
        or (prior["arm"], prior["segment"] + 1, prior["end_step"])
        != (contract["arm"], contract["segment"], contract["start_step"])
        or prior["engineering_receipt_sha256"] != contract["engineering_receipt_sha256"]
        or prior["production_config_sha256"] != contract["production_config_sha256"]
    ):
        raise ValueError("Mounted previous segment output differs from the bound identity")
    manifest = json.loads(manifests[0].read_bytes())
    for name, record in manifest["files"].items():
        relative = Path(name)
        path = target / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or sha(path) != record["sha256"]
        ):
            raise ValueError("Mounted previous segment member differs from its manifest")
    if manifest["checkpoint_sha256"] != previous["checkpoint_sha256"]:
        raise ValueError("Mounted continuation checkpoint differs from the bound boundary")
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = run_dir / "checkpoint.pt"
    _write_once(target / "run/checkpoint.pt", staged)
    staged.chmod(0o600)
    if sha(staged) != previous["checkpoint_sha256"]:
        raise ValueError("Staged checkpoint copy differs")
    return {
        "publication_sha256": previous["publication_sha256"],
        "checkpoint_sha256": previous["checkpoint_sha256"],
        "files": len(manifest["files"]),
    }


def run_segment(args) -> dict:
    """One bounded production session; the final segment alone evaluates epoch 10."""
    from scripts.kaggle.window_bag_plan import TOTAL_UPDATES

    started = time.monotonic()
    deadline = started + min(args.remaining_seconds, SESSION_SECONDS)
    contract = segment_contract(json.loads(Path(args.segment).read_text()))
    observed = check_runtime(engineering=False)
    code, hashes, fixture, bag, configs, spec, policy, shards, manifests, index, paths, data = (
        prepare(args)
    )
    fit, outer, _, _, _ = data
    arm = contract["arm"]
    values = production_config(configs[arm], spec)
    if canonical(values) != contract["production_config_sha256"]:
        raise ValueError("Production config differs from the engineering-admitted config")
    out = Path(args.output)
    out.mkdir(mode=0o700)
    root = out / arm
    root.mkdir(mode=0o700)
    run_dir = root / "run"
    run_dir.mkdir(mode=0o700)
    extra = ["--output", str(run_dir)]
    previous = None
    if contract["previous_output"] is not None:
        previous = restore_previous(Path(args.input_root), contract, run_dir)
        extra += ["--resume", str(run_dir / "checkpoint.pt")]
    restored = restore(out / "cache", shards, manifests, index, [*fit.study_uid, *outer.study_uid])
    config = root / "config.json"
    config.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")
    final = contract["end_step"] == TOTAL_UPDATES
    if not final:
        extra += ["--stop-after-updates", str(contract["end_step"] - contract["start_step"])]
    command = trainer_command(code, config, paths, args.overlay, out / "cache")
    timings: dict = {}
    execute(
        command,
        extra,
        root,
        "segment",
        0 if final else 75,
        deadline,
        trainer_env(code),
        timings,
        root,
    )
    import torch

    state = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if state["step"] != contract["end_step"] or state["binding"]["code_sha256"] != hashes:
        raise ValueError("Segment did not stop at its scheduled boundary/source")
    del state
    snapshots = sorted((run_dir / "epoch-checkpoints").glob("epoch-*.pt"))
    publication = publish_segment(
        root, run_dir, Path(args.publish_root), contract["output_cap_bytes"], final
    )
    result = {
        "status": SEGMENT_STATUS,
        "contract": contract,
        "science_git": SCIENCE_GIT,
        "package_manifest_sha256": args.package_manifest_sha256,
        "index_sha256": INDEX_SHA256,
        "restored": restored,
        "previous_restored": previous,
        "checkpoint_sha256": publication["checkpoint_sha256"],
        "publication": publication,
        "epoch_snapshots": {p.name: sha(p) for p in snapshots},
        "final_epoch10_evaluated": final,
        "prediction_sha256": sha(run_dir / "outer_predictions.csv") if final else None,
        "seconds": timings,
        "resources_before": observed,
        "resources_after": check_runtime(engineering=False),
        "intermediate_selection": False,
        "competition_submission": False,
    }
    with (Path(args.publish_root) / SEGMENT_RECEIPT).open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": SEGMENT_STATUS, "end_step": contract["end_step"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engineering", "segment"), required=True)
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
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--publish-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--segment", type=Path)
    parser.add_argument("--remaining-seconds", type=float, required=True)
    parser.add_argument("--bootstrap-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.mode == "segment" and args.segment is None:
        parser.error("--mode segment requires --segment")
    if args.mode == "engineering" and args.receipt is None:
        parser.error("--mode engineering requires --receipt")
    (run_engineering if args.mode == "engineering" else run_segment)(args)


if __name__ == "__main__":
    main()
