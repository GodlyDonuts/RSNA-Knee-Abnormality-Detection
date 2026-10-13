"""Prepared Kaggle CPU notebook entry: render source-bound public inputs at the mount.

NOT RUN by the authoring session; root alone reviews, pushes and dispatches. ``--mode probe``
renders the retained eight fitting studies from the attached competition mount (no
download) and checks exact pixel-triplet and adapter tensor parity against the accepted
CPU receipt; ``--mode shard`` renders one root-funded shard of the complete cohort. Both
use the producer's ``render_member`` unchanged and the consolidated wrapper's source binding
and ordered filtered transport. Private outputs (receipts, source order, encoded members,
manifests) stay under ``--private-out`` in the private notebook output; stdout carries only
sanitized counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsna.competitive import kaggle_source_render as ksr  # noqa: E402
from rsna.competitive import public_cache_producer as producer  # noqa: E402
from rsna.competitive import public_input as pi  # noqa: E402
from rsna.competitive import public_source_table as pst  # noqa: E402

WRAPPER_PATH = ROOT / "scripts/newton/build_public_triplet_cache.py"
CODE_FILES = (
    "scripts/kaggle/render_public_source.py",
    "scripts/newton/build_public_triplet_cache.py",
    "rsna/competitive/kaggle_source_render.py",
    "rsna/competitive/public_cache_producer.py",
    "rsna/competitive/public_source_table.py",
    "rsna/competitive/public_input.py",
    "tools/research/public_dinov2_input_reference/dino_adapter.py",
    "tools/research/public_dinov2_input_reference/reference.py",
    "tools/research/public_dinov2_input_reference/provenance.py",
)
MODES = ("probe", "shard")


def wrapper():
    spec = importlib.util.spec_from_file_location("build_public_triplet_cache", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Consolidated wrapper is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_sha256(root: Path = ROOT) -> str:
    return ksr.canonical_sha256(
        {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CODE_FILES}
    )


def private_output(path: Path, mount: Path) -> Path:
    path = path.resolve()
    if path == mount or mount in path.parents or path.exists():
        raise ValueError("Private output must be new and outside the competition mount")
    path.mkdir(mode=0o700, parents=True)
    return path


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    runtime = ksr.check_cpu_runtime()
    code = code_sha256()
    budget = ksr.load_budget(Path(args.budget), args.budget_sha256, code, runtime)
    if args.mode == "shard" and budget["runtime_sha256"] is None:
        raise ValueError("Shards require the runtime accepted from an actual probe")
    w = wrapper()
    mount = ksr.resolve_mount(Path(args.input_root)).resolve()
    out = private_output(Path(args.private_out), mount)
    scratch = Path(args.scratch).resolve()
    scratch.mkdir(mode=0o700, parents=True, exist_ok=False)
    from rsna.competitive.folds import load_folds
    from rsna.competitive.supervision import roster_sha
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release

    config_bytes = w.CONFIG.read_bytes()
    policy = json.loads(config_bytes)["public_input"]
    pi.validate_public_input(policy)
    manifest = Path(args.member_manifest).read_bytes()
    renderer = producer.Renderer(
        policy, w.member_pixel_group(manifest), Path(args.source).read_bytes()
    )
    frame, release = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, release)
    fitting, outer = select_development(frame, folds, 0)
    overlay = json.loads(w.OVERLAY.read_text())
    if (
        roster_sha(fitting) != overlay["fitting_uid_sha256"]
        or roster_sha(outer) != overlay["outer_uid_sha256"]
        or release["manifest_sha256"] != overlay["release_sha256"]
    ):
        raise ValueError("Fitting/development rosters differ from the accepted hashes")
    fitting_uids = list(map(str, fitting.study_uid))
    evaluation_uids = list(map(str, outer.study_uid))
    # The release's reviewed series metadata includes additional columns and
    # accepted plane corrections. The mounted CSV is an original source asset,
    # not a byte-identical substitute for that immutable training input.
    series_csv = Path(args.accepted_series)
    mounted_series_sha256 = hashlib.sha256((mount / "train_series.csv").read_bytes()).hexdigest()
    objects, table, input_hashes, _ = w.bind_sources(
        w.read_inventory(w.INVENTORY_DIR),
        series_csv,
        release,
        fitting_uids,
        evaluation_uids,
        w.CORRECTION_PLAN,
        w.CORRECTION_RECEIPT,
    )
    changes = pst.load_correction(
        w.CORRECTION_PLAN.read_bytes(), json.loads(w.CORRECTION_RECEIPT.read_bytes())
    )
    corrections = ksr.verify_corrections(mount, changes)
    records = w.producer_records(objects)
    source_studies = {u: producer.study_source_sha256(u, records[u]) for u in sorted(records)}
    import pandas as pd

    series = pd.read_csv(series_csv, dtype=str)
    plane_map = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
    binding = {
        "code_sha256": code,
        "runtime_sha256": ksr.canonical_sha256(runtime),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "member_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "renderer": renderer.binding,
        "folds_sha256": overlay["folds_sha256"],
        **input_hashes,
        "mounted_series_csv_sha256": mounted_series_sha256,
        "source_table": table,
        **corrections,
    }

    def complete_cohort(studies_per_shard: int) -> dict:
        return ksr.cohort_manifest(
            fitting_uids=fitting_uids,
            evaluation_uids=evaluation_uids,
            source_studies=source_studies,
            source_revision=pst.CORRECTED_SOURCE_REVISION,
            binding=binding,
            studies_per_shard=studies_per_shard,
        )

    if args.mode == "probe":
        uids, index = ksr.parity_roster(fitting_uids), None
    else:
        cohort = json.loads(Path(args.cohort_manifest).read_bytes())
        if ksr.canonical_sha256(cohort) != args.cohort_manifest_sha256:
            raise ValueError("Cohort manifest differs from its root-recorded hash")
        if cohort != complete_cohort(cohort["studies_per_shard"]):
            raise ValueError("Mounted sources, code or runtime differ from the cohort manifest")
        index = args.shard_index
        if not 0 <= index < len(cohort["shards"]):
            raise ValueError("Shard index is outside the complete-cohort manifest")
        uids = ksr.assign_shards(source_studies, cohort["studies_per_shard"])[index]
    if len(uids) > budget["max_studies"]:
        raise ValueError("Shard exceeds the funded study count")
    funding = {k: budget[k] for k in ("max_study_source_bytes", "min_free_bytes")}
    labels = {r["study_uid"]: r for r in overlay["studies"]} if args.mode == "probe" else {}
    from rsna.schema import TARGETS

    namespace = renderer.namespace
    from tools.research.public_dinov2_input_reference import reference as ref

    shard = out / "shard"
    records_out, parity, source_parity, sizes, seconds = [], {}, {}, [], []
    triplet_matches = 0
    for uid in uids:
        tick = time.monotonic()
        if time.monotonic() - started > budget["max_wall_seconds"]:
            raise TimeoutError("Funded wall time exhausted")
        try:
            _, order = w.transport_study(
                mount / producer.SPLIT / uid,
                scratch,
                uid,
                records[uid],
                funding,
                origin="kaggle_mount",
            )
            order["source_revision"] = pst.CORRECTED_SOURCE_REVISION
            if args.mode == "probe":
                # Compare to the original source BEFORE transport, using the
                # same recorded metadata row order. A historical hash from a
                # different filesystem's enumeration is supplementary evidence.
                import torch

                from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

                source_dir = mount / producer.SPLIT / uid
                actual_series = {entry.name for entry in source_dir.iterdir() if entry.is_dir()}
                if actual_series != set(order["series_order"]):
                    raise ValueError(
                        "Direct reference probe requires only the accepted source series"
                    )
                row = labels[uid]
                y = torch.tensor([row["labels"][t] for t in TARGETS], dtype=torch.float32)
                mask = torch.tensor([row["mask"][t] for t in TARGETS], dtype=torch.float32)
                with ref.preserve_series_order(uid, order["series_order"]):
                    direct = render_inputs(
                        namespace, mount / producer.SPLIT / uid, plane_map, y, mask
                    )
                source_parity[uid] = ksr.parity_study_bytes(direct, y.numpy(), mask.numpy())
            with ref.preserve_series_order(uid, order["series_order"]):
                payload, receipt, inputs = producer.render_member(
                    uid,
                    scratch,
                    records[uid],
                    plane_map,
                    renderer=renderer,
                    source_sha256=source_studies[uid],
                    source_revision=pst.CORRECTED_SOURCE_REVISION,
                    role="fitting" if uid in set(fitting_uids) else "evaluation",
                )
                if args.mode == "probe":
                    public = ref.render_study(namespace, scratch / producer.SPLIT / uid, plane_map)
                    windows = ref.windows(namespace, public["images"])
                    expected = windows.permute(1, 0, 2, 3, 4).float() / 255
                    import torch

                    if torch.equal(inputs["images"], expected) and torch.equal(
                        inputs["series_mask"], public["slot_mask"]
                    ):
                        triplet_matches += 1
                    row = labels[uid]
                    parity[uid] = ksr.parity_study_bytes(
                        inputs,
                        [row["labels"][t] for t in TARGETS],
                        [row["mask"][t] for t in TARGETS],
                    )
        finally:
            transported = scratch / producer.SPLIT
            if transported.exists():
                for path in [transported, *transported.rglob("*")]:
                    if path.is_dir() and not path.is_symlink():
                        path.chmod(0o700)
                shutil.rmtree(transported)
        sizes.append(ksr.measured_encodings(payload))
        records_out.append(ksr.write_shard_member(shard, uid, payload, receipt, order, inputs))
        seconds.append(time.monotonic() - tick)
        if sum(r["encoded"]["bytes"] for r in records_out) > budget["max_output_bytes"]:
            raise ValueError("Encoded shard exceeds the funded output bytes")
    totals = {k: sum(s[k] for s in sizes) for k in ("exact", "gzip", "npz_deflate")}
    totals_mean = totals["gzip"] / len(records_out)
    if args.mode == "probe":
        plan = ksr.plan_shards(
            cohort_studies=len(source_studies),
            seconds_per_study=max(seconds),
            encoded_bytes_per_study=max(s["gzip"] for s in sizes),
            scratch_bytes_per_study=budget["max_study_source_bytes"],
            max_wall_seconds=budget["max_wall_seconds"],
            max_output_bytes=budget["max_output_bytes"],
            scratch_free_bytes=shutil.disk_usage(scratch).free,
            min_free_bytes=budget["min_free_bytes"],
        )
        cohort = complete_cohort(plan["studies_per_shard"])
    producer._write_exclusive(out / ksr.COHORT_MANIFEST, producer._json_bytes(cohort))
    shard_sha = ksr.seal_shard(shard, cohort, index, records_out, runtime)
    summary = {
        "mode": args.mode,
        "placement": ksr.PLACEMENT,
        "studies": len(records_out),
        "shard_index": index,
        "shard_manifest_sha256": shard_sha,
        "cohort_manifest_sha256": ksr.canonical_sha256(cohort),
        "code_sha256": code,
        "runtime": runtime,
        "runtime_sha256": ksr.canonical_sha256(runtime),
        "source_table": {
            k: table[k] for k in sorted(table) if not isinstance(table[k], (list, dict))
        },
        **corrections,
        "encoded_bytes": totals,
        "mean_gzip_bytes_per_study": totals_mean,
        "seconds_per_study": {"mean": sum(seconds) / len(seconds), "max": max(seconds)},
        "resources": ksr.resources([Path(args.private_out).parent, scratch]),
        "wall_seconds": time.monotonic() - started,
        "labels_stored": False,
        "source_downloaded": False,
        "clinical_evidence": False,
    }
    if args.mode == "probe":
        orders = {
            "ranked": uids,
            "sorted": sorted(uids),
        }
        digests = {
            name: ksr.parity_digest(parity[u] for u in order) for name, order in orders.items()
        }
        direct_digests = {
            name: ksr.parity_digest(source_parity[u] for u in order)
            for name, order in orders.items()
        }
        source_equal = all(parity[u] == source_parity[u] for u in uids)
        matched = [n for n, d in digests.items() if d == ksr.ACCEPTED_PARITY_TENSOR_SHA256]
        summary |= {
            "original_pixel_triplet_matches": triplet_matches,
            "tensor_sha256": digests,
            "accepted_tensor_sha256": ksr.ACCEPTED_PARITY_TENSOR_SHA256,
            "direct_source_tensor_sha256": direct_digests,
            "direct_source_studies_matched": sum(parity[u] == source_parity[u] for u in uids),
            "parity_reference": "original_source_before_transport_same_recorded_series_order",
            "historical_reference_matched": bool(matched),
            "accepted_parity": source_equal and triplet_matches == len(uids),
            "accepted_parity_order": matched[0] if matched else None,
            "proposed_shards": plan,
        }
    payload = producer._json_bytes(summary)
    producer._write_exclusive(out / "summary.json", payload)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--private-out", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--budget-sha256", required=True)
    for name in (
        "source",
        "member-manifest",
        "studies",
        "artifact-root",
        "folds",
        "accepted-series",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cohort-manifest")
    parser.add_argument("--cohort-manifest-sha256")
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    if args.mode == "shard" and (
        args.cohort_manifest is None
        or args.cohort_manifest_sha256 is None
        or args.shard_index is None
    ):
        parser.error("--mode shard requires the cohort manifest, its hash and a shard index")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
