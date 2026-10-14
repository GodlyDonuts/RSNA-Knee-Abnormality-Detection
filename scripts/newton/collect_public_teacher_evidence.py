"""Root-owned frozen public-teacher evidence collection for fitting studies only.

`--mode render` is the eight-study acceptance: it renders the retained original fitting
DICOMs with the reviewed Pilkwang v15 renderer (recorded series order replayed when
given). `--mode cache` is full collection from the source-bound public triplet cache and
requires an accepted render receipt with the identical teacher/source/runtime binding.
Neither mode reads labels, reports, gold, confirmation or evaluation pixels; the fitting
roster file holds UIDs only. Outputs are immutable and reconciled, never overwritten.
Not run by the authoring session.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rsna.competitive import public_input  # noqa: E402
from rsna.competitive import teacher_evidence as te  # noqa: E402
from rsna.competitive.public_source_table import CORRECTED_SOURCE_REVISION  # noqa: E402
from scripts.newton.compare_public_dinov2_inputs import (  # noqa: E402
    RAW_SPLIT,
    file_sha256,
    member_pixel_group,
    select_roster,
)
from tools.research.public_dinov2_input_reference import (  # noqa: E402
    contract,
    provenance,
    reference,
)
from tools.research.public_teacher_evidence import teacher  # noqa: E402

PUBLIC_CONFIG = (
    ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1/public-triplet-config.json"
)
# Fields that must be identical between the accepted eight-study receipt and full collection.
TEACHER_IDENTITY = (
    "kind",
    "teacher",
    "source_sha256",
    "renderer_node_text_sha256",
    "teacher_node_text_sha256",
    "teacher_review_sha256",
    "checkpoint_sha256",
    "member_manifest_sha256",
    "member_config",
    "pixel_group_sha256",
    "backbone_config_sha256",
    "runtime",
    "fitting_roster_sha256",
    "targets",
    "windows",
)


def runtime(device: torch.device) -> dict:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": device.type,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def read_roster(path: Path) -> list[str]:
    uids = json.loads(Path(path).read_text())
    if not isinstance(uids, list) or not all(isinstance(u, str) and u for u in uids):
        raise te.EvidenceError("Fitting roster file must be a JSON list of study UIDs only")
    return uids


def render_rows(namespace: dict, args: argparse.Namespace, uids: list[str]) -> Iterator[tuple]:
    series = pd.read_csv(args.series, usecols=["SeriesInstanceUID", "Anatomical_Plane"])
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    order = json.loads(Path(args.series_order).read_text()) if args.series_order else {}
    for uid in uids:
        replay = (
            reference.preserve_series_order(uid, order[uid])
            if uid in order
            else contextlib.nullcontext()
        )
        with replay:
            row = reference.render_study(
                namespace, Path(args.dicom_root) / RAW_SPLIT / uid, plane_map, RAW_SPLIT
            )
        contract.validate_rendering(row["images"], row["slot_mask"])
        yield row["images"], row["slot_mask"]


def cache_rows(args: argparse.Namespace, uids: list[str]) -> tuple[str, Iterator[tuple]]:
    policy = json.loads(PUBLIC_CONFIG.read_text())["public_input"]
    index, index_sha = public_input.load_public_index(
        Path(args.cache), policy, args.source_revision
    )
    if (
        index["fitting_uids_sha256"] != te.FITTING_ROSTER_SHA256
        or index["evaluation_uids_sha256"] != te.EVALUATION_ROSTER_SHA256
    ):
        raise te.EvidenceError("Public triplet cache is not the accepted fitting/evaluation split")
    root = Path(args.cache) / public_input.CACHE_DIRECTORY

    def rows() -> Iterator[tuple]:
        for uid in uids:
            member = index["members"][uid]
            payload = (root / member["path"]).read_bytes()
            if (
                len(payload) != member["bytes"]
                or hashlib.sha256(payload).hexdigest() != member["sha256"]
            ):
                raise te.EvidenceError("Public triplet member bytes differ from the index")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                images, mask = archive["images"], archive["slot_mask"]
            yield torch.from_numpy(images), torch.from_numpy(mask)

    return index_sha, rows()


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint_raw = Path(args.checkpoint).read_bytes()
    manifest = Path(args.member_manifest).read_bytes()
    source = Path(args.source).read_bytes()
    config_raw = Path(args.backbone_config).read_bytes()
    member = member_pixel_group(manifest)
    namespace, receipt = teacher.load_namespace(source, member["pixel_group"])
    fitting = read_roster(Path(args.fitting_roster))
    if args.mode == "render":
        if args.roster_size > te.ACCEPTANCE_STUDIES:
            raise te.EvidenceError("Render acceptance is bounded to eight fitting studies")
        requested = select_roster(pd.Series(fitting), args.roster_size)
        cap = te.ACCEPTANCE_STUDIES
    else:
        requested = sorted(fitting)
        cap = te.FITTING_STUDIES
    requested = te.fitting_request(fitting, requested, max_studies=cap)
    binding = {
        "kind": te.KIND,
        "teacher": te.TEACHER,
        "source_sha256": receipt["source_sha256"],
        "renderer_node_text_sha256": receipt["node_text_sha256"],
        "teacher_node_text_sha256": receipt["teacher_node_text_sha256"],
        "teacher_review_sha256": receipt["teacher_review_sha256"],
        "checkpoint_sha256": teacher.sha256(checkpoint_raw),
        "member_manifest_sha256": provenance.sha256(manifest),
        "member_config": member["config"],
        "pixel_group_sha256": public_input.canonical_sha256(member["pixel_group"]),
        "backbone_config_sha256": teacher.sha256(config_raw),
        "runtime": runtime(device),
        "fitting_roster_sha256": te.FITTING_ROSTER_SHA256,
        "requested_roster_sha256": public_input.roster_sha256(requested),
        "requested_studies": requested,
        "input": {"mode": args.mode},
        "targets": list(te.TARGETS),
        "windows": te.WINDOWS,
    }
    if args.mode == "cache":
        accepted, _ = te.load_evidence(Path(args.acceptance))
        prior = accepted["binding"]
        if (
            prior["input"].get("mode") != "render"
            or len(prior["requested_studies"]) != te.ACCEPTANCE_STUDIES
            or any(prior[k] != binding[k] for k in TEACHER_IDENTITY)
        ):
            raise te.EvidenceError("Full collection requires the identical accepted teacher")
        index_sha, rows = cache_rows(args, requested)
        binding["input"] = {
            "mode": "cache",
            "index_sha256": index_sha,
            "source_revision": args.source_revision,
            "acceptance_receipt_sha256": file_sha256(Path(args.acceptance) / te.RECEIPT_FILE),
        }
    else:
        binding["input"] = {
            "mode": "render",
            "series_order_sha256": file_sha256(Path(args.series_order))
            if args.series_order
            else None,
        }
        rows = render_rows(namespace, args, requested)
    existing = te.reconcile(Path(args.out), binding)
    if existing is not None:
        return {"status": "reconciled", **existing["evidence"], **existing["summary"]}
    config = teacher.backbone_config(config_raw, args.backbone_config_sha256)
    model = teacher.build_teacher(
        namespace,
        member["config"],
        teacher.load_checkpoint(checkpoint_raw),
        teacher.dinov2_backbone(config),
        device,
    )
    collected = [teacher.collect_study(namespace, model, i, m, device) for i, m in rows]
    written = te.write_evidence(Path(args.out), binding, collected)
    return {"status": "collected", **written["evidence"], **written["summary"]}


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("render", "cache"), required=True)
    for name in ("source", "member-manifest", "checkpoint", "backbone-config", "fitting-roster"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--backbone-config-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--roster-size", type=int, default=te.ACCEPTANCE_STUDIES)
    parser.add_argument("--dicom-root")
    parser.add_argument("--series")
    parser.add_argument("--series-order")
    parser.add_argument("--cache")
    parser.add_argument("--acceptance")
    parser.add_argument("--source-revision", default=CORRECTED_SOURCE_REVISION)
    args = parser.parse_args(argv)
    needed = ("dicom_root", "series") if args.mode == "render" else ("cache", "acceptance")
    if any(getattr(args, name) is None for name in needed):
        parser.error(f"--mode {args.mode} requires {', '.join(needed)}")
    return args


def main() -> None:
    try:
        report = run(parse())
    except Exception as error:  # type only; no UIDs, paths or messages leave the node
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
