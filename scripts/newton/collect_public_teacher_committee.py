"""Collect the accepted twenty-teacher committee on the fitting-only public cache.

This executable needs actual shared/full MRI acceptance, a complete accepted cache
and separately admitted node-local funding. It does not allocate GPUs or publish
individual evidence. All original graph/checkpoint/source bindings are retained.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rsna.competitive import committee_evidence as ce  # noqa: E402
from rsna.competitive import public_input as pi  # noqa: E402
from rsna.competitive import teacher_evidence as te  # noqa: E402
from scripts.newton.collect_public_teacher_evidence import (  # noqa: E402
    cache_rows,
    read_roster,
    runtime,
)
from scripts.newton.compare_public_dinov2_inputs import member_pixel_group  # noqa: E402
from tools.research.public_teacher_evidence import committee, teacher  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_acceptance(value: dict, source: dict) -> None:
    if (
        value.get("status") != "actual_twenty_teacher_shared_prefix_eight_MRI_bitwise_accepted"
        or value.get("studies") != 8
        or value.get("members") != 20
        or value.get("targets") != 12
        or value.get("windows") != 10
        or value.get("all_full_vs_shared_fingerprint_window_logits_and_probabilities_bit_exact")
        is not True
        or value.get("original_source_sha256") != source["source_sha256"]
        or value.get("teacher_node_text_sha256") != source["teacher_node_text_sha256"]
        or value.get("prefix", {}).get("original_inspection_sha256")
        != committee.INSPECTED_PREFIX_DIGEST
        or value.get("labels_reports_or_evaluation_pixels_read") is not False
    ):
        raise ce.CommitteeEvidenceError("Actual original shared/full MRI acceptance required")


def validate_funding(value: dict, out: Path) -> None:
    if (
        value.get("status") != "root_admitted_private_committee_evidence_storage"
        or value.get("cap_bytes") != ce.MAX_EVIDENCE_BYTES
        or value.get("output") != str(out)
        or value.get("shared_growth_bytes") != 0
        or type(value.get("retained_obligations_bytes")) is not int
        or value["retained_obligations_bytes"] < 0
        or value.get("floor_bytes") != 20 * 2**30
        or not str(out).startswith("/tmp/")
    ):
        raise ce.CommitteeEvidenceError("Separate private evidence storage admission required")
    if shutil.disk_usage(out.parent).free <= (
        value["retained_obligations_bytes"] + value["floor_bytes"] + ce.MAX_EVIDENCE_BYTES
    ):
        raise ce.CommitteeEvidenceError("Fresh private storage reserve insufficient")


def run(args: argparse.Namespace) -> dict:
    root, out = Path(args.teacher_root), Path(args.out)
    funding = Path(args.funding)
    validate_funding(json.loads(funding.read_text()), out)
    source = (
        root
        / "code/initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json"
    ).read_bytes()
    member_raw = Path(args.member_manifest).read_bytes()
    member = member_pixel_group(member_raw)
    namespace, source_binding = teacher.load_namespace(source, member["pixel_group"])
    acceptance = Path(args.acceptance)
    validate_acceptance(json.loads(acceptance.read_text()), source_binding)
    fitting = read_roster(root / "fitting-uids.json")
    requested = te.fitting_request(fitting, sorted(fitting), max_studies=te.FITTING_STUDIES)
    index_sha, rows = cache_rows(args, requested)
    config_raw = (root / "backbone-config.json").read_bytes()
    config = teacher.backbone_config(config_raw, args.backbone_config_sha256)
    inspected = json.loads((root / "original-committee-prefix-inspection.json").read_text())
    expected = {v["file"]: v for v in inspected["members"]}
    device = torch.device("cuda")
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != json.loads(acceptance.read_text())["GPU"]
    ):
        raise ce.CommitteeEvidenceError("Accepted allocated GPU context required")
    models, references, records = [], [], []
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if len(manifest["members"]) != ce.MEMBERS:
            raise ce.CommitteeEvidenceError("Exact twenty original teachers required")
        for item in manifest["members"]:
            checkpoint_sha = expected[item["file"]]["checkpoint_sha256"]
            checkpoint = teacher.load_checkpoint(archive.read(item["file"]), checkpoint_sha)
            model = namespace["Model"](
                teacher.dinov2_backbone(config),
                config["hidden_size"],
                pool=item["config"]["pool"],
                prior=item["config"]["prior"],
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device).eval().requires_grad_(False)
            models.append(model)
            references.append(torch.as_tensor(checkpoint["fingerprint"]).detach().cpu())
            records.append(
                {
                    "id": item["file"],
                    "file": item["file"],
                    "checkpoint_sha256": checkpoint_sha,
                    "config": item["config"],
                    "config_sha256": pi.canonical_sha256(item["config"]),
                }
            )
            del checkpoint
            gc.collect()
    built = committee.Committee(namespace, models, [v["id"] for v in records])
    built.verify_fingerprints(device, references)
    binding = {
        "kind": ce.KIND,
        "source": ce.SOURCE,
        "members": records,
        "committee_receipt_sha256": sha(acceptance),
        "member_manifest_sha256": hashlib.sha256(member_raw).hexdigest(),
        "prefix": built.prefix_summary,
        "forward": "shared_prefix_bitwise_checked",
        **{
            k: source_binding[k]
            for k in ("source_sha256", "teacher_node_text_sha256", "teacher_review_sha256")
        },
        "renderer_node_text_sha256": source_binding["node_text_sha256"],
        "pixel_group_sha256": pi.canonical_sha256(member["pixel_group"]),
        "backbone_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "runtime": runtime(device),
        "fitting_roster_sha256": te.FITTING_ROSTER_SHA256,
        "requested_roster_sha256": pi.roster_sha256(requested),
        "requested_studies": requested,
        "input": {
            "mode": "cache",
            "index_sha256": index_sha,
            "source_revision": args.source_revision,
        },
        "targets": list(te.TARGETS),
        "windows": te.WINDOWS,
        "storage": {"cap_bytes": ce.MAX_EVIDENCE_BYTES, "funding_receipt_sha256": sha(funding)},
    }
    existing = ce.reconcile(out, binding)
    if existing is not None:
        return {"status": "reconciled", "evidence": existing["evidence"]}
    collected = []
    for images, mask in rows:
        collected.append(built.collect_study(images, mask, device, mode="shared", check_full=False))
        if len(collected) % 100 == 0:
            print(
                json.dumps(
                    {"status": "private_fitting_collection_progress", "studies": len(collected)}
                ),
                flush=True,
            )
    receipt = ce.write_evidence(out, binding, collected)
    return {
        "status": "private_fitting_committee_collected",
        "evidence": receipt["evidence"],
        "summary": receipt["summary"],
        "accuracy_gain_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "teacher-root",
        "member-manifest",
        "archive",
        "acceptance",
        "funding",
        "cache",
        "source-revision",
        "backbone-config-sha256",
        "out",
    ):
        parser.add_argument("--" + name, required=True)
    try:
        print(json.dumps(run(parser.parse_args()), allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
