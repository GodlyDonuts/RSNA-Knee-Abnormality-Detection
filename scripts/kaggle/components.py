"""Optional checkpoint-only Kaggle payloads over one immutable common bundle.

Local/stdlib only: no network, credential, weight acquisition or account access.
A standalone ``bundle-manifest.json`` without ``common_bundle`` is unchanged.
A component manifest binds one exact, fully byte-verified private common Dataset
(identity, version, manifest, member table, encoder, licence and code) and adds
only checkpoint-specific members. Every mismatch fails closed; there is no latest
version lookup and no partial fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath

COMMON_MANIFEST = "common-manifest.json"
COMMON_KIND = "rsna_common_encoder_code_bundle"
SHA = re.compile(r"[0-9a-f]{64}")
SLUG = re.compile(r"[a-z0-9_-]+/[a-z0-9-]+")
COMMON_FILES = {"encoder.json", "wheelhouse.json", "DINOv3-LICENSE.md"}
COMMON_PREFIXES = ("encoder/", "code/")
CANDIDATE_FILES = {
    "checkpoint.pt",
    "snapshot.json",
    "fitting-admission.json",
    "baseline-proof.json",
}
RESERVED = {"bundle-manifest.json", "dataset-metadata.json", COMMON_MANIFEST}
COMMON_OPERATION = "common-operation.json"
COMMON_OPERATION_KIND = "rsna_common_dataset_operation"
COMMON_OPERATION_KEYS = {
    "schema_version",
    "kind",
    "account",
    "dataset_slug",
    "dataset_version",
    "private",
    "bundle_path",
    "manifest_sha256",
    "manifest_bytes",
    "dataset_metadata_sha256",
    "files_sha256",
    "member_count",
    "total_bytes",
    "regenerated_common_sha256",
    "checkpoint_format",
    "license_source",
    "license_sha256",
    "encoder_spec_sha256",
    "code_sha256",
    "wheelhouse_manifest_sha256",
    "shared_wheelhouse",
    "source_receipt",
    "node",
    "checkpoint_entry",
    "competition_submission_eligible",
    "retry_prohibited",
}
BINDING_KEYS = {
    "schema_version",
    "dataset_slug",
    "dataset_id",
    "dataset_version",
    "manifest_sha256",
    "manifest_bytes",
    "files_sha256",
    "member_count",
    "total_bytes",
    "verification_sha256",
    "checkpoint_format",
    "license_source",
    "license_sha256",
    "encoder_spec_sha256",
    "code_sha256",
    "wheelhouse_manifest_sha256",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def confined(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Component member path must be a nonempty POSIX string")
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ("", ".", "..") for x in name.split("/")) or str(p) != name:
        raise ValueError("Noncanonical component member path")
    return p


def role(name):
    confined(name)
    if name in RESERVED:
        raise ValueError("Reserved manifest name cannot be a payload member")
    if name in CANDIDATE_FILES:
        return "candidate"
    if name in COMMON_FILES or name.startswith(COMMON_PREFIXES):
        return "common"
    raise ValueError("Unclassified payload member cannot be split")


def validate_table(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("Nonempty member table required")
    folded = set()
    for name, item in files.items():
        confined(name)
        if name.casefold() in folded:
            raise ValueError("Case-insensitive member path collision")
        folded.add(name.casefold())
        if (
            not isinstance(item, dict)
            or set(item) != {"sha256", "bytes"}
            or not isinstance(item["sha256"], str)
            or not SHA.fullmatch(item["sha256"])
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
        ):
            raise ValueError("Member record must be an exact SHA-256 and byte size")
    return files


def split(files):
    """Partition a complete standalone member table into common and candidate parts."""
    validate_table(files)
    common, candidate = {}, {}
    for name, item in sorted(files.items()):
        (common if role(name) == "common" else candidate)[name] = item
    if "checkpoint.pt" not in candidate or "snapshot.json" not in candidate:
        raise ValueError("Checkpoint payload must carry its checkpoint and snapshot")
    return common, candidate


def code_hash(files):
    return object_hash({k: v for k, v in sorted(files.items()) if k.startswith("code/")})


def build_common_manifest(
    files,
    *,
    checkpoint_format,
    license_source,
    license_sha256,
    encoder_spec_sha256,
    shared_wheelhouse,
    competition,
):
    validate_table(files)
    if any(role(name) != "common" for name in files):
        raise ValueError("Checkpoint-specific member cannot enter a common bundle")
    manifest = {
        "schema_version": 1,
        "kind": COMMON_KIND,
        "competition": competition,
        "checkpoint_format": checkpoint_format,
        "license_source": license_source,
        "license_sha256": license_sha256,
        "encoder_spec_sha256": encoder_spec_sha256,
        "code_sha256": code_hash(files),
        "shared_wheelhouse": shared_wheelhouse,
        "files": dict(sorted(files.items())),
        "diagnostic_only": False,
        "competition_submission_eligible": False,
        "private_required": True,
    }
    validate_common_manifest(manifest)
    return manifest


def validate_common_manifest(manifest):
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != COMMON_KIND
        or manifest.get("checkpoint_format") not in {"full", "frozen_base_delta"}
        or manifest.get("diagnostic_only") is not False
        or manifest.get("competition_submission_eligible") is not False
        or manifest.get("private_required") is not True
        or not re.fullmatch(
            r"https://github\.com/facebookresearch/dinov3/blob/[0-9a-f]{40}/LICENSE\.md",
            manifest.get("license_source", ""),
        )
        or not SHA.fullmatch(manifest.get("license_sha256", ""))
        or not SHA.fullmatch(manifest.get("encoder_spec_sha256", ""))
        or not isinstance(manifest.get("shared_wheelhouse"), dict)
    ):
        raise ValueError("Exact private non-diagnostic common bundle manifest required")
    files = validate_table(manifest.get("files"))
    if any(role(name) != "common" for name in files):
        raise ValueError("Checkpoint-specific member cannot enter a common bundle")
    weights = "encoder/weights.safetensors" in files
    if (
        files.get("DINOv3-LICENSE.md", {}).get("sha256") != manifest["license_sha256"]
        or files.get("encoder.json", {}).get("sha256") != manifest["encoder_spec_sha256"]
        or "encoder/config.json" not in files
        or "wheelhouse.json" not in files
        or weights != (manifest["checkpoint_format"] == "frozen_base_delta")
        or manifest.get("code_sha256") != code_hash(files)
        or not any(name.startswith("code/rsna/") for name in files)
    ):
        raise ValueError("Common licence/encoder/base/code/wheel members differ from manifest")
    return manifest


def binding(manifest_path, verification_path, dataset_slug):
    """Bind exact common bytes to one full-download private version-1 receipt."""
    manifest = validate_common_manifest(json.loads(Path(manifest_path).read_text()))
    proof = json.loads(Path(verification_path).read_text())
    expected = dict(manifest["files"])
    expected[COMMON_MANIFEST] = {
        "sha256": digest(manifest_path),
        "bytes": Path(manifest_path).stat().st_size,
    }
    if (
        not SLUG.fullmatch(dataset_slug)
        or proof.get("status") != "all_payload_bytes_verified"
        or proof.get("downloaded_all_bytes") is not True
        or proof.get("private") is not True
        or proof.get("version") != 1
        or proof.get("slug") != dataset_slug
        or type(proof.get("id")) is not int
        or proof["id"] < 1
        or proof.get("competition_submission") is not False
        or proof.get("files") != expected
    ):
        raise ValueError("Common bundle needs a complete private version-1 byte proof")
    files = manifest["files"]
    record = {
        "schema_version": 1,
        "dataset_slug": dataset_slug,
        "dataset_id": proof["id"],
        "dataset_version": 1,
        "manifest_sha256": expected[COMMON_MANIFEST]["sha256"],
        "manifest_bytes": expected[COMMON_MANIFEST]["bytes"],
        "files_sha256": object_hash(files),
        "member_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "verification_sha256": digest(verification_path),
        "checkpoint_format": manifest["checkpoint_format"],
        "license_source": manifest["license_source"],
        "license_sha256": manifest["license_sha256"],
        "encoder_spec_sha256": manifest["encoder_spec_sha256"],
        "code_sha256": manifest["code_sha256"],
        "wheelhouse_manifest_sha256": files["wheelhouse.json"]["sha256"],
    }
    validate_binding(record)
    return record, manifest


def validate_binding(record):
    if (
        not isinstance(record, dict)
        or set(record) != BINDING_KEYS
        or record["schema_version"] != 1
        or type(record["dataset_version"]) is not int
        or record["dataset_version"] != 1
        or type(record["dataset_id"]) is not int
        or record["dataset_id"] < 1
        or not isinstance(record["dataset_slug"], str)
        or not SLUG.fullmatch(record["dataset_slug"])
        or record["checkpoint_format"] not in {"full", "frozen_base_delta"}
        or any(
            type(record[k]) is not int or record[k] < 1
            for k in ("manifest_bytes", "member_count", "total_bytes")
        )
        or not all(
            isinstance(record[k], str) and SHA.fullmatch(record[k])
            for k in (
                "manifest_sha256",
                "files_sha256",
                "verification_sha256",
                "license_sha256",
                "encoder_spec_sha256",
                "code_sha256",
                "wheelhouse_manifest_sha256",
            )
        )
    ):
        raise ValueError("Exact versioned common bundle binding required")
    return record


def check_binding(record, manifest_path, verification_path):
    """Recompute the binding from retained local receipts; any drift fails closed."""
    validate_binding(record)
    found, manifest = binding(manifest_path, verification_path, record["dataset_slug"])
    if found != record:
        raise ValueError("Common bundle identity/version/hash binding changed")
    return manifest


def merge(candidate_files, common_manifest, record):
    """Return the exact standalone-equivalent member table, or fail closed."""
    validate_binding(record)
    validate_common_manifest(common_manifest)
    validate_table(candidate_files)
    common = common_manifest["files"]
    if (
        object_hash(common) != record["files_sha256"]
        or len(common) != record["member_count"]
        or common_manifest["checkpoint_format"] != record["checkpoint_format"]
        or common_manifest["license_sha256"] != record["license_sha256"]
        or common_manifest["encoder_spec_sha256"] != record["encoder_spec_sha256"]
        or common_manifest["code_sha256"] != record["code_sha256"]
    ):
        raise ValueError("Common member table differs from its bound version")
    if any(role(name) != "candidate" for name in candidate_files):
        raise ValueError("Checkpoint-only payload carries a common member")
    names = {name.casefold() for name in common}
    if any(name.casefold() in names for name in candidate_files):
        raise ValueError("Conflicting checkpoint/common member path")
    merged = {**common, **candidate_files}
    validate_table(merged)
    return dict(sorted(merged.items()))


def standalone_source(dataset, receipt_path):
    """Bind a standalone payload tree to its original full-download private v1 receipt."""
    dataset = Path(dataset)
    manifest_file = dataset / "bundle-manifest.json"
    manifest = json.loads(manifest_file.read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    expected = dict(validate_table(manifest.get("files")))
    expected["bundle-manifest.json"] = {
        "sha256": digest(manifest_file),
        "bytes": manifest_file.stat().st_size,
    }
    if (
        receipt.get("status") != "all_payload_bytes_verified"
        or receipt.get("downloaded_all_bytes") is not True
        or receipt.get("private") is not True
        or receipt.get("version") != 1
        or receipt.get("competition_submission") is not False
        or type(receipt.get("id")) is not int
        or receipt["id"] < 1
        or receipt.get("slug") != manifest.get("dataset_slug")
        or receipt.get("files") != expected
    ):
        raise ValueError("Standalone payload differs from its full-download private v1 receipt")
    return manifest, {
        "dataset_slug": receipt["slug"],
        "dataset_id": receipt["id"],
        "dataset_version": 1,
        "receipt_sha256": digest(receipt_path),
        "bundle_manifest_sha256": expected["bundle-manifest.json"]["sha256"],
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
    }


def bootstrap_difference(old, regenerated):
    """Names whose bytes differ between an old payload and the candidate's regeneration."""
    validate_table(old)
    validate_table(regenerated)
    return sorted(n for n in set(old) | set(regenerated) if old.get(n) != regenerated.get(n))


def extract_common(
    dataset,
    output,
    *,
    competition,
    shared_wheelhouse,
    checkpoint_format,
    dataset_slug,
    source_receipt,
    regenerated,
):
    """Copy common members out of an already fully verified standalone payload.

    No weights are downloaded: bytes come from an existing prepared Dataset tree,
    bound to its original full-download receipt and re-hashed before and after
    copying. Old bytes are reusable only when they are byte-identical to the
    common table regenerated from the candidate's own integrated sources
    (``workflow.common_table``); no older payload is assumed to match.
    """
    dataset = Path(dataset)
    manifest, source = standalone_source(dataset, source_receipt)
    if (
        manifest.get("common_bundle") is not None
        or manifest.get("diagnostic_only") is not False
        or manifest.get("checkpoint_format", "full") != checkpoint_format
        or manifest.get("shared_wheelhouse") != shared_wheelhouse
    ):
        raise ValueError("Only an exact standalone non-diagnostic payload can seed a common bundle")
    common, _ = split(manifest["files"])
    differing = bootstrap_difference(common, regenerated)
    if differing:
        raise ValueError(
            "Old standalone common bytes differ from the candidate's regenerated members: "
            + ", ".join(differing[:8])
            + (" ..." if len(differing) > 8 else "")
        )
    if not SLUG.fullmatch(dataset_slug) or dataset_slug == manifest.get("dataset_slug"):
        raise ValueError("A distinct explicit common Dataset slug is required")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    for name, item in common.items():
        source_file = dataset / confined(name)
        if source_file.is_symlink() or not source_file.resolve().is_relative_to(dataset.resolve()):
            raise ValueError("Common source is linked or outside its payload")
        target = output / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if digest(source_file) != item["sha256"] or source_file.stat().st_size != item["bytes"]:
            raise ValueError("Standalone payload bytes differ from its manifest")
        shutil.copyfile(source_file, target)
        if digest(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise ValueError("Common copy differs")
    value = build_common_manifest(
        common,
        checkpoint_format=checkpoint_format,
        license_source=manifest["license_source"],
        license_sha256=manifest["license_sha256"],
        encoder_spec_sha256=common["encoder.json"]["sha256"],
        shared_wheelhouse=shared_wheelhouse,
        competition=competition,
    )
    metadata = {
        "id": dataset_slug,
        "title": dataset_slug.split("/")[1].replace("-", " "),
        "licenses": [{"name": "other"}],
    }
    for name, item in ((COMMON_MANIFEST, value), ("dataset-metadata.json", metadata)):
        with (output / name).open("x") as handle:
            json.dump(item, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return value, source


def node_binding(path):
    """Host/uid/device/inode identity of a node-local directory."""
    info = Path(path).stat()
    return {
        "root": str(Path(path)),
        "host": os.uname().nodename,
        "uid": info.st_uid,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def prepare_common_operation(
    dataset,
    directory,
    *,
    competition,
    shared_wheelhouse,
    checkpoint_format,
    dataset_slug,
    source_receipt,
    regenerated,
):
    """Local only: extract one exact common tree and write its reviewed create operation.

    The operation is a separate kind with no checkpoint, ledger entry or
    submission eligibility. It binds the frozen common bytes, the owning
    account, the original standalone receipt and the node holding the tree.
    """
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=False)
    tree = directory / "common"
    try:
        manifest, source = extract_common(
            dataset,
            tree,
            competition=competition,
            shared_wheelhouse=shared_wheelhouse,
            checkpoint_format=checkpoint_format,
            dataset_slug=dataset_slug,
            source_receipt=source_receipt,
            regenerated=regenerated,
        )
    except Exception:
        if not any(directory.iterdir()):
            directory.rmdir()
        raise
    files = manifest["files"]
    operation = {
        "schema_version": 1,
        "kind": COMMON_OPERATION_KIND,
        "account": dataset_slug.split("/")[0],
        "dataset_slug": dataset_slug,
        "dataset_version": 1,
        "private": True,
        "bundle_path": "common",
        "manifest_sha256": digest(tree / COMMON_MANIFEST),
        "manifest_bytes": (tree / COMMON_MANIFEST).stat().st_size,
        "dataset_metadata_sha256": digest(tree / "dataset-metadata.json"),
        "files_sha256": object_hash(files),
        "member_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "regenerated_common_sha256": object_hash(dict(sorted(regenerated.items()))),
        "checkpoint_format": manifest["checkpoint_format"],
        "license_source": manifest["license_source"],
        "license_sha256": manifest["license_sha256"],
        "encoder_spec_sha256": manifest["encoder_spec_sha256"],
        "code_sha256": manifest["code_sha256"],
        "wheelhouse_manifest_sha256": files["wheelhouse.json"]["sha256"],
        "shared_wheelhouse": shared_wheelhouse,
        "source_receipt": source,
        "node": node_binding(tree),
        "checkpoint_entry": None,
        "competition_submission_eligible": False,
        "retry_prohibited": True,
    }
    validate_common_operation(operation)
    with (directory / COMMON_OPERATION).open("x") as handle:
        handle.write(common_operation_text(operation))
        handle.flush()
        os.fsync(handle.fileno())
    return operation


def common_operation_text(operation):
    """Exact deterministic bytes of ``common-operation.json``.

    The shared create intent embeds the operation, so after losing the owning
    node an operator can re-serialize it and must reproduce the original hash.
    """
    return json.dumps(operation, sort_keys=True, indent=2, allow_nan=False) + "\n"


def validate_common_operation(operation):
    """Structure-only check of a common create operation; no tree, node or host access."""
    if (
        not isinstance(operation, dict)
        or set(operation) != COMMON_OPERATION_KEYS
        or operation["schema_version"] != 1
        or operation["kind"] != COMMON_OPERATION_KIND
        or operation["bundle_path"] != "common"
        or operation["dataset_version"] != 1
        or operation["private"] is not True
        or operation["checkpoint_entry"] is not None
        or operation["competition_submission_eligible"] is not False
        or operation["retry_prohibited"] is not True
        or not SLUG.fullmatch(operation["dataset_slug"])
        or operation["dataset_slug"].split("/")[0] != operation["account"]
        or not isinstance(operation["source_receipt"], dict)
        or operation["source_receipt"].get("dataset_slug") == operation["dataset_slug"]
    ):
        raise ValueError("Exact private common Dataset create operation required")
    return operation


def load_common_operation(path):
    """Validate a common create operation against its exact local tree and node."""
    path = Path(path)
    operation = validate_common_operation(json.loads(path.read_text()))
    tree = path.parent / operation["bundle_path"]
    if tree.is_symlink() or node_binding(tree) != operation["node"]:
        raise ValueError("Common tree moved or node identity changed")
    manifest_file = tree / COMMON_MANIFEST
    if (
        digest(manifest_file) != operation["manifest_sha256"]
        or manifest_file.stat().st_size != operation["manifest_bytes"]
        or digest(tree / "dataset-metadata.json") != operation["dataset_metadata_sha256"]
    ):
        raise ValueError("Common manifest or Dataset metadata changed")
    manifest = validate_common_manifest(json.loads(manifest_file.read_text()))
    files = manifest["files"]
    metadata = json.loads((tree / "dataset-metadata.json").read_text())
    if (
        object_hash(files) != operation["files_sha256"]
        or operation["regenerated_common_sha256"] != operation["files_sha256"]
        or len(files) != operation["member_count"]
        or sum(item["bytes"] for item in files.values()) != operation["total_bytes"]
        or manifest["shared_wheelhouse"] != operation["shared_wheelhouse"]
        or files["wheelhouse.json"]["sha256"] != operation["wheelhouse_manifest_sha256"]
        or metadata.get("id") != operation["dataset_slug"]
        or any(
            manifest[k] != operation[k]
            for k in (
                "checkpoint_format",
                "license_source",
                "license_sha256",
                "encoder_spec_sha256",
                "code_sha256",
            )
        )
    ):
        raise ValueError("Common operation differs from its frozen member table")
    return operation, manifest


def savings(files, manifest_bytes=0, candidates=1):
    """Upload bytes for ``candidates`` standalone vs checkpoint-only payloads."""
    common, candidate = split(files)
    common_bytes = sum(item["bytes"] for item in common.values())
    candidate_bytes = sum(item["bytes"] for item in candidate.values()) + manifest_bytes
    standalone = common_bytes + candidate_bytes
    component = common_bytes + candidates * candidate_bytes
    return {
        "candidates": candidates,
        "standalone_payload_bytes_each": standalone,
        "checkpoint_only_payload_bytes_each": candidate_bytes,
        "common_payload_bytes_once": common_bytes,
        "standalone_total_bytes": candidates * standalone,
        "component_total_bytes_including_one_common": component,
        "saved_bytes": candidates * standalone - component,
        "per_additional_candidate_fraction_saved": common_bytes / standalone,
        "common_members": len(common),
        "candidate_members": len(candidate),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Byte comparison from an existing receipt")
    parser.add_argument("verification", type=Path)
    parser.add_argument("--candidates", type=int, default=1)
    args = parser.parse_args()
    proof = json.loads(args.verification.read_text())
    table = {k: v for k, v in proof["files"].items() if k != "bundle-manifest.json"}
    manifest = proof["files"].get("bundle-manifest.json", {}).get("bytes", 0)
    print(json.dumps(savings(table, manifest, args.candidates), indent=2, sort_keys=True))
