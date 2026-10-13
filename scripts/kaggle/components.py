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
        "competition_submission_eligible": True,
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
        or manifest.get("competition_submission_eligible") is not True
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


def extract_common(
    dataset, output, *, competition, shared_wheelhouse, checkpoint_format, dataset_slug
):
    """Copy common members out of an already fully verified standalone payload.

    No weights are downloaded: bytes come from an existing prepared Dataset tree
    and are re-hashed against its bundle manifest before and after copying.
    """
    dataset = Path(dataset)
    manifest = json.loads((dataset / "bundle-manifest.json").read_text())
    if (
        manifest.get("common_bundle") is not None
        or manifest.get("diagnostic_only") is not False
        or manifest.get("checkpoint_format", "full") != checkpoint_format
        or manifest.get("shared_wheelhouse") != shared_wheelhouse
    ):
        raise ValueError("Only an exact standalone non-diagnostic payload can seed a common bundle")
    common, _ = split(manifest["files"])
    if not SLUG.fullmatch(dataset_slug) or dataset_slug == manifest.get("dataset_slug"):
        raise ValueError("A distinct explicit common Dataset slug is required")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    for name, item in common.items():
        source = dataset / confined(name)
        if source.is_symlink() or not source.resolve().is_relative_to(dataset.resolve()):
            raise ValueError("Common source is linked or outside its payload")
        target = output / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if digest(source) != item["sha256"] or source.stat().st_size != item["bytes"]:
            raise ValueError("Standalone payload bytes differ from its manifest")
        shutil.copyfile(source, target)
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
    return value


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
