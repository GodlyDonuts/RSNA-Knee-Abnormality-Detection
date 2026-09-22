"""Verify local pinned encoders; explicit --download stages missing bytes on Slurm only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SOURCES = {
    "dinov2_b14_reg": ("facebook/dinov2-with-registers-base", "model.safetensors", "safetensors"),
    "convnext_tiny": ("facebook/convnext-tiny-224", "pytorch_model.bin", "pytorch"),
    "dinov3_l16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors", "safetensors"),
}
DINOV3_LICENSE_NAME = "dinov3-license"
DINOV3_LICENSE_URL = "https://ai.meta.com/resources/models-and-libraries/dinov3-license"


class TokenSafeRedirect(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the Hub; signed storage URLs need no bearer."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (
            host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))
        ):
            raise ValueError("unrecognized_model_download_redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and host != "huggingface.co":
            redirected.remove_header("Authorization")
        return redirected


def relative_file(root: Path, value: str, prefix: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("normalized_relative_artifact_path_required")
    if not value.startswith(prefix + "/") or any(ord(c) < 32 for c in value):
        raise ValueError("unexpected_artifact_location")
    result = root / path
    for parent in (result, *result.parents):
        if parent.is_symlink():
            raise ValueError("symlink_artifact_path_rejected")
        if parent == root:
            break
    return result


def verify(path: Path, expected_sha: str, expected_bytes: int) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError("artifact_missing_or_size_mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError("artifact_sha256_mismatch")
    return {"sha256": expected_sha, "bytes": expected_bytes}


def download_missing(
    url: str, destination: Path, sha: str, size: int, token: str | None = None
) -> None:
    """Stream to a private temporary file, verify, then publish without overwrite."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".stage-lock")
    temporary = None
    # Exclusive lock prevents duplicate transfers and preserves uncertain interrupted attempts.
    with lock.open("x"):
        try:
            if destination.exists():
                verify(destination, sha, size)
                return
            descriptor, name = tempfile.mkstemp(prefix=".stage-", dir=destination.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as output:
                request = urllib.request.Request(url, headers={"User-Agent": "rsna-pinned-stage/1"})
                if token is not None:
                    request.add_header("Authorization", "Bearer " + token)
                    response = urllib.request.build_opener(TokenSafeRedirect()).open(
                        request, timeout=60
                    )
                else:
                    response = urllib.request.urlopen(request, timeout=60)
                with response:
                    seen = 0
                    for chunk in iter(lambda: response.read(2**20), b""):
                        seen += len(chunk)
                        if seen > size:
                            raise ValueError("download_exceeds_pinned_size")
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            verify(temporary, sha, size)
            # link is atomic and fails if any destination appeared during the transfer.
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)


def stage(manifest: Path, repo: Path, download: bool = False) -> dict:
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 100_000:
        raise ValueError("small_regular_manifest_required")
    raw = manifest.read_bytes()
    data = json.loads(raw)
    encoder, remote = data["encoder"], data["download"]
    gated = encoder["architecture"] == "dinov3_l16"
    source, filename, state_format = SOURCES[encoder["architecture"]]
    if data.get("schema_version") != 1 or encoder["source_id"] != source:
        raise ValueError("unsupported_official_source")
    revision = encoder["source_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or set(revision) == {"0"}:
        raise ValueError("immutable_source_revision_required")
    if encoder["state_dict_format"] != state_format or remote["weights_filename"] != filename:
        raise ValueError("unexpected_weight_format")
    expected_license = "other" if gated else "apache-2.0"
    if remote["config_filename"] != "config.json" or encoder["license"] != expected_license:
        raise ValueError("unexpected_config_or_license")
    if encoder["transformers_version"] != "5.6.1":
        raise ValueError("unexpected_transformers_pin")
    for key, maximum in (
        ("config", 100_000),
        ("weights", 2_000_000_000 if gated else 1_000_000_000),
    ):
        size = remote[key + "_bytes"]
        if type(size) is not int or not 0 < size <= maximum:
            raise ValueError("invalid_pinned_artifact_size")
        if not re.fullmatch(r"[0-9a-f]{64}", encoder[key + "_sha256"]):
            raise ValueError("invalid_pinned_artifact_sha256")
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError("regular_repository_directory_required")
    root = repo.absolute()
    config = relative_file(root, encoder["config_path"], "configs/competitive/weights")
    weights = relative_file(root, encoder["weights_path"], "artifacts/encoder-weights")
    config_check = verify(config, encoder["config_sha256"], remote["config_bytes"])
    license_check = None
    if gated:
        policy = data.get("license_policy", {})
        if policy.get("name") != DINOV3_LICENSE_NAME or policy.get("url") != DINOV3_LICENSE_URL:
            raise ValueError("dinov3_custom_license_metadata_required")
        if not re.fullmatch(r"[0-9a-f]{64}", str(policy.get("sha256", ""))):
            raise ValueError("exact_dinov3_license_bytes_required")
        if type(policy.get("bytes")) is not int or not 0 < policy["bytes"] <= 100_000:
            raise ValueError("bounded_dinov3_license_bytes_required")
        license_path = relative_file(root, policy["path"], "configs/competitive/weights")
        license_check = verify(license_path, policy["sha256"], policy["bytes"])
    downloaded = False
    if not weights.exists():
        if not download:
            raise ValueError("weights_missing_offline_verification_only")
        if not re.fullmatch(r"[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", "")):
            raise ValueError("download_requires_allocated_slurm_job")
        token = os.environ.get("HF_TOKEN") if gated else None
        if gated and (not token or not token.startswith("hf_") or any(c.isspace() for c in token)):
            raise ValueError("authorized_hf_token_environment_required")
        url = f"https://huggingface.co/{source}/resolve/{revision}/{filename}"
        download_missing(url, weights, encoder["weights_sha256"], remote["weights_bytes"], token)
        downloaded = True
    weight_check = verify(weights, encoder["weights_sha256"], remote["weights_bytes"])
    return {
        "schema_version": 1,
        "status": "verified",
        "read_only": not downloaded,
        "downloaded": downloaded,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_id": source,
        "source_revision": revision,
        "license": expected_license,
        "license_artifact": license_check,
        "access_policy": "Existing authorized token only; no gate acceptance or fallback"
        if gated
        else "public",
        "artifacts": {"config": config_check, "weights": weight_check},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument(
        "--download", action="store_true", help="Explicit opt-in; requires a Slurm allocation"
    )
    args = p.parse_args()
    try:
        print(json.dumps(stage(args.manifest, args.repo, args.download), indent=2))
    except (OSError, ValueError, KeyError, TypeError):
        # HTTP exceptions can contain signed redirect URLs; never serialize them.
        print(json.dumps({"status": "blocked", "reason": "pinned_artifact_staging_failed"}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
