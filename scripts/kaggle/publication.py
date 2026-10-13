"""Staged one-shot Dataset transport. No import-time network or credential access."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

API = "https://api.kaggle.com/v1/"
FLOOR = 20 * 1024**3
SHA = re.compile(r"[0-9a-f]{64}")
SLUG = re.compile(r"[a-z0-9_-]+/[a-z0-9-]+")
ALLOWED_SUFFIXES = {".json", ".py", ".md", ".pt", ".safetensors", ".toml", ".txt"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path, value):
    """Durable no-overwrite operation journal; contains no credentials."""
    with Path(path).open("x") as f:
        os.chmod(path, 0o600)
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def confined(name):
    if not isinstance(name, str):
        raise ValueError("Payload path must be a string")
    p = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or p.is_absolute()
        or any(x in ("", ".", "..") for x in name.split("/"))
        or str(p) != name
    ):
        raise ValueError("Noncanonical payload path")
    return p


def expected_files(data, manifest, manifest_name="bundle-manifest.json"):
    if manifest_name not in {"bundle-manifest.json", "common-manifest.json"}:
        raise ValueError("Unknown payload manifest name")
    files = dict(manifest["files"])
    if not files or {"bundle-manifest.json", "common-manifest.json", "dataset-metadata.json"} & set(
        files
    ):
        raise ValueError("Manifest payload membership is invalid")
    for name, item in files.items():
        p = confined(name)
        full = data / p
        if (
            p.suffix.lower() not in ALLOWED_SUFFIXES
            or full.is_symlink()
            or not full.is_file()
            or not full.resolve().is_relative_to(data.resolve())
            or type(item.get("bytes")) is not int
            or item["bytes"] < 0
            or not SHA.fullmatch(item.get("sha256", ""))
            or full.stat().st_size != item["bytes"]
            or digest(full) != item["sha256"]
        ):
            raise ValueError("Unsupported, changed or unsafe payload")
    m = data / manifest_name
    files[m.name] = {"sha256": digest(m), "bytes": m.stat().st_size}
    actual = {str(x.relative_to(data)) for x in data.rglob("*") if x.is_file() or x.is_symlink()}
    if actual != set(files) | {"dataset-metadata.json"}:
        raise ValueError("Missing or extraneous payload")
    return files


def build_zip(data, files, output):
    """One explicit ZIP with complete logical paths; no per-directory repacking."""
    if (
        output.exists()
        or shutil.disk_usage(output.parent).free
        < FLOOR + sum(v["bytes"] for v in files.values()) + 1024**2
    ):
        raise ValueError("Existing archive or insufficient 20 GiB reserve")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, item in sorted(files.items()):
            confined(name)
            path = data / name
            if path.is_symlink() or not path.resolve().is_relative_to(data.resolve()):
                raise ValueError("Archive source is linked or outside payload")
            if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("Archive input changed")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, z.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, 4 * 1024**2)
    validate_zip(output, files)
    return {"sha256": digest(output), "bytes": output.stat().st_size, "members": len(files)}


def validate_zip(output, files):
    with zipfile.ZipFile(output) as z:
        if z.namelist() != sorted(files):
            raise ValueError("ZIP membership changed")
        for info in z.infolist():
            h = hashlib.sha256()
            with z.open(info) as f:
                for block in iter(lambda: f.read(4 * 1024**2), b""):
                    h.update(block)
            if (
                h.hexdigest() != files[info.filename]["sha256"]
                or info.file_size != files[info.filename]["bytes"]
            ):
                raise ValueError("ZIP payload differs")


def require_absent(remote, slug):
    if not SLUG.fullmatch(slug) or slug.split("/")[0] != remote.username:
        raise ValueError("Dataset owner differs from authenticated account")
    # Do not interpret GetDataset403/404 or dataset_status as authoritative absence.
    refs = set()
    for page in range(1, 101):
        result = remote.call(
            "datasets.DatasetApiService",
            "ListDatasets",
            {"group": "DATASET_SELECTION_GROUP_MY", "page": page},
        )
        rows = result.get("datasets", [])
        if not isinstance(rows, list):
            raise ValueError("Invalid owned Dataset page")
        if not rows:
            return {
                "method": "complete_authenticated_owned_listing",
                "pages": page,
                "count": len(refs),
                "refs_sha256": hashlib.sha256(canonical(sorted(refs))).hexdigest(),
            }
        for row in rows:
            ref = row.get("ref")
            if (
                not isinstance(ref, str)
                or not SLUG.fullmatch(ref)
                or ref.split("/")[0] != remote.username
                or ref in refs
                or ref == slug
            ):
                raise ValueError("Existing target or ambiguous owned Dataset listing")
            refs.add(ref)
    raise ValueError("Owned Dataset listing did not terminate")


def metadata(remote, slug):
    value = remote.dataset(slug)
    if (
        value.get("isPrivate") is not True
        or value.get("currentVersionNumber") != 1
        or type(value.get("id")) is not int
        or value["id"] <= 0
    ):
        raise ValueError("Dataset is not private dedicated version 1")
    return {"id": value["id"], "version": 1, "slug": slug, "private": True}


def list_files(remote, slug):
    owner, name = slug.split("/")
    found, seen, token = {}, set(), ""
    for _ in range(100):
        result = remote.call(
            "datasets.DatasetApiService",
            "ListDatasetFiles",
            {
                "ownerSlug": owner,
                "datasetSlug": name,
                "datasetVersionNumber": 1,
                "pageSize": 100,
                "pageToken": token,
            },
        )
        rows = result.get("datasetFiles", [])
        if not isinstance(rows, list):
            raise ValueError("Invalid file list")
        for row in rows:
            filename = row.get("name")
            confined(filename)
            # Protobuf can omit a zero scalar. Zero-length files are legitimate.
            size = row.get("totalBytes", 0)
            if filename in found or type(size) is not int or size < 0:
                raise ValueError("Duplicate file or invalid size")
            found[filename] = size
        token = result.get("nextPageToken", "")
        if not isinstance(token, str) or token in seen:
            raise ValueError("Invalid/repeated file pagination token")
        if not token:
            return found
        seen.add(token)
    raise ValueError("File list exceeded finite pagination")


def map_members(actual, expected, manifest_name="bundle-manifest.json"):
    candidates = [n for n in actual if n == manifest_name or n.endswith("/" + manifest_name)]
    if len(candidates) != 1 or manifest_name not in expected:
        raise ValueError("A unique mounted bundle manifest is required")
    prefix = candidates[0][: -len(manifest_name)]
    if any(not name.startswith(prefix) for name in actual):
        raise ValueError("Mixed remote archive wrappers")
    mapping = {name[len(prefix) :]: name for name in actual}
    if set(mapping) != set(expected):
        raise ValueError("Remote logical membership differs")
    if any(actual[remote_name] != expected[name]["bytes"] for name, remote_name in mapping.items()):
        raise ValueError("Remote file size differs")
    return mapping


def stream_file(remote, slug, name, expected, max_seconds=900):
    """Stream immutable file bytes, no disk writes, no token on redirect hosts."""
    import requests

    owner, dataset = slug.split("/")
    remote.session.cookies.clear()
    started = time.monotonic()
    response = remote.session.post(
        API + "datasets.DatasetApiService/DownloadDataset",
        json={
            "ownerSlug": owner,
            "datasetSlug": dataset,
            "datasetVersionNumber": 1,
            "fileName": name,
        },
        timeout=(15, 180),
        allow_redirects=False,
        stream=True,
    )
    try:
        for _ in range(6):
            if time.monotonic() - started > max_seconds:
                raise ValueError("File download exceeded total deadline")
            if response.status_code == 200:
                h, count = hashlib.sha256(), 0
                for block in response.iter_content(4 * 1024**2):
                    count += len(block)
                    if count > expected["bytes"] or time.monotonic() - started > max_seconds:
                        raise ValueError("File download exceeds size/deadline bound")
                    h.update(block)
                if count != expected["bytes"] or h.hexdigest() != expected["sha256"]:
                    raise ValueError("Remote file bytes differ; transport wrappers are not guessed")
                return {"sha256": h.hexdigest(), "bytes": count}
            if response.status_code not in (301, 302, 303, 307, 308):
                raise ValueError("Immutable download unavailable")
            target = urljoin(response.url, response.headers.get("Location", ""))
            parsed = urlparse(target)
            if (
                parsed.scheme != "https"
                or parsed.username
                or parsed.password
                or parsed.port not in (None, 443)
                or not (
                    parsed.hostname == "storage.googleapis.com"
                    or (parsed.hostname or "").endswith(".kaggleusercontent.com")
                )
            ):
                raise ValueError("Unapproved download redirect")
            response.close()
            # Standalone request carries no API Authorization header or cookies.
            response = requests.get(target, timeout=(15, 180), allow_redirects=False, stream=True)
        raise ValueError("Download exceeded redirect bound")
    finally:
        response.close()


def reconcile(remote, slug, files, stream=stream_file, manifest_name="bundle-manifest.json"):
    before = metadata(remote, slug)
    owner, name = slug.split("/")
    ready = remote.call(
        "datasets.DatasetApiService", "GetDatasetStatus", {"ownerSlug": owner, "datasetSlug": name}
    )
    if ready.get("status") not in ("READY", 5):
        raise ValueError("Dataset ingestion is not ready")
    mapping = map_members(list_files(remote, slug), files, manifest_name)
    verified = {
        logical: stream(remote, slug, path, files[logical])
        for logical, path in sorted(mapping.items())
    }
    if metadata(remote, slug) != before:
        raise ValueError("Dataset identity changed during verification")
    return {
        "schema_version": 1,
        "status": "all_payload_bytes_verified",
        **before,
        "files": verified,
        "remote_names": mapping,
        "downloaded_all_bytes": True,
        "competition_submission": False,
    }


def sdk_create_once(token, slug, title, upload):
    """Final CreateDataset is a direct typed RPC, never high-level with_retry."""
    import requests

    if {x.name for x in upload.iterdir()} != {"bundle.zip"}:
        raise ValueError("Upload directory must contain only the explicit ZIP")
    if {name: importlib.metadata.version(name) for name in ("kaggle", "kagglesdk")} != {
        "kaggle": "2.2.3",
        "kagglesdk": "0.1.30",
    }:
        raise ValueError("Unreviewed SDK version")
    old = os.environ.get("KAGGLE_API_TOKEN")
    old_send = requests.Session.send

    def bounded_send(self, request, **kwargs):
        kwargs["timeout"] = (15, 180)
        return old_send(self, request, **kwargs)

    requests.Session.send = bounded_send
    os.environ["KAGGLE_API_TOKEN"] = token
    try:
        with (
            open(os.devnull, "w") as silence,
            contextlib.redirect_stdout(silence),
            contextlib.redirect_stderr(silence),
        ):
            from kaggle.api.kaggle_api_extended import (
                ApiBlobType,
                ApiCreateDatasetRequest,
                KaggleApi,
                ResumableUploadContext,
            )

            api = KaggleApi()
            api.authenticate()
            api.with_retry = lambda f: f
            req = ApiCreateDatasetRequest()
            req.owner_slug, req.slug = slug.split("/")
            req.title, req.license_name, req.is_private = title, "other", True
            req.files = []
            with ResumableUploadContext() as context:
                api.upload_files(
                    req,
                    None,
                    str(upload),
                    ApiBlobType.DATASET,
                    context,
                    quiet=True,
                    dir_mode="skip",
                )
                with api.build_kaggle_client() as client:
                    result = client.datasets.dataset_api_client.create_dataset(req)
            if getattr(result, "error", None):
                raise ValueError("Dataset create result is uncertain")
            return {
                "returned": True,
                "status": str(getattr(result, "status", "")),
                "final_create_rpc_calls": 1,
                "private_requested": True,
                "sdk_convert_to_csv_argument_used": False,
            }
    finally:
        requests.Session.send = old_send
        if old is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = old


def create_once(remote, token, slug, files, upload, journal, context, create=sdk_create_once):
    if (journal / "intent.json").exists():
        raise ValueError("Prior intent exists; read-only reconciliation only")
    validate_zip(upload / "bundle.zip", files)
    absence = require_absent(remote, slug)
    if digest(upload / "bundle.zip") != context["archive_sha256"]:
        raise ValueError("Transport archive changed")
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    exclusive_json(
        journal / "intent.json",
        {
            "schema_version": 1,
            "kind": "one_private_dataset_create",
            "slug": slug,
            "private": True,
            "version": 1,
            "files": files,
            "context": context,
            "absence_proof": absence,
            "retry_prohibited": True,
            "publisher_sha256": digest(Path(__file__)),
            "competition_submission": False,
        },
    )
    try:
        result = create(token, slug, slug.split("/")[1].replace("-", " "), upload)
        exclusive_json(journal / "response.json", result)
    except Exception as exc:
        exclusive_json(
            journal / "uncertain.json",
            {"status": "uncertain", "error_type": type(exc).__name__, "retry_prohibited": True},
        )
        raise RuntimeError("Create outcome uncertain; read-only reconciliation required") from None
    return result


def _worker():
    import sys

    if sys.argv[1:] != ["--sdk-worker"]:
        raise ValueError("SDK worker requires explicit mode")
    value = json.load(sys.stdin)
    if set(value) != {"token", "slug", "title", "upload", "archive_sha256"}:
        raise ValueError("Invalid SDK worker input")
    upload = Path(value["upload"])
    if digest(upload / "bundle.zip") != value["archive_sha256"]:
        raise ValueError("SDK worker archive changed")
    result = sdk_create_once(value["token"], value["slug"], value["title"], upload)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        _worker()
    except Exception as exc:
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(exc).__name__, "automatic_retry": False}
            )
        )
        raise SystemExit(1) from None
