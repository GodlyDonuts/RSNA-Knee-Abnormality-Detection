"""Guarded Kaggle code-competition queue; never uploads a local sample CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

COMPETITION = "rsna-knee-abnormality-detection"
CADENCE = (2, 4, 6, 8, 10)
API = "https://api.kaggle.com/v1/"
FLOOR = 20 * 1024**3
ROOT = Path(__file__).resolve().parents[2]
SHA = re.compile(r"[0-9a-f]{64}")
SLUG = re.compile(r"[a-z0-9_-]+/[a-z0-9-]+")
RUNTIME_IMAGE = re.compile(r"gcr\.io/kaggle(?:-private-byod|-images)/python@sha256:[0-9a-f]{64}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def new_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def relative_file(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError("Payload paths must be relative and confined")
    full = root / p
    if full.is_symlink() or not full.is_file() or not full.resolve().is_relative_to(root.resolve()):
        raise ValueError("Payload file is missing, linked, or outside its root")
    return full


class RemoteError(RuntimeError):
    """Deliberately excludes response text, authorization headers and signed URLs."""


class Remote:
    def __init__(self, env_file=ROOT / ".env"):
        # Parse exactly one value, with no dependency on importing the auto-auth CLI.
        if env_file.stat().st_mode & 0o077:
            raise ValueError("Credential file must have no group/other permissions")
        values = {}
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "KAGGLE_API_TOKEN":
                values[key.strip()] = value.strip().strip("\"'")
        token = values.get("KAGGLE_API_TOKEN", "")
        if not token.startswith("KGAT_"):
            raise ValueError("A Kaggle access token is required")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer " + token})
        # Requests' default adapter has no automatic retries, including POSTs.
        identity = self.call("security.OAuthService", "IntrospectToken", {"token": token})
        if identity.get("active") is not True or not identity.get("username"):
            raise RemoteError("Kaggle token identity is inactive or unknown")
        self.username = identity["username"]

    def call(self, service, method, body):
        try:
            # Kaggle's CLI creates a fresh HTTP session per API operation. Do not
            # mix website response cookies with subsequent bearer-only RPCs.
            self.session.cookies.clear()
            response = self.session.post(
                API + service + "/" + method,
                json=body,
                timeout=(10, 60),
                allow_redirects=False,
            )
            if response.status_code != 200:
                raise RemoteError(f"Kaggle {method} HTTP {response.status_code}")
            value = response.json()
            if not isinstance(value, dict) or value.get("code", 0) >= 400:
                raise RemoteError(f"Kaggle {method} rejected its request")
            return value
        except (requests.RequestException, ValueError):
            raise RemoteError(f"Kaggle {method} transport/result uncertain") from None

    def competition(self, method, extra=None):
        return self.call(
            "competitions.CompetitionApiService",
            method,
            {"competitionName": COMPETITION, **(extra or {})},
        )

    def account(self):
        competition = self.competition("GetCompetition")
        limits = self.competition("GetSubmissionLimits")
        # Protobuf JSON omits zero/default fields. Allowance itself must be explicit
        # unless zero, which conservatively blocks submission.
        count = limits.get("numToday", 0)
        remaining = limits.get("numAllowedNow", 0)
        if type(count) is not int or type(remaining) is not int or min(count, remaining) < 0:
            raise RemoteError("Invalid server quota result")
        return {
            "checked_at": utcnow(),
            "account": self.username,
            "competition": COMPETITION,
            "entered": competition.get("userHasEntered") is True,
            "competition_id": competition.get("id"),
            "num_today": count,
            "num_total": limits.get("numTotal", 0),
            "num_allowed_now": remaining,
            "limited_by_total": limits.get("limitedByTotal", False),
            "reset_at": None,
            "reset_verified": False,
            "quota_source": API + "competitions.CompetitionApiService/GetSubmissionLimits",
        }

    def submissions(self):
        result = []
        token = ""
        for _ in range(100):
            page = self.competition(
                "ListSubmissions", {"page": -1, "pageSize": 100, "pageToken": token}
            )
            result.extend(page.get("submissions", []))
            token = page.get("nextPageToken", "")
            if not token:
                return result
        raise RemoteError("Submission pagination exceeded its finite bound")

    def require_absent_kernel(self, slug):
        owner, name = slug.split("/")
        if owner != self.username:
            raise ValueError("Dedicated notebook must belong to the authenticated account")
        try:
            self.call(
                "kernels.KernelsApiService", "GetKernel", {"userName": owner, "kernelSlug": name}
            )
        except RemoteError as exc:
            if str(exc) == "Kaggle GetKernel HTTP 404":
                return {"method": "GetKernel404"}
            if str(exc) != "Kaggle GetKernel HTTP 403":
                raise
        else:
            raise ValueError("Dedicated notebook already exists; do not overwrite")
        # Kaggle hides some absent private resources behind403. That response
        # alone never proves absence; exhaust the authenticated profile listing.
        refs = []
        unnamed = 0
        for page in range(1, 101):
            result = self.call(
                "kernels.KernelsApiService",
                "ListKernels",
                {
                    "group": "PROFILE",
                    "page": page,
                    "pageSize": 100,
                    "language": "all",
                    "kernelType": "all",
                    "outputType": "all",
                },
            )
            rows = result.get("kernels", [])
            if not isinstance(rows, list):
                raise RemoteError("Owned notebook listing is invalid")
            if not rows:
                proof = {
                    "method": "complete_authenticated_profile_after403",
                    "pages": page,
                    "refs_sha256": object_hash(sorted(refs)),
                    "count": len(refs),
                }
                if not unnamed:
                    return proof
                # Kaggle can return a private notebook placeholder with no ref.
                # Search the exact slug only after verifying that the same
                # authenticated search finds a known owned notebook. A matching
                # placeholder remains ambiguous and blocks a new SaveKernel.
                if not refs:
                    raise RemoteError("Owned notebook listing has only unnamed identities")

                def search(query):
                    found = []
                    for search_page in range(1, 101):
                        response = self.call(
                            "kernels.KernelsApiService",
                            "ListKernels",
                            {
                                "group": "PROFILE",
                                "page": search_page,
                                "pageSize": 100,
                                "language": "all",
                                "kernelType": "all",
                                "outputType": "all",
                                "search": query,
                            },
                        )
                        matches = response.get("kernels", [])
                        if not isinstance(matches, list):
                            raise RemoteError("Owned notebook search is invalid")
                        if not matches:
                            return found, search_page
                        for match in matches:
                            ref = match.get("ref")
                            if (
                                not isinstance(ref, str)
                                or not ref.startswith(owner + "/")
                                or ref in found
                            ):
                                raise RemoteError("Owned notebook search has ambiguous identities")
                            found.append(ref)
                    raise RemoteError("Owned notebook search pagination did not terminate")

                control = sorted(refs)[0]
                control_refs, control_pages = search(control)
                if control not in control_refs:
                    raise RemoteError("Owned notebook search missed a known notebook")
                target_refs, target_pages = search(slug)
                if slug in target_refs:
                    raise ValueError("Dedicated notebook already exists; do not overwrite")
                return {
                    **proof,
                    "method": "complete_authenticated_profile_search_after403",
                    "unnamed_count": unnamed,
                    "positive_control_ref": control,
                    "positive_control_pages": control_pages,
                    "target_search_pages": target_pages,
                    "target_refs_sha256": object_hash(sorted(target_refs)),
                }
            for row in rows:
                ref = row.get("ref")
                if ref is None:
                    unnamed += 1
                    continue
                if not isinstance(ref, str) or not ref.startswith(owner + "/") or ref in refs:
                    raise RemoteError("Owned notebook listing has invalid or repeated identities")
                if ref == slug:
                    raise ValueError("Dedicated notebook already exists; do not overwrite")
                refs.append(ref)
        raise RemoteError("Owned notebook pagination did not terminate")

    def kernel(self, method, slug, version):
        owner, name = slug.split("/")
        body = {"userName": owner, "kernelSlug": name}
        before = self.call("kernels.KernelsApiService", "GetKernel", body)
        if before.get("metadata", {}).get("currentVersionNumber") != version:
            raise ValueError("Kernel current version differs from the immutable submitted version")
        if method == "GetKernel":
            return before
        result = self.call("kernels.KernelsApiService", method, body)
        after = self.call("kernels.KernelsApiService", "GetKernel", body)
        if after.get("metadata", {}).get("currentVersionNumber") != version:
            raise ValueError("Kernel version changed during status inspection")
        return result

    def dataset(self, slug):
        owner, name = slug.split("/")
        return self.call(
            "datasets.DatasetApiService", "GetDataset", {"ownerSlug": owner, "datasetSlug": name}
        )

    def output_json(self, slug, version, name):
        owner, kernel = slug.split("/")
        try:
            self.session.cookies.clear()
            response = self.session.post(
                API + "kernels.KernelsApiService/DownloadKernelOutput",
                json={
                    "ownerSlug": owner,
                    "kernelSlug": kernel,
                    "versionNumber": version,
                    "filePath": name,
                },
                timeout=(10, 60),
                allow_redirects=False,
                stream=True,
            )
            for _ in range(5):
                with response:
                    if response.status_code == 200:
                        content = bytearray()
                        for block in response.iter_content(65536):
                            content.extend(block)
                            if len(content) > 1024 * 1024:
                                raise RemoteError("Kernel receipt exceeds bounded JSON size")
                        return json.loads(content)
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        raise RemoteError("Cannot retrieve versioned kernel receipt")
                    url = urljoin(response.url, response.headers.get("Location", ""))
                parsed = urlparse(url)
                host = parsed.hostname or ""
                if parsed.scheme != "https" or not (
                    host
                    in {
                        "storage.googleapis.com",
                        "api.kaggle.com",
                        "www.kaggle.com",
                        "www.kaggleusercontent.com",
                    }
                    or host.endswith(".googleusercontent.com")
                ):
                    raise RemoteError("Unrecognized Kaggle output redirect host")
                # Signed storage requests never carry Kaggle authorization.
                session = self.session if host == "api.kaggle.com" else requests
                response = session.get(url, timeout=(10, 60), allow_redirects=False, stream=True)
            response.close()
            raise RemoteError("Too many kernel output redirects")
        except (requests.RequestException, ValueError):
            raise RemoteError("Kernel output verification failed") from None


@contextmanager
def ledger(path, account):
    """One shared file per account, all runs/folds; atomic intents survive a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    os.chmod(path, 0o600)
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS entries (sha TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at TEXT, sha TEXT, event TEXT)"
    )
    db.commit()
    db.execute("BEGIN IMMEDIATE")
    try:
        expected = canonical({"account": account, "competition": COMPETITION}).decode()
        previous = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if previous and previous[0] != expected:
            raise ValueError("Ledger belongs to another account or competition")
        db.execute("INSERT OR IGNORE INTO meta VALUES ('identity', ?)", (expected,))
        db.commit()
        yield db
    finally:
        db.close()


def read_entry(db, sha):
    row = db.execute("SELECT record FROM entries WHERE sha=?", (sha,)).fetchone()
    if row is None:
        raise ValueError("Checkpoint is not queued")
    return json.loads(row[0])


def save_entry(db, entry, event):
    sha = entry["checkpoint_sha256"]
    db.execute("INSERT OR REPLACE INTO entries VALUES (?, ?)", (sha, canonical(entry).decode()))
    db.execute("INSERT INTO events(at,sha,event) VALUES (?,?,?)", (utcnow(), sha, event))
    db.commit()


def verify_account(db, remote):
    expected = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    if remote.username != expected["account"]:
        raise ValueError("Authenticated account differs from the global ledger identity")


def rollback_on_error(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            for value in args:
                if isinstance(value, sqlite3.Connection):
                    value.rollback()
            raise

    return wrapped


def validate_snapshot(sidecar, repo=ROOT, fitting_admission=None):
    """Read trusted weights_only snapshot; validate internal state as well as sidecar."""
    import importlib.util

    import torch

    sys.path.insert(0, str(repo))
    from rsna.competitive.infer import validate_inference_checkpoint

    meta = json.loads(sidecar.read_text())
    baseline = meta.get("status") == "explicit_epoch1_baseline_snapshot"
    if (
        (meta.get("status") != "scheduled_epoch_snapshot" and not baseline)
        or type(meta.get("epoch")) is not int
        or (baseline and meta["epoch"] != 1)
        or (
            not baseline
            and meta["epoch"]
            not in (range(1, 11) if meta.get("submission_every_epochs") == 1 else CADENCE)
        )
        or type(meta.get("submission_every_epochs")) is not int
        or meta.get("submission_every_epochs") not in ({2} if baseline else {1, 2})
        or meta.get("submission_eligible") is not True
        or meta.get("diagnostic_only") is not False
        or type(meta.get("optimizer_steps")) is not int
        or meta["optimizer_steps"] <= 0
    ):
        raise ValueError("Only trained, completed, non-diagnostic authorized epochs are eligible")
    path = relative_file(sidecar.parent, meta["checkpoint_file"])
    if digest(path) != meta.get("checkpoint_sha256") or path.stat().st_size != meta.get(
        "checkpoint_bytes"
    ):
        raise ValueError("Epoch snapshot hash/size changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") not in {1, 2}:
        raise ValueError("Unknown inference checkpoint storage schema")
    if baseline:
        proof_path = relative_file(sidecar.parent, meta.get("baseline_proof_file", ""))
        if digest(proof_path) != meta.get("baseline_proof_sha256"):
            raise ValueError("Baseline replay proof changed")
        proof = json.loads(proof_path.read_text())
        if (
            proof.get("status") != "completed_epoch1_replay"
            or proof.get("checkpoint_sha256") != meta["checkpoint_sha256"]
            or proof.get("checkpoint_bytes") != meta["checkpoint_bytes"]
            or proof.get("optimizer_steps") != meta["optimizer_steps"]
            or proof.get("source_checkpoint_sha256")
            != "b50ae5591b7b5b02f36423676f1f7fcbedacd69c8d848172bee459e4ae8d044a"
        ):
            raise ValueError("Baseline proof does not bind the replay")
        adapter = Path(__file__).with_name("baseline_infer.py")
        spec = importlib.util.spec_from_file_location("rsna_baseline_infer", adapter)
        if spec is None or spec.loader is None:
            raise ValueError("Baseline inference adapter is missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        binding = module.validate_baseline(checkpoint)
    elif fitting_admission is not None:
        spec = importlib.util.spec_from_file_location(
            "rsna_reviewed_infer", Path(__file__).with_name("reviewed_infer.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        binding = module.validate_reviewed(
            checkpoint, module.read_admission(fitting_admission), meta["checkpoint_sha256"]
        )
    else:
        binding = validate_inference_checkpoint(checkpoint, allow_scheduled_epoch=True)
    config = binding["config"]
    if checkpoint["schema_version"] == 2 and (
        config.get("checkpoint_format") != "frozen_base_delta"
        or checkpoint.get("model_storage", {}).get("format") != "frozen_base_delta_v1"
        or checkpoint["model_storage"].get("encoder_provenance") != config.get("encoder")
    ):
        raise ValueError("Delta storage does not bind the original encoder provenance")
    code = {
        str(p.relative_to(repo / "rsna")): digest(p) for p in sorted((repo / "rsna").rglob("*.py"))
    }
    if binding.get("code_sha256") != code or object_hash(binding) != meta.get("binding_sha256"):
        raise ValueError("Snapshot code or binding differs from the exact trained package")
    if (
        checkpoint.get("scheduled_epoch") is not (not baseline)
        or checkpoint.get("epoch") != meta["epoch"]
        or checkpoint.get("next_update") != 0
        or checkpoint.get("step") != meta["optimizer_steps"]
        or binding["release"]["binding_sha256"] != meta.get("release_binding_sha256")
        or any(config.get(k) != meta.get(k) for k in ("fold", "seed", "submission_every_epochs"))
        or config.get("encoder", {}).get("architecture") not in {"dinov3_b16", "dinov3_l16"}
    ):
        raise ValueError("Snapshot provenance/eligibility does not match its contents")
    return meta, binding, path


@rollback_on_error
def enqueue(db, sidecar, repo=ROOT, fitting_admission=None):
    meta, _, _ = (
        validate_snapshot(sidecar, repo)
        if fitting_admission is None
        else validate_snapshot(sidecar, repo, fitting_admission=fitting_admission)
    )
    db.execute("BEGIN IMMEDIATE")
    sha = meta["checkpoint_sha256"]
    row = db.execute("SELECT record FROM entries WHERE sha=?", (sha,)).fetchone()
    if row:
        db.rollback()
        return json.loads(row[0])
    record = {
        **meta,
        "snapshot_sidecar_sha256": digest(sidecar),
        "state": "queued",
        "queued_at": utcnow(),
    }
    save_entry(db, record, "queued immutable epoch")
    return record


def verify_payload(directory, manifest):
    files = manifest.get("files", {})
    if not files:
        raise ValueError("Empty inference bundle")
    for name, record in files.items():
        path = relative_file(directory, name)
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Inference bundle file changed")


def validate_delta_base(checkpoint_file, encoder_spec, binding):
    """Reconstruct the saved partition from local checked weights before packaging."""
    import torch

    from rsna.competitive.adapters import configure_encoder_training
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta
    from rsna.competitive.model import CompetitiveStudyNet

    relation_kwargs = {}
    if binding["config"].get("relation") is not None:
        from rsna.competitive.relations import relation_model_kwargs

        relation_kwargs = relation_model_kwargs(binding["config"]["relation"])

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        encoder, dimension, provenance = create_encoder(encoder_spec)
        if provenance != binding["config"]["encoder"]:
            raise ValueError("Original encoder source differs from the delta training binding")
        config = binding["config"]
        configure_encoder_training(encoder, config["freeze_encoder"])
        model = CompetitiveStudyNet(
            encoder,
            dimension,
            pooling=config["pooling"],
            hidden_dim=config["hidden_dim"],
            encode_chunk_size=config["encode_chunk_size"],
            slice_context=config.get("slice_context", "none"),
            **({"plane_memory": config["plane_memory"]} if config.get("plane_memory") else {}),
            **({"patch_summary": config["patch_summary"]} if config.get("patch_summary") else {}),
            **relation_kwargs,
        )
        guard = FrozenBaseDelta(model, provenance)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        guard.restore(model, checkpoint)
    finally:
        torch.set_num_threads(previous_threads)


def prepare_bundle(
    sidecar,
    encoder_spec,
    wheelhouse,
    output,
    dataset_slug,
    kernel_slug,
    license_file,
    license_source,
    license_sha256,
    repo=ROOT,
    shared_wheelhouse=None,
    runtime_image=None,
    fitting_admission=None,
    inference_engine="legacy",
):
    """Local only. Requires real epoch + completed Linux wheel lock; no placeholders."""
    if inference_engine not in {"legacy", "resident_two_gpu_fp32"}:
        raise ValueError("Unadmitted inference execution engine")
    if runtime_image is not None and not RUNTIME_IMAGE.fullmatch(runtime_image):
        raise ValueError("An exact Kaggle runtime image digest is required")
    if not SLUG.fullmatch(dataset_slug) or not SLUG.fullmatch(kernel_slug):
        raise ValueError("Use explicit owner/slug identifiers")
    meta, binding, checkpoint = (
        validate_snapshot(sidecar, repo)
        if fitting_admission is None
        else validate_snapshot(sidecar, repo, fitting_admission=fitting_admission)
    )
    delta = binding["config"].get("checkpoint_format") == "frozen_base_delta"
    if (
        meta["checkpoint_sha256"][:12] not in dataset_slug
        or meta["checkpoint_sha256"][:12] not in kernel_slug
    ):
        raise ValueError("Dedicated Dataset/kernel slugs must include the checkpoint SHA prefix")
    encoder = json.loads(encoder_spec.read_text())
    if not re.fullmatch(
        r"https://github\.com/facebookresearch/dinov3/blob/[0-9a-f]{40}/LICENSE\.md", license_source
    ):
        raise ValueError("Pin the official full DINOv3 license to an immutable source commit")
    if not SHA.fullmatch(license_sha256) or digest(license_file) != license_sha256:
        raise ValueError("Full DINOv3 license text hash differs from its source record")
    wheel_manifest = json.loads((wheelhouse / "wheelhouse.json").read_text())
    if wheel_manifest.get("platform") != "linux_x86_64" or not wheel_manifest.get("cuda"):
        raise ValueError("A validated Linux CUDA wheelhouse is required")
    if (
        "requirements.lock" not in wheel_manifest.get("files", {})
        or not any(name.endswith(".whl") for name in wheel_manifest.get("files", {}))
        or wheel_manifest.get("offline_install_verified") is not True
    ):
        raise ValueError("Wheelhouse needs a completed offline installation acceptance")
    verify_payload(wheelhouse, wheel_manifest)
    acceptance = json.loads(relative_file(wheelhouse, "acceptance.json").read_text())
    if (
        acceptance.get("status") != "cpu_offline_install_passed"
        or wheel_manifest.get("pyproject_sha256") != digest(repo / "pyproject.toml")
        or acceptance.get("pyproject_sha256") != wheel_manifest["pyproject_sha256"]
        or "acceptance.json" not in wheel_manifest["files"]
    ):
        raise ValueError("Offline dependency acceptance does not bind the current project pins")
    lock = relative_file(wheelhouse, "requirements.lock")
    if not lock.read_text().strip() or "--hash=sha256:" not in lock.read_text():
        raise ValueError("A complete hash-pinned dependency lock is required")
    shared = None
    if shared_wheelhouse is not None:
        shared = json.loads(shared_wheelhouse.read_text())
        validate_shared_wheelhouse(shared)
        if (
            shared["manifest_sha256"] != digest(wheelhouse / "wheelhouse.json")
            or shared["pyproject_sha256"] != digest(repo / "pyproject.toml")
            or shared["dataset_slug"] == dataset_slug
        ):
            raise ValueError("Shared wheelhouse binding differs from this accepted runtime")
    sources = {
        "checkpoint.pt": checkpoint,
        "snapshot.json": sidecar,
        "DINOv3-LICENSE.md": license_file,
    }
    baseline = meta.get("status") == "explicit_epoch1_baseline_snapshot"
    if baseline:
        sources["baseline-proof.json"] = relative_file(sidecar.parent, meta["baseline_proof_file"])
    for path in sorted((repo / "rsna").rglob("*.py")):
        sources["code/rsna/" + str(path.relative_to(repo / "rsna"))] = path
    sources["code/chunked_infer.py"] = Path(__file__).with_name("chunked_infer.py")
    if inference_engine == "resident_two_gpu_fp32":
        if baseline:
            raise ValueError("Resident execution does not broaden epoch-one admission")
        for name in (
            "resident_entry.py",
            "resident_infer.py",
            "resident_pool.py",
            "resident_worker.py",
        ):
            sources["code/" + name] = Path(__file__).with_name(name)
    if baseline:
        sources["code/baseline_infer.py"] = Path(__file__).with_name("baseline_infer.py")
    if fitting_admission is not None:
        if baseline:
            raise ValueError("Reviewed supervision cannot use the epoch-one baseline adapter")
        sources["fitting-admission.json"] = fitting_admission
        sources["code/reviewed_infer.py"] = Path(__file__).with_name("reviewed_infer.py")
    # Delta storage needs the exact original base; full storage is self-contained.
    local_encoder = dict(encoder)
    for kind in ("config", "weights") if delta else ("config",):
        path = Path(encoder[f"{kind}_path"])
        if not path.is_absolute():
            path = encoder_spec.parent / path
        if digest(path) != encoder[f"{kind}_sha256"]:
            raise ValueError("Encoder file differs from its provenance")
        name = "encoder/" + kind + path.suffix
        sources[name] = path
        local_encoder[f"{kind}_path"] = str(path.resolve())
        encoder[f"{kind}_path"] = name
    if not delta:
        encoder.pop("weights_path", None)
    expected_encoder = {k: v for k, v in encoder.items() if not k.endswith("_path")}
    trained_encoder = binding["config"]["encoder"]
    if any(
        trained_encoder.get(k) != v
        for k, v in expected_encoder.items()
        if not (delta and k == "adapter")
    ):
        raise ValueError("Bundled encoder differs from training")
    if delta:
        validate_delta_base(checkpoint, local_encoder, binding)
    if shared is None:
        for name in wheel_manifest["files"]:
            sources["wheels/" + name] = relative_file(wheelhouse, name)
    if shutil.disk_usage(output.parent).free < FLOOR + sum(
        p.stat().st_size for p in sources.values()
    ):
        raise ValueError("Bundle would violate the 20 GiB free-space floor")
    output.mkdir(exist_ok=False)
    data = output / "dataset"
    data.mkdir()
    for name, source in sources.items():
        target = data / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    new_json(data / "encoder.json", encoder)
    new_json(data / "wheelhouse.json", wheel_manifest)
    files = {
        str(p.relative_to(data)): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(data.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema_version": 1,
        "competition": COMPETITION,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "epoch": meta["epoch"],
        "baseline_epoch_one": baseline,
        "snapshot_sidecar_sha256": digest(sidecar),
        "release_binding_sha256": meta["release_binding_sha256"],
        "dataset_slug": dataset_slug,
        "dataset_version": 1,
        "training_binding_sha256": meta["binding_sha256"],
        "license_source": license_source,
        "license_sha256": license_sha256,
        "files": files,
        "diagnostic_only": False,
        "python": wheel_manifest["python"],
        "cuda": wheel_manifest["cuda"],
        "shared_wheelhouse": shared,
        "runtime_validated": False,
        "hidden_test_score_observed": False,
        "checkpoint_format": "frozen_base_delta" if delta else "full",
        "inference_policy": {
            "series_sampling": binding["config"].get("series_sampling", "selected"),
            "slice_sampling": "uniform"
            if binding["config"].get("slice_sampling") == "stratified"
            else binding["config"].get("slice_sampling", "uniform"),
            "slice_context": binding["config"].get("slice_context", "none"),
            "max_series": binding["config"].get("max_series", 6),
            "cache_side": binding.get("cache_side", 256),
        },
    }
    if inference_engine == "resident_two_gpu_fp32":
        manifest["inference_policy"].update(
            engine=inference_engine,
            max_seconds=28800,
            device_workers=2,
            single_decode=True,
            precision="fp32",
            CPU_cores=4,
            required_GPU="T4",
            runtime_scale_validated=False,
        )
    if fitting_admission is not None:
        manifest["fitting_admission_sha256"] = digest(fitting_admission)
    new_json(data / "bundle-manifest.json", manifest)
    manifest_sha = digest(data / "bundle-manifest.json")
    # Kaggle CLI supports only Dataset slugs here. Dedicated version 1 plus a
    # byte-bound loader prevents silent drift even if the Dataset later changes.
    new_json(
        data / "dataset-metadata.json",
        {
            "id": dataset_slug,
            "title": dataset_slug.split("/")[1].replace("-", " "),
            "licenses": [{"name": "other"}],
        },
    )
    kernel = output / "kernel"
    kernel.mkdir()
    entry = Path(__file__).with_name("notebook_entry.py").read_text()
    (kernel / "run.py").write_text(
        "EXPECTED_MANIFEST_SHA256 = " + repr(manifest_sha) + "\n" + entry
    )
    kernel_meta = {
        "id": kernel_slug,
        "title": kernel_slug.split("/")[1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [dataset_slug] + ([shared["dataset_slug"]] if shared else []),
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }
    if runtime_image is not None:
        kernel_meta.update(docker_image=runtime_image, docker_image_pinning_type="original")
    new_json(kernel / "kernel-metadata.json", kernel_meta)
    operation = {
        "schema_version": 1,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "epoch": meta["epoch"],
        "dataset_slug": dataset_slug,
        "dataset_version": 1,
        "kernel_slug": kernel_slug,
        "manifest_sha256": manifest_sha,
        "kernel_source_sha256": digest(kernel / "run.py"),
        "kernel_metadata_sha256": digest(kernel / "kernel-metadata.json"),
        "dataset_metadata_sha256": digest(data / "dataset-metadata.json"),
        "shared_wheelhouse": shared,
        "bundle_path": "dataset",
        "kernel_path": "kernel",
        "prepared_at": utcnow(),
    }
    if runtime_image is not None:
        operation["runtime_image"] = runtime_image
    if fitting_admission is not None:
        operation["fitting_admission_sha256"] = digest(fitting_admission)
    new_json(output / "operation.json", operation)
    return operation


def load_operation(path):
    operation = json.loads(path.read_text())
    base = path.parent
    data = base / "dataset"
    kernel = base / "kernel"
    if (
        digest(data / "bundle-manifest.json") != operation["manifest_sha256"]
        or digest(kernel / "run.py") != operation["kernel_source_sha256"]
        or digest(kernel / "kernel-metadata.json") != operation["kernel_metadata_sha256"]
        or digest(data / "dataset-metadata.json") != operation["dataset_metadata_sha256"]
    ):
        raise ValueError("Prepared operation changed")
    manifest = json.loads((data / "bundle-manifest.json").read_text())
    if manifest.get("shared_wheelhouse") != operation.get("shared_wheelhouse"):
        raise ValueError("Shared runtime differs from bound notebook manifest")
    if operation.get("shared_wheelhouse") is not None:
        validate_shared_wheelhouse(operation["shared_wheelhouse"])
    expected = set(manifest["files"]) | {"bundle-manifest.json", "dataset-metadata.json"}
    actual = {str(p.relative_to(data)) for p in data.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError("Inference dataset contains missing or extraneous payload files")
    verify_payload(data, manifest)
    return operation


def validate_shared_wheelhouse(shared):
    hashes = (
        "archive_sha256",
        "manifest_sha256",
        "pyproject_sha256",
        "gpu_probe_source_sha256",
        "gpu_receipt_sha256",
    )
    if (
        shared.get("schema_version") != 1
        or not all(isinstance(shared.get(k), str) and SHA.fullmatch(shared[k]) for k in hashes)
        or not SLUG.fullmatch(shared.get("dataset_slug", ""))
        or not SLUG.fullmatch(shared.get("gpu_probe_slug", ""))
        or type(shared.get("dataset_version")) is not int
        or shared.get("dataset_version") != 1
        or type(shared.get("gpu_probe_version")) is not int
        or shared.get("gpu_probe_version") != 1
        or type(shared.get("dataset_id")) is not int
        or shared["dataset_id"] < 1
        or type(shared.get("archive_bytes")) is not int
        or shared["archive_bytes"] < 1
        or shared.get("archive_file") != "wheelhouse-cp312-cu130.bin"
    ):
        raise ValueError("A complete exact private shared-wheelhouse proof is required")


def dataset_sources(operation):
    result = [operation["dataset_slug"]]
    if operation.get("shared_wheelhouse") is not None:
        validate_shared_wheelhouse(operation["shared_wheelhouse"])
        result.append(operation["shared_wheelhouse"]["dataset_slug"])
    return result


def verify_shared_wheelhouse(remote, shared):
    validate_shared_wheelhouse(shared)
    if any(
        shared[key].split("/")[0] != remote.username for key in ("dataset_slug", "gpu_probe_slug")
    ):
        raise ValueError("Shared wheelhouse belongs to another account")
    dataset = remote.dataset(shared["dataset_slug"])
    if (
        dataset.get("isPrivate") is not True
        or dataset.get("currentVersionNumber") != 1
        or dataset.get("id") != shared["dataset_id"]
    ):
        raise ValueError("Shared wheelhouse Dataset identity/privacy/version drift")
    kernel = remote.kernel("GetKernel", shared["gpu_probe_slug"], 1)
    meta = kernel.get("metadata", {})
    if (
        hashlib.sha256(kernel.get("blob", {}).get("source", "").encode()).hexdigest()
        != shared["gpu_probe_source_sha256"]
        or meta.get("isPrivate") is not True
        or meta.get("enableInternet") is not False
        or meta.get("machineShape") != "NvidiaTeslaT4"
        or meta.get("datasetDataSources") != [shared["dataset_slug"]]
        or meta.get("competitionDataSources")
    ):
        raise ValueError("Shared wheelhouse GPU acceptance source drift")
    if (
        remote.kernel("GetKernelSessionStatus", shared["gpu_probe_slug"], 1).get("status")
        != "COMPLETE"
    ):
        raise ValueError("Shared wheelhouse GPU probe has not completed")
    receipt = remote.output_json(shared["gpu_probe_slug"], 1, "wheelhouse-gpu-acceptance.json")
    if (
        object_hash(receipt) != shared["gpu_receipt_sha256"]
        or receipt.get("status") != "passed"
        or receipt.get("wheelhouse_archive_sha256") != shared["archive_sha256"]
        or receipt.get("wheelhouse_manifest_sha256") != shared["manifest_sha256"]
        or receipt.get("cuda_tensor_backward") is not True
        or receipt.get("random_tiny_dinov3_cuda_forward") is not True
    ):
        raise ValueError("Shared wheelhouse GPU acceptance bytes differ")
    return {
        "id": dataset["id"],
        "version": 1,
        "is_private": True,
        "gpu_receipt_sha256": shared["gpu_receipt_sha256"],
    }


def verify_remote_dataset(remote, operation):
    data = remote.dataset(operation["dataset_slug"])
    if data.get("isPrivate") is not True or data.get("currentVersionNumber") != 1:
        raise ValueError("Dataset must be private and remain at dedicated version 1")
    result = {"id": data.get("id"), "version": 1, "is_private": True}
    if operation.get("shared_wheelhouse") is not None:
        result["shared_wheelhouse"] = verify_shared_wheelhouse(
            remote, operation["shared_wheelhouse"]
        )
    return result


def kernel_request(operation_path, operation):
    meta = json.loads((operation_path.parent / "kernel/kernel-metadata.json").read_text())
    if meta["is_private"] is not True or meta["enable_internet"] is not False:
        raise ValueError("Only private offline kernels are permitted")
    if (
        meta["id"] != operation["kernel_slug"]
        or meta["dataset_sources"] != dataset_sources(operation)
        or meta["competition_sources"] != [COMPETITION]
    ):
        raise ValueError("Unexpected notebook identity or sources")
    request = {
        "slug": meta["id"],
        "newTitle": meta["title"],
        "text": (operation_path.parent / "kernel/run.py").read_text(),
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "machineShape": "NvidiaTeslaT4",
        "enableInternet": False,
        "datasetDataSources": meta["dataset_sources"],
        "competitionDataSources": [COMPETITION],
        "sessionTimeoutSeconds": 32400,
    }

    image = meta.get("docker_image")
    pinning = meta.get("docker_image_pinning_type")
    if image is not None or pinning is not None or operation.get("runtime_image") is not None:
        if (
            not isinstance(image, str)
            or not RUNTIME_IMAGE.fullmatch(image)
            or pinning != "original"
            or operation.get("runtime_image") != image
        ):
            raise ValueError("Kernel runtime pin differs from its bound exact image")
        request.update(dockerImage=image, dockerImagePinningType=pinning)
    return request


@rollback_on_error
def push(remote, db, operation_path):
    operation = load_operation(operation_path)
    db.execute("BEGIN IMMEDIATE")
    verify_account(db, remote)
    entry = read_entry(db, operation["checkpoint_sha256"])
    if entry["state"] != "queued":
        raise ValueError("Push already attempted; reconcile the exact kernel read-only")
    if (
        operation["dataset_slug"].split("/")[0] != remote.username
        or operation["kernel_slug"].split("/")[0] != remote.username
    ):
        raise ValueError("Prepared resources must belong to the authenticated account")
    dataset = verify_remote_dataset(remote, operation)
    request = kernel_request(operation_path, operation)
    absence = remote.require_absent_kernel(operation["kernel_slug"])
    entry.update(
        kernel_absence_proof=absence,
        state="push_intent",
        operation=operation,
        operation_sha256=digest(operation_path),
        dataset=dataset,
    )
    save_entry(db, entry, "durable intent before one SaveKernel call")
    db.execute("BEGIN IMMEDIATE")
    try:
        result = remote.call("kernels.KernelsApiService", "SaveKernel", request)
        version = result.get("versionNumber")
        if type(version) is not int or version < 1 or result.get("error"):
            raise RemoteError("Kernel push needs read-only reconciliation")
        entry.update(state="pushed", kernel_version=version)
        save_entry(db, entry, "kernel version returned")
    except Exception:
        entry["state"] = "push_uncertain"
        save_entry(db, entry, "push uncertain; retries prohibited")
        raise
    return entry


def check_kernel(remote, entry, version):
    op = entry["operation"]
    response = remote.kernel("GetKernel", op["kernel_slug"], version)
    meta = response.get("metadata", {})
    source = response.get("blob", {}).get("source", "")
    if (
        meta.get("isPrivate") is not True
        or meta.get("enableInternet") is not False
        or meta.get("machineShape") != "NvidiaTeslaT4"
        or meta.get("competitionDataSources") != [COMPETITION]
        or meta.get("datasetDataSources") != dataset_sources(op)
        or (op.get("runtime_image") is not None and meta.get("dockerImage") != op["runtime_image"])
        or hashlib.sha256(source.encode()).hexdigest() != op["kernel_source_sha256"]
    ):
        raise ValueError("Remote kernel is not the exact prepared private offline source")
    return remote.kernel("GetKernelSessionStatus", op["kernel_slug"], version)


@rollback_on_error
def reconcile_push(remote, db, sha, version):
    db.execute("BEGIN IMMEDIATE")
    verify_account(db, remote)
    entry = read_entry(db, sha)
    if entry["state"] not in {"push_intent", "push_uncertain", "pushed"}:
        raise ValueError("Entry has no kernel push to reconcile")
    check_kernel(remote, entry, version)
    entry.update(state="pushed", kernel_version=version)
    save_entry(db, entry, "read-only exact kernel source/version reconciliation")
    return entry


def allowance(db, account, now=None):
    if not account.get("entered"):
        raise ValueError("Kaggle account has not entered this competition")
    now = now or datetime.now(timezone.utc)
    checked = datetime.fromisoformat(account["checked_at"])
    if not 0 <= (now - checked).total_seconds() <= 120:
        raise ValueError("Submission quota observation is stale")
    records = [json.loads(row[0]) for row in db.execute("SELECT record FROM entries")]
    if any(x["state"] in {"submit_intent", "submit_uncertain"} for x in records):
        raise ValueError("Uncertain submission requires reconciliation before any next mutation")
    today = now.date().isoformat()
    local = sum(
        item.get("attempted_at", "").startswith(today)
        for entry in records
        for item in [entry, *entry.get("superseded_operations", [])]
    )
    return max(0, min(5 - local, 5 - account["num_today"], account["num_allowed_now"]))


@rollback_on_error
def submit(remote, db, sha):
    db.execute("BEGIN IMMEDIATE")
    verify_account(db, remote)
    entry = read_entry(db, sha)
    if entry["state"] == "submitted":
        db.rollback()
        return entry
    if entry["state"] != "pushed":
        raise ValueError("An exact completed kernel is required; reconcile uncertain attempts")
    verify_remote_dataset(remote, entry["operation"])
    state = check_kernel(remote, entry, entry["kernel_version"])
    if state.get("status") != "COMPLETE":
        raise ValueError("Kaggle kernel has not successfully completed")
    receipt = remote.output_json(
        entry["operation"]["kernel_slug"], entry["kernel_version"], "kaggle-inference-receipt.json"
    )
    if (
        receipt.get("status") != "offline_inference_complete"
        or receipt.get("bundle_manifest_sha256") != entry["operation"]["manifest_sha256"]
        or receipt.get("checkpoint_sha256") != sha
        or type(receipt.get("studies")) is not int
        or receipt["studies"] < 1
    ):
        raise ValueError("Completed notebook did not prove inference with the exact trained bundle")
    account = remote.account()
    if allowance(db, account) == 0:
        # This is a next inspection time, explicitly not a claimed server reset.
        next_check = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
        entry.update(next_quota_check_utc=next_check.isoformat(), last_quota=account)
        save_entry(db, entry, "queued overflow; refresh authoritative allowance after UTC boundary")
        return entry
    op = entry["operation"]
    identity = object_hash(
        {"checkpoint_sha256": sha, "kernel": op["kernel_slug"], "version": entry["kernel_version"]}
    )
    description = "rsna-epoch-v1:" + identity
    entry.update(
        state="submit_intent",
        attempted_at=utcnow(),
        submission_identity=identity,
        description=description,
        quota_before=account,
        inference_receipt=receipt,
        inference_receipt_sha256=object_hash(receipt),
    )
    save_entry(db, entry, "durable intent before one CreateCodeSubmission call")
    # Keep the account-wide write lock through the remote request. Another process
    # sees the intent on restart; none can consume the same allowance concurrently.
    db.execute("BEGIN IMMEDIATE")
    try:
        owner, slug = op["kernel_slug"].split("/")
        result = remote.competition(
            "CreateCodeSubmission",
            {
                "fileName": "submission.csv",
                "kernelOwner": owner,
                "kernelSlug": slug,
                "kernelVersion": entry["kernel_version"],
                "submissionDescription": description,
            },
        )
        reference = result.get("ref")
        if type(reference) is not int or reference <= 0:
            raise RemoteError("Submission result needs read-only reconciliation")
        entry.update(state="submitted", submission_id=reference, submitted_at=utcnow())
        save_entry(db, entry, "submission reference returned")
    except Exception:
        entry["state"] = "submit_uncertain"
        save_entry(db, entry, "submission uncertain; no automatic retry")
        raise
    return entry


@rollback_on_error
def reconcile_submission(remote, db, sha):
    db.execute("BEGIN IMMEDIATE")
    verify_account(db, remote)
    entry = read_entry(db, sha)
    if entry["state"] not in {"submit_intent", "submit_uncertain", "submitted"}:
        raise ValueError("There is no submission attempt to reconcile")
    matches = [x for x in remote.submissions() if x.get("description") == entry["description"]]
    if len(matches) != 1 or not matches[0].get("ref"):
        raise ValueError(
            "Submission reconciliation ambiguous/absent; keep blocked, never auto-retry"
        )
    found = matches[0]
    if entry.get("submission_id") not in (None, found["ref"]):
        raise ValueError("Submission identity changed")
    entry.update(
        state="submitted",
        submission_id=found["ref"],
        remote_status=found.get("status"),
        remote_error_description=found.get("errorDescription"),
        remote_total_bytes=found.get("totalBytes"),
        remote_outcome=submission_outcome(found),
        reconciled_at=utcnow(),
    )
    # Scores are recorded only as operational results; never select checkpoints.
    entry["public_score"] = found.get("publicScore")
    save_entry(db, entry, "read-only exact identity reconciliation")
    return entry


def submission_outcome(found):
    if found.get("errorDescription"):
        return "failed"
    if found.get("status") == "COMPLETE" and found.get("publicScore") is not None:
        return "scored"
    if found.get("status") == "COMPLETE":
        return "complete_unscored"
    return "pending_or_unknown"


def status(remote, db, sha):
    """Read-only remote progress; a fresh live query never claims hidden scoring."""
    verify_account(db, remote)
    entry = read_entry(db, sha)
    result = {"checkpoint_sha256": sha, "state": entry["state"], "checked_at": utcnow()}
    if entry.get("kernel_version"):
        result["kernel_version"] = entry["kernel_version"]
        result["kernel_status"] = check_kernel(remote, entry, entry["kernel_version"])
    if entry.get("submission_id"):
        found = [x for x in remote.submissions() if x.get("ref") == entry["submission_id"]]
        if len(found) != 1:
            raise ValueError("Submission ID is missing or ambiguous")
        result["submission"] = {
            key: found[0].get(key)
            for key in ("ref", "status", "errorDescription", "totalBytes", "publicScore")
        }
        result["submission"]["outcome"] = submission_outcome(found[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--ledger", type=Path, default=os.environ.get("RSNA_KAGGLE_LEDGER"))
    parser.add_argument("--account", default="godlydonuts")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("account", help="read-only live entry and quota check")
    sub.add_parser("list", help="local queue states across all runs; no remote requests")
    p = sub.add_parser("validate", help="local exact-payload validation before private upload")
    p.add_argument("operation", type=Path)
    p = sub.add_parser("queue", help="validate and queue immutable epoch snapshots only")
    p.add_argument("snapshots", type=Path, nargs="+")
    p = sub.add_parser("prepare", help="local complete Dataset/notebook bundle; no upload")
    for name in ("snapshot", "encoder-spec", "wheelhouse", "output", "license-file"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--shared-wheelhouse", type=Path)
    p.add_argument("--runtime-image")
    p.add_argument(
        "--inference-engine", choices=("legacy", "resident_two_gpu_fp32"), default="legacy"
    )
    p.add_argument("--fitting-admission", type=Path)
    p.add_argument("--dataset-slug", required=True)
    p.add_argument("--kernel-slug", required=True)
    p.add_argument("--license-source", required=True)
    p.add_argument("--license-sha256", required=True)
    p = sub.add_parser("push", help="one guarded kernel mutation after private Dataset publication")
    p.add_argument("operation", type=Path)
    for name in ("submit", "status", "reconcile-submission", "reconcile-push"):
        p = sub.add_parser(name)
        p.add_argument("sha")
        if name == "reconcile-push":
            p.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    if args.command not in {"account", "prepare", "validate"} and args.ledger is None:
        parser.error("Set the single canonical --ledger or RSNA_KAGGLE_LEDGER")
    try:
        if args.command == "account":
            result = Remote(args.env_file).account()
        elif args.command == "validate":
            result = load_operation(args.operation)
        elif args.command == "prepare":
            result = prepare_bundle(
                args.snapshot,
                args.encoder_spec,
                args.wheelhouse,
                args.output,
                args.dataset_slug,
                args.kernel_slug,
                args.license_file,
                args.license_source,
                args.license_sha256,
                shared_wheelhouse=args.shared_wheelhouse,
                runtime_image=args.runtime_image,
                fitting_admission=args.fitting_admission,
                inference_engine=args.inference_engine,
            )
        else:
            with ledger(args.ledger, args.account) as db:
                if args.command == "list":
                    result = [
                        json.loads(row[0])
                        for row in db.execute("SELECT record FROM entries ORDER BY rowid")
                    ]
                elif args.command == "queue":
                    result = [enqueue(db, path) for path in args.snapshots]
                elif args.command == "push":
                    result = push(Remote(args.env_file), db, args.operation)
                elif args.command == "submit":
                    result = submit(Remote(args.env_file), db, args.sha)
                elif args.command == "status":
                    result = status(Remote(args.env_file), db, args.sha)
                elif args.command == "reconcile-push":
                    result = reconcile_push(Remote(args.env_file), db, args.sha, args.version)
                else:
                    result = reconcile_submission(Remote(args.env_file), db, args.sha)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        # Do not print exception payloads, response headers, signed URLs or paths.
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(exc).__name__, "action": args.command}
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
