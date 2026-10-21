"""Private fixed-version inputs and immutable epoch storage for one Colab run."""

import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

ROOT = Path("/content/rsna-independent-C100-20261021-v1")
CONTROL = Path(__file__).resolve().parent
HF_REPO = "GodlyDonuts/RSNA-Colab-Deadline-20261021"
INPUT_SLUG = "godlydonuts/rsna-colab-c100-contexts-20261021-v1"
SCIENCE = "94a3921aefdf333624caf41aff77aa10b4a0c57e"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def once(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")


def account(token):
    import requests

    w = load("Colab_private_canonical_API", CONTROL / "workflow.py")
    p = load("Colab_private_publication", CONTROL / "publication.py")
    r = w.Remote.__new__(w.Remote)
    r.session = requests.Session()
    r.session.headers.update({"Authorization": "Bearer " + token})
    i = r.call("security.OAuthService", "IntrospectToken", {"token": token})
    assert i.get("active") is True and i.get("username") == "godlydonuts"
    r.username = i["username"]
    return r, p


def download_file(remote, slug, name, target, expected, max_seconds=1800):
    """Authenticate only with Kaggle; never forward its token to storage hosts."""
    import requests

    assert not target.exists()
    owner, dataset = slug.split("/")
    remote.session.cookies.clear()
    response = remote.session.post(
        "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset",
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
    started = time.monotonic()
    try:
        for _ in range(6):
            if response.status_code == 200:
                digest = hashlib.sha256()
                count = 0
                with target.open("xb") as f:
                    for block in response.iter_content(8 * 2**20):
                        count += len(block)
                        assert (
                            count <= expected["bytes"] and time.monotonic() - started < max_seconds
                        )
                        digest.update(block)
                        f.write(block)
                assert count == expected["bytes"] and digest.hexdigest() == expected["sha256"]
                return target
            assert response.status_code in (301, 302, 303, 307, 308), "Private download unavailable"
            address = urljoin(response.url, response.headers.get("Location", ""))
            parsed = urlparse(address)
            assert parsed.scheme == "https" and not parsed.username and not parsed.password
            assert parsed.port in (None, 443) and (
                parsed.hostname == "storage.googleapis.com"
                or (parsed.hostname or "").endswith(".kaggleusercontent.com")
            )
            response.close()
            response = requests.get(address, timeout=(15, 180), allow_redirects=False, stream=True)
        raise ValueError("Unexpected redirect depth")
    finally:
        response.close()


def extract(path, manifest, seen):
    with tarfile.open(path, "r:") as tar:
        for member in tar:
            name = member.name
            parts = PurePosixPath(name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not parts.is_absolute() and ".." not in parts.parts
            assert name in manifest["members"] and name not in seen
            proof = manifest["members"][name]
            assert member.size == proof["bytes"]
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            h = hashlib.sha256()
            count = 0
            with tar.extractfile(member) as src, dest.open("xb") as out:
                while block := src.read(8 * 2**20):
                    count += len(block)
                    assert count <= proof["bytes"]
                    h.update(block)
                    out.write(block)
            assert count == proof["bytes"] and h.hexdigest() == proof["sha256"]
            seen.add(name)


def download_inputs(HF_token, Kaggle_token, receipt):
    from huggingface_hub import HfApi, hf_hub_download

    os.umask(0o077)
    assert not ROOT.exists(), "Reconcile original local inputs before any continuation"
    assert shutil.disk_usage("/content").free > 170 * 2**30
    remote, pub = account(Kaggle_token)
    before = pub.metadata(remote, INPUT_SLUG)
    expected = receipt["files"]
    assert receipt["slug"] == INPUT_SLUG and receipt["version"] == 1
    assert pub.list_files(remote, INPUT_SLUG) == {n: p["bytes"] for n, p in expected.items()}
    ROOT.mkdir(mode=0o700)
    cache = ROOT / "archives"
    cache.mkdir(mode=0o700)
    manifest_path = download_file(
        remote,
        INPUT_SLUG,
        "input-manifest.bin",
        cache / "input-manifest.bin",
        expected["input-manifest.bin"],
    )
    manifest = json.loads(manifest_path.read_text())
    assert (
        manifest["source_git"] == SCIENCE
        and manifest["cohort"] == 3887
        and manifest["seed"] == 20261021
    )
    assert manifest["repo"] == HF_REPO and manifest["context_bytes"] == 67005012716
    api = HfApi(token=HF_token)
    assert api.repo_info(HF_REPO, repo_type="dataset").private
    seen = set()
    for a in manifest["archives"]:
        if a["provider"] == "HF":
            path = Path(
                hf_hub_download(
                    HF_REPO,
                    a["path"],
                    repo_type="dataset",
                    revision=a["revision"],
                    token=HF_token,
                    cache_dir=cache / "hf",
                )
            )
            assert path.stat().st_size == a["bytes"] and sha(path) == a["sha256"]
        else:
            assert a["provider"] == "Kaggle" and a["slug"] == INPUT_SLUG and a["version"] == 1
            assert expected[a["remote_name"]] == {k: a[k] for k in ("bytes", "sha256")}
            path = download_file(remote, INPUT_SLUG, a["remote_name"], cache / a["remote_name"], a)
        extract(path, manifest, seen)
        print(
            json.dumps(
                {
                    "phase": "original_archive_independently_verified",
                    "archive": a["path"],
                    "members_completed": len(seen),
                }
            ),
            flush=True,
        )
    assert seen == set(manifest["members"]) and len(manifest["rows"]) == 3887
    assert pub.metadata(remote, INPUT_SLUG) == before
    manifest.update(
        actual_private_revision=f"Kaggle:{before['id']}:v1+original_HF_revisions",
        actual_private_Kaggle=before,
        all_original_member_bytes_SHA_verified_on_Colab=True,
        downloaded_manifest_sha256=sha(manifest_path),
    )
    once(ROOT / "input-verified.private.json", manifest)
    print(
        json.dumps(
            {
                "phase": "all_3887_original_inputs_verified",
                "Kaggle_id": before["id"],
                "manifest_sha256": sha(manifest_path),
                "training_started": False,
            }
        ),
        flush=True,
    )
    return manifest


class Store:
    """Single Colab writer in a separately admitted immutable Dataset namespace."""

    def __init__(self, token, lease):
        self.token = token
        self.remote, self.pub = account(token)
        self.lease = lease
        assert lease["status"] == "canonical_isolated_Colab_checkpoint_namespace_admitted"
        assert lease["input_slug"] == INPUT_SLUG and lease["seed"] == 20261021
        assert lease["maximum_private_dataset_creations"] == 7
        assert lease["maximum_bytes_per_dataset"] == 2 * 2**30
        assert lease["only_one_Colab_writer"] is True and lease["automatic_mutation_retry"] is False
        assert lease["GPU_allocations_or_competition_entries_authorized_by_this_lease"] is False
        assert all(
            "/" not in name and sha(CONTROL / name) == h for name, h in lease["code_sha256"].items()
        )
        assert set(lease["slugs"]) == {f"epoch-{e:03d}" for e in range(1, 7)} | {"terminal"}
        assert len(set(lease["slugs"].values())) == 7 and all(
            s.startswith("godlydonuts/rsna-colab-c100-") for s in lease["slugs"].values()
        )
        self.create = load(
            "Colab_private_multifile", CONTROL / "kaggle_io.py"
        ).private_multifile_publisher(self.pub)

    def publish(self, key, files, destination):
        slug = self.lease["slugs"][key]
        assert not (destination / "Kaggle-intent.json").exists()
        assert len(files) <= 10 and sum(p.stat().st_size for p in files.values()) < 2 * 2**30
        absence = self.pub.require_absent(self.remote, slug)
        upload = destination / "private-upload"
        upload.mkdir(mode=0o700)
        proofs = {}
        for name, path in files.items():
            assert "/" not in name and path.is_file() and not path.is_symlink()
            remote_name = name + ".bin"
            os.link(path, upload / remote_name)
            proofs[remote_name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        once(
            destination / "Kaggle-intent.json",
            {
                "slug": slug,
                "files": proofs,
                "lease": self.lease,
                "absence": absence,
                "automatic_retry": False,
            },
        )
        returned = self.create(self.token, slug, "Private Colab C100 " + key, upload)
        once(destination / "Kaggle-response.json", returned)
        start = time.monotonic()
        while True:
            ready = self.remote.call(
                "datasets.DatasetApiService",
                "GetDatasetStatus",
                {"ownerSlug": "godlydonuts", "datasetSlug": slug.split("/")[1]},
            )
            if ready.get("status") in ("READY", 5):
                break
            assert time.monotonic() - start < 1800, "Reconcile original ingestion; never recreate"
            time.sleep(15)
        metadata = self.pub.metadata(self.remote, slug)
        assert self.pub.list_files(self.remote, slug) == {n: p["bytes"] for n, p in proofs.items()}
        for name, proof in proofs.items():
            self.pub.stream_file(self.remote, slug, name, proof, max_seconds=1800)
        assert self.pub.metadata(self.remote, slug) == metadata
        result = {
            "status": "private_epoch_full_download_size_SHA_verified",
            **metadata,
            "files": proofs,
            "revision": f"Kaggle:{metadata['id']}:v1",
            "key": key,
            "new_GPU_or_competition_calls": 0,
        }
        once(destination / "verified.json", result)
        print(
            json.dumps(
                {"phase": "private_epoch_preserved", "key": key, "Dataset_id": metadata["id"]}
            ),
            flush=True,
        )
        return result
