"""Privately preserve the exact staged decoder-error contingency source once."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/kaggle"))
from project_code import files_at  # noqa: E402

PACKAGE_COMMIT = "fcbac6b31ca36f28127b6be04286e4bd52f15e5c"
TRAINING_COMMIT = "b74060b98f01bff1b924070ccba5ce44dd002f40"
MANIFEST_SHA = "cce4f1d52ab9c4173277cc0fd2c220c25803ae3275ef8120a019d376f4a66fb9"
REPO = "GodlyDonuts/RSNA-Data"
PREFIX = "experiments/dinov3-v1/milestones/kaggle-decoder-errors-contingency-fcbac6b"
REMOTE_STAGE = "/home/sa305415/rsna-knee/kaggle/code-projections/fcbac6b-b740-decodererrors-v1"
REMOTE_JOURNAL = REMOTE_STAGE + "-hf-journal"
LOCAL_JOURNAL = ROOT / "artifacts/hf-decoder-errors-code-projection-v1"
CONTROL_SOCKET = Path("/tmp/rsna-ssh-20260925-goal/newton")
HOST = "sa305415@newton.ist.ucf.edu"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def credential() -> str:
    env = ROOT / ".env"
    if env.stat().st_mode & 0o077:
        raise ValueError("Local credential file must be private")
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    token = values.get("HF_TOKEN", "")
    if not token.startswith("hf_"):
        raise ValueError("Private Hugging Face credential unavailable")
    return token


def ssh(code: str, *args: str, input_text: str | None = None) -> str:
    if not CONTROL_SOCKET.is_socket():
        raise ValueError("Live Newton SSH control socket required")
    remote = shlex.join(["python3", "-c", code, *args])
    result = subprocess.run(
        ["ssh", "-S", str(CONTROL_SOCKET), "-o", "BatchMode=yes", HOST, remote],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise ValueError(f"Newton journal/status request failed: {result.returncode}")
    return result.stdout.strip()


def exact_projection() -> tuple[dict[str, bytes], bytes]:
    package = files_at(ROOT, PACKAGE_COMMIT)
    training = files_at(ROOT, TRAINING_COMMIT)
    train_code = {
        path: digest(payload) for path, (payload, _) in training.items() if path.startswith("rsna/")
    }
    package = {path: value for path, value in package.items() if not path.startswith("rsna/")}
    package.update({path: value for path, value in training.items() if path.startswith("rsna/")})
    if {
        path: digest(payload) for path, (payload, _) in package.items() if path.startswith("rsna/")
    } != train_code:
        raise ValueError("Pinned trained runtime differs")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode().strip()

    manifest = {
        "schema_version": 1,
        "purpose": "minimal_exact_code_projection",
        "package_git_commit": PACKAGE_COMMIT,
        "training_git_commit": TRAINING_COMMIT,
        "package_git_tree": git("rev-parse", PACKAGE_COMMIT + "^{tree}"),
        "training_code_sha256": train_code,
        "training_code_origin": TRAINING_COMMIT,
        "packaging_code_origin": PACKAGE_COMMIT,
        "files": {
            path: {"sha256": digest(payload), "bytes": len(payload), "mode": mode}
            for path, (payload, mode) in sorted(package.items())
        },
        "excluded": "Git metadata, reports, labels, image/cache arrays, weights, credentials and unrelated repository files.",
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if digest(manifest_bytes) != MANIFEST_SHA or len(package) != 51:
        raise ValueError("Exact code projection changed")
    payloads = {f"{PREFIX}/code/{path}": payload for path, (payload, _) in package.items()}
    payloads[f"{PREFIX}/projection.json"] = manifest_bytes
    return payloads, manifest_bytes


def remote_record(name: str, value: dict, *, create_dir: bool = False) -> None:
    code = """
import json,os,sys
from pathlib import Path
folder=Path(sys.argv[1]); name=sys.argv[2]
if sys.argv[3]=='1': folder.mkdir(mode=0o700,parents=True,exist_ok=False)
else: assert folder.is_dir() and not folder.is_symlink()
data=json.loads(sys.stdin.read())
with (folder/name).open('x') as stream:
 json.dump(data,stream,sort_keys=True,indent=2); stream.write('\\n'); stream.flush(); os.fsync(stream.fileno())
"""
    ssh(code, REMOTE_JOURNAL, name, "1" if create_dir else "0", input_text=json.dumps(value))


def main() -> None:
    if LOCAL_JOURNAL.exists() or LOCAL_JOURNAL.is_symlink():
        raise ValueError("Publication already attempted; reconcile read-only")
    if (
        ssh(
            "from pathlib import Path; import sys; print(int(Path(sys.argv[1]).exists()))",
            REMOTE_JOURNAL,
        )
        != "0"
    ):
        raise ValueError("Newton publication journal exists; reconcile read-only")
    remote_sha = ssh(
        "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())",
        REMOTE_STAGE + "/projection.json",
    )
    if remote_sha != MANIFEST_SHA:
        raise ValueError("Newton staged projection changed")
    payloads, manifest = exact_projection()
    api = HfApi(token=credential())
    info = api.repo_info(REPO, repo_type="dataset")
    if info.private is not True:
        raise ValueError("Target dataset is not private")
    if list(api.get_paths_info(REPO, paths=sorted(payloads), repo_type="dataset")):
        raise ValueError("Private milestone path already contains an object")
    intent = {
        "schema_version": 1,
        "status": "one_private_code_projection_commit_intent",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO,
        "prefix": PREFIX,
        "parent_commit": info.sha,
        "package_git_commit": PACKAGE_COMMIT,
        "training_git_commit": TRAINING_COMMIT,
        "projection_sha256": digest(manifest),
        "files": {
            path: {"bytes": len(payload), "sha256": digest(payload)}
            for path, payload in sorted(payloads.items())
        },
        "competition_submission_created": False,
    }
    LOCAL_JOURNAL.mkdir(mode=0o700)
    save_new(LOCAL_JOURNAL / "intent.json", intent)
    remote_record("intent.json", intent, create_dir=True)
    try:
        commit = api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            parent_commit=info.sha,
            commit_message="Preserve exact tested Kaggle decoder-error contingency source",
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))
                for path, payload in sorted(payloads.items())
            ],
        )
    except Exception as exc:
        save_new(
            LOCAL_JOURNAL / "uncertain.json",
            {"status": "commit_result_uncertain", "error_type": type(exc).__name__},
        )
        raise
    revision = commit.oid
    result = {
        "schema_version": 1,
        "status": "commit_returned",
        "revision": revision,
        "repo_id": REPO,
    }
    save_new(LOCAL_JOURNAL / "commit-result.json", result)
    remote_record("commit-result.json", result)
    if api.repo_info(REPO, repo_type="dataset", revision=revision).private is not True:
        raise ValueError("Committed revision is not private")
    with tempfile.TemporaryDirectory(prefix="rsna-hf-codeprojection-") as temp:
        for path, payload in sorted(payloads.items()):
            downloaded = Path(
                api.hf_hub_download(
                    REPO, path, repo_type="dataset", revision=revision, local_dir=temp
                )
            )
            if downloaded.stat().st_size != len(payload) or digest(
                downloaded.read_bytes()
            ) != digest(payload):
                raise ValueError("Private code object download differed from exact source")
    verified = {
        "schema_version": 1,
        "status": "private_code_projection_fully_verified",
        "repo_id": REPO,
        "revision": revision,
        "prefix": PREFIX,
        "projection_sha256": MANIFEST_SHA,
        "files_verified_by_download": len(payloads),
        "source_bytes": sum(map(len, payloads.values())),
        "competition_submission_created": False,
    }
    save_new(LOCAL_JOURNAL / "verified.json", verified)
    remote_record("verified.json", verified)
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
