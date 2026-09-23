"""Publish one small, immutable post-fold-0 audit bundle to private Hugging Face."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[2]
REPO = "GodlyDonuts/RSNA-Data"
EXPECTED_GIT = "3bf54fa"  # all selected source files are present by this commit
PREFIX = "experiments/dinov3-v1/milestones/post-fold0-3bf54fa"
JOURNAL = ROOT / "artifacts/hf-post-fold0-milestone-v1"
FILES = (
    "docs/post-fold0-dinov3-plan.md",
    "initiative/audits/post-fold0-score-analysis-v1/README.md",
    "initiative/audits/post-fold0-score-analysis-v1/metrics.json",
    "initiative/audits/post-fold0-score-analysis-v1/recompute.py",
    "initiative/audits/kaggle-epoch8-v1/README.md",
    "initiative/audits/kaggle-epoch8-v1/kernel2-derivation.json",
    "initiative/audits/kaggle-epoch8-v1/kernel2-intent.json",
    "initiative/audits/kaggle-epoch8-v1/kernel2-push-result.json",
    "initiative/audits/kaggle-epoch8-v1/public-example-receipt.json",
    "initiative/audits/kaggle-epoch8-v1/submission-readiness.json",
    "initiative/audits/kaggle-epoch10-v1/README.md",
    "initiative/audits/kaggle-epoch10-v1/public-example-receipt.json",
    "initiative/audits/kaggle-epoch10-v1/submission-reconciliation.json",
    "initiative/audits/kaggle-epoch10-v1/searchguard-intent.json",
    "initiative/audits/kaggle-epoch10-v1/searchguard-result.json",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def credential() -> str:
    env = ROOT / ".env"
    if env.stat().st_mode & 0o077:
        raise ValueError("Local credential file is not private")
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    token = values.get("HF_TOKEN", "")
    if not token.startswith("hf_"):
        raise ValueError("Hugging Face token unavailable")
    return token


def main() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not head.startswith(EXPECTED_GIT):
        raise ValueError("Selected audit source is not at the reviewed Git commit")
    if any(
        subprocess.check_output(["git", "status", "--porcelain", "--", name], cwd=ROOT)
        for name in FILES
    ):
        raise ValueError("Selected milestone files have uncommitted changes")
    files = {name: ROOT / name for name in FILES}
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise ValueError("Selected small audit file is missing or linked")
    total = sum(path.stat().st_size for path in files.values())
    if total > 5 * 1024 * 1024 or shutil.disk_usage(ROOT).free < 20 * 1024**3 + 2 * total:
        raise ValueError("Milestone exceeds small-file or local disk reserve")
    if JOURNAL.exists() or JOURNAL.is_symlink():
        raise ValueError("Milestone publication already attempted; reconcile read-only")
    api = HfApi(token=credential())
    info = api.repo_info(REPO, repo_type="dataset")
    if info.private is not True:
        raise ValueError("Milestone may only enter the private dataset")
    remote_paths = [f"{PREFIX}/{name}" for name in FILES] + [f"{PREFIX}/manifest.json"]
    if list(api.get_paths_info(REPO, paths=remote_paths, repo_type="dataset")):
        raise ValueError("Dedicated private milestone path already has files")
    JOURNAL.mkdir(mode=0o700)
    manifest = {
        "schema_version": 1,
        "status": "verified_post_fold0_milestone",
        "repo_id": REPO,
        "git_commit": head,
        "epoch10_checkpoint_sha256": "1464f64063d2ac8729452f0f3edc112a21516ce51ad3b8718a176a599dc3f28e",
        "epoch10_hf_revision": "8e41da9c831af6bf777daf48b81109257f7e4b82",
        "epoch8_checkpoint_sha256": "e156bcf1d4e0d7509d9a48ac0630af80e7119ee232e23082371c575ffc207373",
        "epoch8_hf_revision": "0d90e6a19288615bb882bca1a84b70c6047179dd",
        "official_dinov3_license": "https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md",
        "public_kaggle_score_claimed": False,
        "epoch8_competition_submission_created": False,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha(path)}
            for name, path in files.items()
        },
    }
    manifest_path = JOURNAL / "manifest.json"
    exclusive_json(manifest_path, manifest)
    sources = {**files, "manifest.json": manifest_path}
    exclusive_json(
        JOURNAL / "intent.json",
        {
            "schema_version": 1,
            "status": "one_private_commit_intent",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_id": REPO,
            "prefix": PREFIX,
            "parent_commit": info.sha,
            "git_commit": head,
            "manifest_sha256": sha(manifest_path),
            "paths": sorted(remote_paths),
        },
    )
    try:
        commit = api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            parent_commit=info.sha,
            commit_message="Archive verified fold-0 analysis and Kaggle submission receipts",
            operations=[
                CommitOperationAdd(path_in_repo=f"{PREFIX}/{name}", path_or_fileobj=path)
                for name, path in sources.items()
            ],
        )
    except Exception as exc:
        exclusive_json(
            JOURNAL / "uncertain.json",
            {"status": "commit_result_uncertain", "error_type": type(exc).__name__},
        )
        raise
    revision = commit.oid
    exclusive_json(
        JOURNAL / "commit-result.json",
        {"status": "commit_returned", "repo_id": REPO, "revision": revision},
    )
    if api.repo_info(REPO, repo_type="dataset", revision=revision).private is not True:
        raise ValueError("Committed revision is not private")
    with tempfile.TemporaryDirectory(prefix="rsna-hf-milestone-") as temp:
        for name, source in sources.items():
            fetched = Path(
                api.hf_hub_download(
                    REPO,
                    f"{PREFIX}/{name}",
                    repo_type="dataset",
                    revision=revision,
                    local_dir=temp,
                )
            )
            if fetched.stat().st_size != source.stat().st_size or sha(fetched) != sha(source):
                raise ValueError("Private milestone download differs from source bytes")
    exclusive_json(
        JOURNAL / "verified.json",
        {
            "schema_version": 1,
            "status": "private_milestone_fully_verified",
            "repo_id": REPO,
            "revision": revision,
            "prefix": PREFIX,
            "file_count": len(sources),
            "manifest_sha256": sha(manifest_path),
            "downloaded_all_bytes": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "private_milestone_fully_verified",
                "revision": revision,
                "files": len(sources),
            }
        )
    )


if __name__ == "__main__":
    main()
