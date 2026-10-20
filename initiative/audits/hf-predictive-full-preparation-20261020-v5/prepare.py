"""Prepare an exact committed small-source delta; no credential or network use."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
BASE = "96b61aeb"
SCOPES = (
    "AGENTS.md", "docs/current-state.md", "docs/codex-continuation.md",
    "rsna/competitive/predictive",
    "scripts/kaggle/gpu_capacity.py",
    "tests/test_kaggle_gpu_capacity.py",
    "tests/test_predictive*.py",
    "tests/test_source_zero_serving_successor.py",
    "initiative/research/predictive-*20261020-v*",
    "initiative/research/source-zero-full-T4-20261020-v1/actual-*.json",
    "initiative/research/source-zero-final-serving-20261020-v1/actual-*.json",
    "initiative/research/source-zero-final-serving-20261020-v2",
    "initiative/audits/hf-predictive-cached-engine-20261020-v4/actual-private-verified.json",
    "initiative/audits/hf-predictive-cached-engine-20261020-v4/manifest.json",
    "initiative/audits/hf-predictive-full-preparation-20261020-v5",
)
EXCLUDED = {
    "initiative/audits/hf-predictive-full-preparation-20261020-v5/manifest.json",
    "initiative/audits/hf-predictive-full-preparation-20261020-v5/actual-private-verified.json",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def select(source_git):
    assert re.fullmatch(r"[0-9a-f]{40}", source_git)
    assert git("rev-parse", source_git).decode().strip() == source_git
    base = git("rev-parse", BASE).decode().strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", base, source_git], cwd=ROOT, check=True)
    names = git(
        "diff", "--name-only", "--diff-filter=ACMR", base, source_git, "--", *SCOPES
    ).decode().splitlines()
    files = {name: git("show", source_git + ":" + name) for name in names if name not in EXCLUDED}
    assert 0 < len(files) < 200 and sum(map(len, files.values())) < 2**20
    for name, body in files.items():
        assert name.endswith((".md", ".json", ".py.txt", ".py"))
        assert not name.startswith("/") and ".." not in Path(name).parts
        assert not re.search(rb"1\.2\.826\.0\.1\.\d{3,}", body), name
        assert not re.search(rb"(?:hf_|kgat_)[A-Za-z0-9]{20,}", body), name
        body.decode("utf8")
    manifest = {
        "source_git": source_git,
        "previous_verified_source_git": base,
        "previous_verified_revision": "ebf7cef1685792afc7788417ff9ac888240acdef",
        "scope": "committed new own controls and sanitized receipts since the v4 source; selected scopes",
        "files": {name: {"bytes": len(body), "sha256": sha(body)} for name, body in files.items()},
        "MRI_reports_UIDs_per_case_predictions_external_weights_included": False,
        "new_remote_operations": 0,
    }
    return files, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    _, manifest = select(args.source_git)
    path = DIRECTORY / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "local_committed_mirror_manifest_prepared_only",
        "source_git": args.source_git, "objects_without_manifest": len(manifest["files"]),
        "bytes_without_manifest": sum(item["bytes"] for item in manifest["files"].values()),
        "remote_operations_or_intents": 0,
    }))


if __name__ == "__main__":
    main()
