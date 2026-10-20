"""Prepare a committed v6 source delta locally; no credential or network use."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
BASE = "e6862ab2c250c117c44d86c2baf136a246024128"
PREVIOUS_REVISION = "71cbced934556179c3ca1ef56968b140d14df208"
PREVIOUS = "initiative/audits/hf-predictive-full-preparation-20261020-v5"
OWN = "initiative/audits/hf-predictive-serving-progress-20261020-v6"
SCOPES = (
    "AGENTS.md",
    "docs/current-state.md",
    "docs/codex-continuation.md",
    "initiative/research/predictive-T4-throughput-20261020-v2",
    "initiative/research/predictive-final-serving-20261020-v1",
    "initiative/research/predictive-final-deployment-20261020-v1",
    "initiative/research/predictive-full-fitting-20261020-v1/evaluation-decision.md",
    "initiative/research/predictive-full-fitting-20261020-v1/parallel-execution-options.md",
    "initiative/research/predictive-full-fitting-20261020-v1/actual-input-progress-1120.json",
    "initiative/research/source-zero-final-serving-20261020-v2",
    "initiative/research/teacher-pair-epoch-preservation-20261020-v1/actual-teacher_rank-epoch-2.json",
    "tests/test_predictive_final_serving.py",
    "tests/test_predictive_final_deployment.py",
    "tests/test_predictive_descriptor_cache.py",
    "tests/test_predictive_serving_mounts.py",
    PREVIOUS + "/README.md",
    PREVIOUS + "/manifest.json",
    PREVIOUS + "/actual-private-verified.json",
    OWN + "/README.md",
    OWN + "/prepare.py",
    OWN + "/publish_once.py",
    OWN + "/program.py.txt",
)
REQUIRED = {
    "AGENTS.md",
    "docs/current-state.md",
    "docs/codex-continuation.md",
    "initiative/research/predictive-T4-throughput-20261020-v2/actual-status-observer-final.json",
    "initiative/research/predictive-T4-throughput-20261020-v2/actual-runtime-analysis.json",
    "initiative/research/predictive-final-serving-20261020-v1/compile.py",
    "initiative/research/predictive-final-serving-20261020-v1/worker.py.txt",
    "initiative/research/predictive-final-serving-20261020-v1/mounts.py",
    "initiative/research/predictive-final-serving-20261020-v1/descriptor_cache.py",
    "initiative/research/predictive-final-serving-20261020-v1/descriptor-cache-cpu-benchmark.json",
    "initiative/research/predictive-final-deployment-20261020-v1/run_once.py",
    "initiative/research/predictive-final-deployment-20261020-v1/cpu.py",
    "initiative/research/predictive-final-deployment-20261020-v1/account.py",
    "initiative/research/predictive-final-deployment-20261020-v1/policy.py",
    "initiative/research/predictive-full-fitting-20261020-v1/evaluation-decision.md",
    "initiative/research/source-zero-final-serving-20261020-v2/actual-final-qualification-accepted.json",
    "initiative/research/source-zero-final-serving-20261020-v2/actual-final-competition-accepted.json",
    "initiative/research/source-zero-final-serving-20261020-v2/actual-final-entry.json",
    "initiative/research/teacher-pair-epoch-preservation-20261020-v1/actual-teacher_rank-epoch-2.json",
    "tests/test_predictive_final_serving.py",
    "tests/test_predictive_final_deployment.py",
    "tests/test_predictive_descriptor_cache.py",
    "tests/test_predictive_serving_mounts.py",
    PREVIOUS + "/actual-private-verified.json",
    OWN + "/README.md",
    OWN + "/prepare.py",
    OWN + "/publish_once.py",
    OWN + "/program.py.txt",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def select(source_git):
    assert re.fullmatch(r"[0-9a-f]{40}", source_git)
    assert git("rev-parse", source_git).decode().strip() == source_git
    assert git("rev-parse", BASE).decode().strip() == BASE
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, source_git], cwd=ROOT, check=True)
    names = git(
        "diff", "--name-only", "--diff-filter=ACMR", BASE, source_git, "--", *SCOPES
    ).decode().splitlines()
    # Selection contains only actual committed bytes; no v6 manifest or result
    # can imply its own future publication is already verified.
    assert all(
        not name.startswith(OWN + "/")
        or Path(name).name in {"README.md", "prepare.py", "publish_once.py", "program.py.txt"}
        for name in names
    )
    assert REQUIRED <= set(names), "Commit the complete reviewed milestone before preparation"
    files = {name: git("show", source_git + ":" + name) for name in names}
    assert 0 < len(files) < 200 and sum(map(len, files.values())) < 2**20
    for name, body in files.items():
        assert name.endswith((".md", ".json", ".py.txt", ".py"))
        assert not name.startswith("/") and ".." not in Path(name).parts
        assert not re.search(rb"1\.2\.826\.0\.1\.\d{3,}", body), name
        assert not re.search(rb"(?:hf_|kgat_)[A-Za-z0-9]{20,}", body), name
        body.decode("utf8")
    previous = json.loads(files[PREVIOUS + "/actual-private-verified.json"])
    assert previous["source_git"] == BASE and previous["revision"] == PREVIOUS_REVISION
    assert previous["canonical_shared_publication_lock_used"] is True
    assert previous["status"] == (
        "selected_sanitized_mission_sources_all_objects_private_download_size_SHA_verified"
    )
    manifest = {
        "source_git": source_git,
        "previous_verified_source_git": BASE,
        "previous_verified_revision": PREVIOUS_REVISION,
        "scope": "Committed own serving/deployment source and actual sanitized receipts since v5",
        "files": {
            name: {"bytes": len(body), "sha256": sha(body)} for name, body in files.items()
        },
        "MRI_reports_UIDs_per_case_predictions_external_weights_included": False,
        "new_remote_operations": 0,
        "this_manifest_is_not_a_publication_or_verification_receipt": True,
    }
    return files, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    _, manifest = select(args.source_git)
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "local_committed_mirror_manifest_prepared_only",
                "source_git": args.source_git,
                "objects_without_manifest": len(manifest["files"]),
                "bytes_without_manifest": sum(item["bytes"] for item in manifest["files"].values()),
                "remote_operations_or_intents": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
