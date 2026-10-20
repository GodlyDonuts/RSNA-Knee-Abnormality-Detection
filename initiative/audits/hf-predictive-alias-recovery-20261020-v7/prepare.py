"""Prepare a committed v7 source delta locally; no credential or network use."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
BASE = "d16501be4c5055015749dcb63ce4268196ff49a0"
PREVIOUS_REVISION = "92394435c7a73e1369904fc582b202863517a34b"
PREVIOUS = "initiative/audits/hf-predictive-serving-progress-20261020-v6"
OWN = "initiative/audits/hf-predictive-alias-recovery-20261020-v7"
SCOPES = (
    "AGENTS.md", "docs/current-state.md", "docs/codex-continuation.md",
    "rsna/competitive/predictive/cases.py",
    "initiative/research/predictive-alias-failure-20261020-v1",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v1",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v2",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v3",
    "initiative/research/predictive-full-fitting-20261020-v2",
    "initiative/research/predictive-final-serving-20261020-v1/contract.py",
    "initiative/research/predictive-final-serving-20261020-v1/README.md",
    "initiative/research/predictive-final-deployment-20261020-v1/policy.py",
    "initiative/research/source-zero-final-serving-20261020-v2/actual-final-hidden-outcome.json",
    "tests/test_predictive_cases_alias_metadata.py",
    "tests/test_predictive_alias_recovery_owner.py",
    "tests/test_predictive_alias_recovery_v2.py",
    "tests/test_predictive_alias_recovery_v3.py",
    "tests/test_predictive_full_fitting_alias_v2.py",
    "tests/test_predictive_final_serving.py",
    "tests/test_predictive_final_deployment.py",
    PREVIOUS + "/README.md", PREVIOUS + "/manifest.json", PREVIOUS + "/actual-private-verified.json",
    OWN + "/README.md", OWN + "/prepare.py", OWN + "/publish_once.py", OWN + "/program.py.txt",
)
REQUIRED = {
    "AGENTS.md", "docs/current-state.md", "docs/codex-continuation.md",
    "rsna/competitive/predictive/cases.py",
    "initiative/research/predictive-alias-failure-20261020-v1/actual-original-terminals.json",
    "initiative/research/predictive-alias-failure-20261020-v1/actual-diagnosis.json",
    "initiative/research/predictive-alias-failure-20261020-v1/actual-regression.json",
    "initiative/research/predictive-alias-failure-20261020-v1/actual-old-fitting-owner-stopped.json",
    "initiative/research/predictive-alias-failure-20261020-v1/actual-first-recovery-preintent-failure.json",
    "initiative/research/predictive-alias-failure-20261020-v1/actual-prefix-byte-admission.json",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v1/owner.py",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v2/owner.py",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v3/owner.py",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v3/reconcile.py",
    "initiative/research/predictive-full-cache-alias-recovery-20261020-v3/actual-recovery-started.json",
    "initiative/research/predictive-full-fitting-20261020-v2/admit_cpu.py.txt",
    "initiative/research/predictive-full-fitting-20261020-v2/alias_provenance.py",
    "initiative/research/predictive-full-fitting-20261020-v2/worker.py.txt",
    "initiative/research/predictive-full-fitting-20261020-v2/publish_epoch.py.txt",
    "initiative/research/predictive-full-fitting-20261020-v2/owner.py",
    "initiative/research/predictive-full-fitting-20261020-v2/actual-cpu-prepared.json",
    "initiative/research/predictive-full-fitting-20261020-v2/actual-owner-started.json",
    "initiative/research/predictive-final-serving-20261020-v1/contract.py",
    "initiative/research/predictive-final-serving-20261020-v1/README.md",
    "initiative/research/predictive-final-deployment-20261020-v1/policy.py",
    "initiative/research/source-zero-final-serving-20261020-v2/actual-final-hidden-outcome.json",
    "tests/test_predictive_cases_alias_metadata.py",
    "tests/test_predictive_alias_recovery_owner.py",
    "tests/test_predictive_alias_recovery_v2.py",
    "tests/test_predictive_alias_recovery_v3.py",
    "tests/test_predictive_full_fitting_alias_v2.py",
    "tests/test_predictive_final_serving.py",
    "tests/test_predictive_final_deployment.py",
    PREVIOUS + "/actual-private-verified.json",
    OWN + "/README.md", OWN + "/prepare.py", OWN + "/publish_once.py", OWN + "/program.py.txt",
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
    # Selection contains only actual committed bytes; no v7 manifest or result
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
        "scope": "Committed own alias correction, failure recovery, full fitting successor and actual sanitized receipts since v6",
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
