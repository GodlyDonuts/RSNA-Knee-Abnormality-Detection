"""Exact epoch002 numerical recovery with local immutable, network-independent storage."""

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path("/content/rsna-independent-C100-20261021-v1")
CONTROL = Path("/content/rsna-colab-private-control-20261021-v2")
FIRST = BASE / "run-C100-seed20261021"
PRIOR = BASE / "run-C100-seed20261021-recovery-v1"
RUN = BASE / "run-C100-seed20261021-recovery-v2"
OLD_RECOVERY = Path("/content/rsna-colab-epoch001-recovery-20261021-v1/resume.py")
OLD_SHA = "27b87a02e899fda775bb9985841b0354c614805bbdc6ea08ea92855cf6bfea0f"
CHECKPOINT_SHA = "0fab361de8da414968391b8f7a596190d68cb1e2966e72e96429bcc8b5dae516"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def once(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")


def transform(text, previous):
    """Change storage/output/recovery control only; original fit and Cases remain exact."""
    body = previous.build(text)
    replace = previous.replace_once
    body = replace(body, f"RUN = Path({str(PRIOR)!r})", f"RUN = Path({str(RUN)!r})")
    body = replace(
        body,
        "resume=ORIGINAL/'checkpoints/epoch-001.pt',",
        "resume=PRIOR/'checkpoints/epoch-002.pt',",
    )
    body = replace(
        body,
        "    publications = {1: done}",
        "    second = Future(); second.set_result(SECOND_LOCAL)\n"
        "    publications = {1: done, 2: second}",
    )
    body = replace(
        body,
        "'all_epochs_private_verified':True,",
        "'all_epochs_private_verified':False, 'all_epochs_local_immutable':True,",
    )
    body = replace(
        body,
        "    print(json.dumps(terminal), flush=True)",
        "    print(json.dumps({**terminal, 'publication_pending':True}), flush=True)",
    )
    ast.parse(body)
    return body


class LocalStore:
    """No account object, credentials, HTTP request, upload, or retry."""

    def __init__(self, token, lease):
        assert token is None and lease["seed"] == 20261021
        self.lease = lease

    def publish(self, key, files, destination):
        assert key in self.lease["slugs"] and len(files) <= 10
        assert sum(p.stat().st_size for p in files.values()) < 2 * 2**30
        assert shutil.disk_usage("/content").free > 20 * 2**30
        proofs = {}
        for name, path in files.items():
            assert path.is_file() and not path.is_symlink() and "/" not in name
            proofs[name + ".bin"] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        result = {
            "status": "local_immutable_checkpoint_ready_publication_pending",
            "key": key,
            "intended_slug": self.lease["slugs"][key],
            "files": proofs,
            "new_account_mutations": 0,
            "remote_preservation_verified": False,
        }
        once(destination / "local-preserved.json", result)
        print(
            json.dumps(
                {
                    "phase": "local_epoch_sealed_publication_pending",
                    "key": key,
                    "files": len(proofs),
                    "new_account_mutations": 0,
                }
            ),
            flush=True,
        )
        return result


def main():
    os.umask(0o077)
    assert not RUN.exists(), "Original successor already exists; never repeat"
    assert sha(OLD_RECOVERY) == OLD_SHA
    assert shutil.disk_usage("/content").free > 30 * 2**30
    sys.path.insert(0, str(CONTROL))
    import torch

    spec = importlib.util.spec_from_file_location("accepted_original_epoch1_recovery", OLD_RECOVERY)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    checkpoint = PRIOR / "checkpoints/epoch-002.pt"
    intent = json.loads((PRIOR / "publication-002/Kaggle-intent.json").read_text())
    assert (
        sha(checkpoint) == CHECKPOINT_SHA == intent["files"]["full-state.private.pt.bin"]["sha256"]
    )
    assert not (PRIOR / "terminal.json").exists()
    assert not (PRIOR / "publication-002/Kaggle-response.json").exists()
    assert not (PRIOR / "publication-002/verified.json").exists()
    updates = [json.loads(s) for s in (PRIOR / "updates.jsonl").read_text().splitlines()]
    assert [v["step"] for v in updates] == list(range(487, 976))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    binding = json.loads((FIRST / "run-start.json").read_text())
    assert state["binding"] == binding and state["step"] == 972 and state["cohort_size"] == 3887
    assert state["recipe"] == binding["recipe"] and state["epoch_snapshots"] is True
    assert len(state["history"]) == 972 and [x["step"] for x in state["history"]] == list(
        range(1, 973)
    )
    assert state["optimizer"]["state"] and state["rng"]
    fields = set(state["history"][0])
    assert state["history"][486:] == [{k: r[k] for k in fields} for r in updates[:486]]
    del state
    first = json.loads((FIRST / "publication-001/reconciled-original.json").read_text())[
        "publication"
    ]
    assert first["id"] == 12756621 and first["private"] is True and first["version"] == 1
    prior_continuation = json.loads((PRIOR / "continuation.json").read_text())
    assert time.time() < prior_continuation["original_ten_hour_deadline"]
    control_path = str(Path(__file__).resolve())
    recovery = {
        "status": "exact_epoch002_resume_with_network_independent_local_preservation",
        "source_checkpoint_sha256": CHECKPOINT_SHA,
        "resume_update": 972,
        "discarded_logged_unsaved_updates": [973, 974, 975],
        "possible_unlogged_update_replayed": 976,
        "original_run": str(FIRST),
        "prior_run": str(PRIOR),
        "resumed_run": str(RUN),
        "control_path": control_path,
        "control_sha256": sha(control_path),
        "previous_recovery_sha256": OLD_SHA,
        "previous_continuation": prior_continuation,
        "original_control_sha256": old.ORIGINAL_SCRIPT_SHA,
        "original_ten_hour_deadline": prior_continuation["original_ten_hour_deadline"],
        "new_account_mutations": 0,
        "remote_publication_deferred": True,
        "numerical_functions_or_recipe_changed": False,
    }
    ns = dict(
        __name__="colab_exact_epoch2_recovery",
        __file__=str(CONTROL / "actual_colab_train.py"),
        RecoveryStore=LocalStore,
        ORIGINAL=FIRST,
        PRIOR=PRIOR,
        RECOVERY=recovery,
        FIRST_PUBLICATION=first,
        SECOND_LOCAL={
            "status": "original_epoch002_retained",
            "files": intent["files"],
            "remote_preservation_verified": False,
        },
    )
    body = transform((CONTROL / "actual_colab_train.py").read_text(), old)
    exec(compile(body, "exact_original_fit_epoch2_local_storage", "exec"), ns)
    print(json.dumps({"phase": "exact_epoch002_resume_admitted", **recovery}), flush=True)
    ns["main"](None)


if __name__ == "__main__":
    main()
