"""Read-only recovery of an original Colab checkpoint publication.

Use only after observing the original publication worker has stopped. This
module cannot upload, create a Dataset, resume fitting, or replace a receipt.
It leaves the frozen trainer/storage source and original journals untouched.
"""

import hashlib
import json
import os
import re
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    require(path.is_file() and not path.is_symlink(), "Original regular file required")
    require(path.stat().st_size < 2**20, "Small original receipt required")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def original_intent(destination, key, lease):
    intent, digest = read_json(destination / "Kaggle-intent.json")
    require(
        lease["status"] == "canonical_isolated_Colab_checkpoint_namespace_admitted"
        and lease["only_one_Colab_writer"] is True
        and lease["automatic_mutation_retry"] is False
        and lease["maximum_private_dataset_creations"] == 7
        and lease["maximum_bytes_per_dataset"] == 2 * 2**30
        and lease["seed"] == 20261021,
        "Original isolated private namespace lease required",
    )
    slug = lease["slugs"][key]
    require(
        key in {f"epoch-{n:03d}" for n in range(1, 7)} | {"terminal"}
        and slug == f"godlydonuts/rsna-colab-c100-{key}-bce57e766fe4"
        and intent["lease"] == lease
        and intent["slug"] == slug
        and intent["automatic_retry"] is False,
        "Original exact lease and Dataset intent required",
    )
    names = (
        {"terminal.json.bin"}
        if key == "terminal"
        else {
            n + ".bin"
            for n in (
                "model.pt",
                "full-state.private.pt",
                "run-start.json",
                "source-permission.txt",
                "colab_train.py",
                "colab_storage.py",
                "kaggle_io.py",
            )
        }
    )
    proofs = intent["files"]
    require(set(proofs) == names, "Complete original object membership required")
    for proof in proofs.values():
        require(
            set(proof) == {"bytes", "sha256"}
            and type(proof["bytes"]) is int
            and proof["bytes"] > 0
            and isinstance(proof["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", proof["sha256"]),
            "Complete original object byte/hash proof required",
        )
    require(sum(p["bytes"] for p in proofs.values()) < 2 * 2**30, "Original storage cap")
    return intent, digest


def reconcile(remote, publication, destination, key, lease, receipt, *, stopped_observation):
    """Use original RPC helpers; output is a distinct small reconciliation file.

    An existing original verified.json is reused without another byte download.
    Otherwise all original Dataset bytes are verified, once per explicit call.
    Any failure leaves all originals intact and never retries a mutation.
    """
    destination, receipt = Path(destination), Path(receipt)
    require(stopped_observation is True, "Observe original publisher stopped before reconciliation")
    require(not receipt.exists() and not receipt.is_symlink(), "Do not overwrite reconciliation")
    require(receipt.name != "verified.json", "Preserve the original publication receipt")
    intent, intent_sha = original_intent(destination, key, lease)
    slug, files = intent["slug"], intent["files"]
    original = destination / "verified.json"
    reused = original.exists()
    before = publication.metadata(remote, slug)
    require(
        before["private"] is True
        and before["version"] == 1
        and before["slug"] == slug
        and type(before["id"]) is int
        and before["id"] > 0,
        "Existing private immutable version 1 required",
    )
    status = remote.call(
        "datasets.DatasetApiService",
        "GetDatasetStatus",
        {
            "ownerSlug": "godlydonuts",
            "datasetSlug": slug.split("/")[1],
        },
    )
    require(status.get("status") in ("READY", 5), "Original Dataset is not ready")
    require(
        publication.list_files(remote, slug) == {n: p["bytes"] for n, p in files.items()},
        "Existing Dataset membership or sizes differ",
    )
    result = {
        "status": "private_epoch_full_download_size_SHA_verified",
        **before,
        "files": files,
        "revision": f"Kaggle:{before['id']}:v1",
        "key": key,
        "new_GPU_or_competition_calls": 0,
    }
    if reused:
        previous, _ = read_json(original)
        require(previous == result, "Original publication proof differs")
    else:
        for name, proof in files.items():
            publication.stream_file(remote, slug, name, proof, max_seconds=1800)
    require(publication.metadata(remote, slug) == before, "Original Dataset identity changed")
    require(
        read_json(destination / "Kaggle-intent.json")[1] == intent_sha,
        "Original intent changed during reconciliation",
    )
    require(original.exists() == reused, "Original publication worker changed state")
    output = {
        "status": "existing_private_Colab_publication_reconciled",
        "original_intent_sha256": intent_sha,
        "original_verified_receipt_reused": reused,
        "all_original_bytes_downloaded_in_this_reconciliation": not reused,
        "publication": result,
        "new_Dataset_create_upload_GPU_or_competition_calls": 0,
        "training_resumed": False,
    }
    with os.fdopen(os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(output, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    return output
