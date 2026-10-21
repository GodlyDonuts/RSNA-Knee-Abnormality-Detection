"""Admit seven existing private Colab publications under the canonical account claim.

No create/upload/kernel/submission API is available in this phase. Original
Colab whole-byte proofs are retained, while final serving bytes are downloaded
once onto private deployment storage. A partial attempt is never auto-retried.
"""

import importlib.util
import json
import os
from pathlib import Path

LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")


def check_lease(p, db, expected):
    found = db.execute(
        "SELECT value FROM meta WHERE key=?", (expected["canonical_ledger_key"],)
    ).fetchone()
    p.require(
        found is not None and json.loads(found[0]) == expected,
        "Actual original canonical Colab lease changed or is missing",
    )


def live_publications(p, remote, publication, receipts, lease):
    p.require(len(receipts) == 7, "All six epochs plus terminal required")
    for value in receipts:
        slug = value["slug"]
        p.require(lease["slugs"][value["key"]] == slug, "Original namespace differs")
        expected = {k: value[k] for k in ("id", "slug", "version", "private")}
        p.require(publication.metadata(remote, slug) == expected, "Private v1 identity differs")
        status = remote.call(
            "datasets.DatasetApiService",
            "GetDatasetStatus",
            {
                "ownerSlug": "godlydonuts",
                "datasetSlug": slug.split("/")[1],
            },
        )
        p.require(status.get("status") in ("READY", 5), "Original Dataset is not ready")
        p.require(
            publication.list_files(remote, slug)
            == {n: x["bytes"] for n, x in value["files"].items()},
            "Original exact published file listing differs",
        )


def require_adopted(cpu, root, prepared, db):
    p = cpu.policy
    lease = cpu.read(root / "control/original-Colab-lease.json")
    check_lease(p, db, lease)
    key = p.FAMILY + prepared["bundle_identity"] + ":dataset"
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    accepted = cpu.read(root / "dataset-verified.json")
    p.require(
        row is not None and json.loads(row[0]) == accepted,
        "Canonical original Dataset admission must equal local proof",
    )
    p.require(
        accepted["all_publications_sha256"] == prepared["all_publications_sha256"]
        and accepted["bundle_identity"] == prepared["bundle_identity"],
        "Do not borrow another Colab final or publication proof",
    )


def execute(cpu, v):
    import requests

    os.umask(0o077)
    p = cpu.policy
    p.require(v["phase"] == "adopt", "Existing Dataset admission only")
    root = cpu.private_path(v["root"])
    cpu.funding(v, root)
    stage = cpu.check_controls(root)
    p.require(stage["source_git"] == v["source_git"], "Frozen controls required")
    prepared = cpu.read(root / "prepared.json")
    helper = cpu.load("Colab_adoption_prepare", root / "control/prepare.py")
    compiler = cpu.load("Colab_adoption_compiler", root / "control/serving/compile.py")
    evidence = helper.entry(root)
    candidate = compiler.contract.candidate(**evidence)
    p.require(prepared["models"] == [candidate], "Original prepared final changed")
    receipts = evidence["epoch_publications"] + [evidence["terminal_publication"]]
    lease = cpu.read(root / "control/original-Colab-lease.json")
    w = cpu.load("Colab_adoption_workflow", root / "control/common/workflow.py")
    pub = cpu.load("Colab_adoption_publication", root / "control/common/publication.py")
    download = cpu.load("Colab_adoption_download", root / "control/colab_storage.py").download_file
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    who = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    p.require(
        who.get("active") is True and who.get("username") == "godlydonuts", "Owner identity differs"
    )
    remote.username = who["username"]
    claim = LEDGER.parent / "coordinator-claim"
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        return {"status": "waiting_existing_canonical_claim_no_mutation"}
    owner = dict(
        kind="Colab_existing_private_final_admission",
        CPU_job=os.environ["SLURM_JOB_ID"],
        bundle_identity=prepared["bundle_identity"],
    )
    w.new_json(claim / "owner.json", owner)
    try:
        with w.ledger(LEDGER, "godlydonuts") as db:
            w.verify_account(db, remote)
            check_lease(p, db, lease)
            key = p.FAMILY + prepared["bundle_identity"] + ":dataset"
            p.require(
                db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone() is None,
                "Existing canonical admission requires reconciliation",
            )
            p.no_prior_intent(root / "adopt-intent.json", {}, "adopt")
            live_publications(p, remote, pub, receipts, lease)
            cpu.write(
                root / "adopt-intent.json",
                dict(
                    **owner,
                    automatic_retry=False,
                    all_publications_sha256=candidate["all_publications_sha256"],
                ),
            )
            destination = root / "original-final"
            destination.mkdir(mode=0o700)
            final = receipts[-2]
            # Keep the original optimizer archive remote; Colab already verified it.
            names = sorted(set(final["files"]) - {"full-state.private.pt.bin"})
            for name in names:
                download(remote, final["slug"], name, destination / name, final["files"][name])
            download(
                remote,
                receipts[-1]["slug"],
                "terminal.json.bin",
                destination / "terminal.json.bin",
                receipts[-1]["files"]["terminal.json.bin"],
            )
            p.require(
                (destination / "run-start.json.bin").read_bytes()
                == evidence["binding_text"].encode()
                and (destination / "terminal.json.bin").read_bytes()
                == evidence["terminal_text"].encode()
                and (destination / "source-permission.txt.bin").read_bytes()
                == evidence["license_bytes"],
                "Actual downloaded terminal/binding/license differs",
            )
            live_publications(p, remote, pub, receipts, lease)
            cpu.funding(v, root)
            accepted = dict(
                status="original_Colab_bytes_verified_existing_Dataset_admitted",
                **{k: final[k] for k in ("id", "slug", "version", "private", "files")},
                bundle_identity=prepared["bundle_identity"],
                all_publications_sha256=candidate["all_publications_sha256"],
                original_Colab_downloaded_all_bytes=True,
                original_publications=receipts,
                deployment_downloaded_names=names + ["terminal.json.bin"],
                full_optimizer_downloaded_again=False,
                new_Dataset_uploads_or_GPU_or_entries=0,
            )
            cpu.write(root / "dataset-verified.json", accepted)
            db.execute("INSERT INTO meta VALUES (?,?)", (key, json.dumps(accepted)))
            db.commit()
            return accepted
    finally:
        p.require(cpu.read(claim / "owner.json") == owner, "Account claim changed")
        (claim / "owner.json").unlink()
        claim.rmdir()


if __name__ == "__main__":
    import sys

    spec = importlib.util.spec_from_file_location(
        "Colab_adoption_cpu", Path(__file__).with_name("cpu.py")
    )
    cpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu)
    print(json.dumps(execute(cpu, json.load(sys.stdin))), flush=True)
