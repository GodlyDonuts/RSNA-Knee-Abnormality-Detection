"""Owning-node packer and canonical Dataset phases for the existing pointwise final."""

# Executed inside the frozen original account-program namespace.
# ruff: noqa: F821
import sqlite3

MODEL_SHA = "da61c53480931166cd13c89a30f8062e386353e77c257fc9a94dd26afb215084"
MODEL_BYTES = 292867251
MODEL_KEY = "licensed-parent-full-epoch-serving:pointwise-final:" + MODEL_SHA
DATASET_SLUG = "godlydonuts/rsna-pointwise-final-e3-" + MODEL_SHA[:12]
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")
ORIGINAL_ROOT = Path("/tmp/rsna-teacher-pair-full-fitting48-20261020-v1")
ORIGINAL_SERVING = ORIGINAL_ROOT / "final-serving-v2/pointwise"
COMPILED_SHA = {
    "qualification": "c398324df327fdfdc541f1db56857764dc823029d5719510dc69d88532a60637",
    "competition": "5d04caf90941aed231e38838528c3fd8cecb626214fd6e81b645810d1b0a2f10",
}
PERMISSION = "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79"


def pointwise_bounds(v):
    assert (
        os.uname().nodename == "evc48"
        and not os.environ.get("SLURM_JOB_GPUS")
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    )
    assert LEDGER.is_file() and not LEDGER.is_symlink()
    f = v["funding"]
    assert f["status"] == "accepted_pointwise_final_separate_node_storage"
    assert f["incremental_node_bytes"] == 640 * 2**20
    assert f["incremental_shared_bytes"] == 2**20 and f["incremental_shared_files"] == 0
    assert f["shared_transient_scope_reuses_existing_canonical_claim"] is True
    assert f["shared_other_liabilities_bytes"] >= 121 * 2**20 and f["shared_future_files"] >= 376
    assert (
        f["node_baseline"] == "original_pair_compiler_conservative_remaining24GiB_plus115GiB136MiB"
    )
    assert f["no_deletion_credit"] is True
    original_paid = sum(
        p.stat().st_size
        for p in ORIGINAL_ROOT.rglob("*")
        if p.is_file() and not p.is_symlink() and "inputs" not in p.relative_to(ORIGINAL_ROOT).parts
    )
    assert original_paid < 24 * 2**30
    other_node_unpaid = 115 * 2**30 + 136 * 2**20 + max(0, 24 * 2**30 - original_paid)
    used = sum(p.stat().st_size for p in D.rglob("*") if p.is_file()) if D.exists() else 0
    assert used <= 640 * 2**20
    assert shutil.disk_usage("/tmp").free > 20 * 2**30 + other_node_unpaid + max(
        0, 640 * 2**20 - used
    )
    q = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    a = next(line.split() for line in q.splitlines() if line.strip().startswith("/home/sa305415"))
    assert (int(a[2]) - int(a[1])) * 1024 > 20 * 2**30 + f["shared_other_liabilities_bytes"] + 2**20
    assert int(a[6]) - int(a[5]) > 5000 + f["shared_future_files"]
    return {
        "node_used_bytes": used,
        "other_node_unpaid_bytes": other_node_unpaid,
        "original_scope_paid_bytes": original_paid,
        "shared_quota_sha256": hashlib.sha256(q.encode()).hexdigest(),
    }


def original_pointwise(v):
    compiled = read(ORIGINAL_SERVING / "compiled.json")
    assert compiled == v["original_compiled"]
    assert (
        compiled["status"] == "actual_H100_pair_fixed_final_serving_sources_compiled"
        and compiled["CPU_job"] == "977528"
    )
    assert compiled["source_git"] == "a0851fc19904bc6b86fd93026f86f5e1b86746d3"
    assert compiled["training_source_git"] == "7f75fd46b5d0e772fd2767400e81d70b593619b4"
    assert (
        compiled["role"] == "pointwise"
        and compiled["epoch"] == 3
        and compiled["model_sha256"] == MODEL_SHA
    )
    for mode, h in COMPILED_SHA.items():
        path = ORIGINAL_SERVING / (mode + ".private.py")
        assert (
            compiled["outputs"][mode] == {"sha256": h, "bytes": path.stat().st_size}
            and sha(path) == h
        )
    training = read(ORIGINAL_SERVING / "actual-accepted.json")
    pub = read(ORIGINAL_SERVING / "publication-private-verified.json")
    validator = ORIGINAL_ROOT / "final-serving-v2/validate.py"
    assert sha(validator) == "d887431cc07a9ac2f1b441b44ee52c3f7ba0720c524f350b8e5ceb202680a147"
    scope = {"__name__": "actual_unchanged_pointwise_validator"}
    validator_text = validator.read_text()
    assert validator_text.count("\nimport torch\n") == 1
    exec(
        compile(
            validator_text.replace("\nimport torch\n", "\n"),
            "original_pointwise_provenance_only",
            "exec",
        ),
        scope,
    )
    scope["provenance"](training, pub, MODEL_SHA)
    assert pub["revision"] == "e36bb88ffa2aa4ac62680a29a5d7ba0fdd7e7bf6"
    assert pub["model_bytes"] == MODEL_BYTES
    model = ORIGINAL_ROOT / "pointwise/epoch-checkpoints/epoch-003.pt"
    assert model.is_file() and not model.is_symlink() and model.stat().st_size == MODEL_BYTES
    return training, pub, compiled


def inspect_original(v):
    assert not D.exists(), "Existing pointwise deployment needs read-only reconciliation"
    with sqlite3.connect(LEDGER.as_uri() + "?mode=ro", uri=True) as db:
        assert (
            db.execute(
                "SELECT value FROM meta WHERE key IN (?,?)", (MODEL_KEY, MODEL_KEY + ":dataset")
            ).fetchone()
            is None
        )
    if (LEDGER.parent / "coordinator-claim").exists():
        return {"status": "waiting_existing_canonical_claim_no_mutation"}
    return {
        "status": "actual_pointwise_compiled_and_absent_route_reconciled_read_only",
        "model_sha256": MODEL_SHA,
        "CPU_compiler": "977528",
        "new_mutations": 0,
    }


def prepare_bundle(original):
    publication = load("publication")
    training, pub, compiled = original
    bundle, upload = D / "bundle", D / "upload"
    bundle.mkdir(mode=0o700)
    upload.mkdir(mode=0o700)
    journal = ORIGINAL_ROOT / "publication/pointwise/epoch-003"
    intent = read(journal / "intent.json")
    paths = {
        "epoch-003.pt": ORIGINAL_ROOT / "pointwise/epoch-checkpoints/epoch-003.pt",
        "run.json": ORIGINAL_ROOT / "pointwise/run.json",
        "source-permission.txt": Path(
            "/tmp/rsna-S6-earlier-H100-route-20261019-v1/auxiliary-v2/source/source-permission.txt"
        ),
        "own-source.json": journal / "own-source.json",
        "provenance.json": journal / "provenance.json",
        "manifest.json": journal / "manifest.json",
    }
    assert sha(paths["source-permission.txt"]) == PERMISSION
    files = {}
    for name, path in paths.items():
        original_name = "model.pt" if name == "epoch-003.pt" else name
        expected = intent["files"][original_name]
        assert (
            path.stat().st_size == expected["bytes"]
            and publication.digest(path) == expected["sha256"]
        )
        destination = bundle / name
        shutil.copyfile(path, destination)
        os.chmod(destination, 0o600)
        files[name] = expected
    assert files["epoch-003.pt"] == {"bytes": MODEL_BYTES, "sha256": MODEL_SHA}
    manifest = {
        "model_sha256": MODEL_SHA,
        "files": files,
        "private": True,
        "original_publication": pub,
        "compiled": compiled,
        "inference_source_recompiled": False,
        "new_training": False,
    }
    w = load("workflow")
    w.new_json(bundle / "bundle-manifest.json", manifest)
    files["bundle-manifest.json"] = {
        "bytes": (bundle / "bundle-manifest.json").stat().st_size,
        "sha256": sha(bundle / "bundle-manifest.json"),
    }
    archive = publication.build_zip(bundle, files, upload / "bundle.zip")
    w.new_json(
        D / "bundle-prepared.json",
        {
            "dataset_slug": DATASET_SLUG,
            "files": files,
            "archive": archive,
            "model_sha256": MODEL_SHA,
        },
    )


def dataset_phase(v, db, remote, w):
    publication = load("publication")
    prepared = read(D / "bundle-prepared.json")
    key = MODEL_KEY + ":dataset"
    found = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    row = json.loads(found[0]) if found else {}
    if v["phase"] == "dataset":
        assert not row and not (D / "dataset-intent.json").exists(), (
            "Existing Dataset state: never recreate"
        )
        absence = publication.require_absent(remote, DATASET_SLUG)
        publication.validate_zip(D / "upload/bundle.zip", prepared["files"])
        assert sha(D / "upload/bundle.zip") == prepared["archive"]["sha256"]
        row = {
            "state": "pointwise_dataset_create_intent",
            "slug": DATASET_SLUG,
            "version": 1,
            "model_sha256": MODEL_SHA,
            "automatic_retry": False,
            "absence": absence,
            "files": prepared["files"],
        }
        w.new_json(D / "dataset-intent.json", row)
        db.execute("INSERT INTO meta VALUES (?,?)", (key, json.dumps(row)))
        db.commit()
        try:
            response = publication.sdk_create_once(
                v["token"], DATASET_SLUG, "RSNA pointwise final epoch3", D / "upload"
            )
        except BaseException:
            w.new_json(D / "dataset-uncertain.json", {"automatic_retry": False})
            raise
        w.new_json(D / "dataset-response.json", response)
        row["state"] = "pointwise_dataset_create_returned"
        db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(row), key))
        db.commit()
        return {
            "status": "pointwise_dataset_create_returned",
            "new_Dataset_calls": 1,
            "new_GPU_or_entries": 0,
        }
    assert row and row["model_sha256"] == MODEL_SHA
    if (D / "dataset-verified.json").exists():
        return verified_dataset(remote)
    owner, name = DATASET_SLUG.split("/")
    state = remote.call(
        "datasets.DatasetApiService", "GetDatasetStatus", {"ownerSlug": owner, "datasetSlug": name}
    )
    if state.get("status") not in ("READY", 5):
        return {
            "status": "waiting_original_dataset_ingestion_no_mutation",
            "actual_status": state.get("status"),
        }
    result = publication.reconcile(remote, DATASET_SLUG, prepared["files"])
    w.new_json(D / "dataset-verified.json", result)
    row.update(state="pointwise_dataset_fully_verified", verification=result)
    db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(row), key))
    db.commit()
    return result


def verified_dataset(remote):
    publication = load("publication")
    result, prepared = read(D / "dataset-verified.json"), read(D / "bundle-prepared.json")
    assert (
        result["status"] == "all_payload_bytes_verified" and result["downloaded_all_bytes"] is True
    )
    assert result["files"] == prepared["files"] and result["slug"] == DATASET_SLUG
    assert publication.metadata(remote, DATASET_SLUG) == {
        k: result[k] for k in ("id", "version", "slug", "private")
    }
    # Compiler bounds its name search to five nested levels; admit actual archive mapping.
    mounted = result["remote_names"]["epoch-003.pt"]
    assert Path(mounted).name == "epoch-003.pt" and len(Path(mounted).parts) <= 5
    return result
