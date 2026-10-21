"""Exercise actual staged Colab compilation and canonical existing-Dataset adoption."""

import ast
import contextlib
import copy
import importlib.util
import json
import sqlite3
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CTL = load("Colab_account_test_controls", HERE / "controls.py")
FIX = load("Colab_account_test_fixtures", CTL.SERVING / "test_serving.py")
ADOPT = load("Colab_account_test_adopt", HERE / "adopt.py")
PREPARE = load("Colab_account_test_prepare", HERE / "prepare.py")


@pytest.fixture
def capsule(tmp_path):
    adapter, evidence = FIX.case.__wrapped__()
    config = dict(
        root=CTL.ROOT_PREFIX + "synthetic",
        owning_node="evc27",
        Colab_evidence={k: v for k, v in evidence.items() if k != "license_bytes"},
    )
    files = CTL.build(config)
    root = tmp_path / "private-final"
    for name, text in files.items():
        path = root / "control" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    compiler = load("Colab_actual_staged_facade", root / "control/serving/compile.py")
    compiler.adapter._base.LICENSE = adapter._base.LICENSE
    policy = load("Colab_actual_staged_policy", root / "control/policy.py")
    return root, config, evidence, compiler, policy, files


def test_actual_staged_compiler_and_account_binding(capsule, monkeypatch):
    _, _, evidence, compiler, policy, files = capsule
    parent = "synthetic parent source"
    compiler.adapter._base.PARENT_SOURCE = policy.sha(parent)
    FIX.rewrite_binding(evidence, lambda b: b.update(parent_source_sha256=policy.sha(parent)))
    monkeypatch.setattr(compiler.implementation.C, "Contract", lambda *_a, **_kw: compiler.adapter)
    slug = evidence["epoch_publications"][-1]["slug"]
    source, receipt = compiler.build(
        parent, [evidence], mode="qualification", model_dataset_ref=slug
    )
    embedded = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"FILES", "BINDING", "PERMISSION", "PARENT_SOURCE"}
    }
    original = files["serving/worker.py.txt"]
    for name, value in embedded.items():
        original = original.replace("__" + name + "_VALUE__", repr(value))
    assert original == source
    assert embedded["BINDING"]["branch"] == "Colab-C100-seed20261021"
    assert receipt["actual_trained_GPU_execution"] is False
    assert "sdk_create_once" not in files["account.py"]
    assert '"CreateCodeSubmission"' in files["account.py"]
    assert '"fileName": "submission.csv"' in files["account.py"]
    assert "require_adopted" in files["account.py"]
    assert "protect_primary_budget" in files["account.py"]
    assert policy.ARMS == ("C",)
    assert policy.FAMILY not in (
        policy.PRIMARY_FAMILY,
        "licensed-parent-full-epoch-serving:predictive-C100-final-v1:",
    )
    assert policy.sha(files["common/workflow.py"]) == policy.WORKFLOW_SHA
    assert policy.sha(files["common/gpu_capacity.py"]) == policy.CAPACITY_SHA


@pytest.mark.parametrize("fault", ["node", "path", "incomplete", "epochs", "unpreserved"])
def test_cannot_stage_unfinished_or_wrong_training(capsule, fault):
    config = copy.deepcopy(capsule[1])
    t = json.loads(config["Colab_evidence"]["terminal_text"])
    if fault == "node":
        config["owning_node"] = "evc105"
    elif fault == "path":
        config["root"] = "/tmp/unrelated"
    elif fault == "incomplete":
        t["updates"] = 140
    elif fault == "epochs":
        t["epochs"] = 5
    else:
        t["all_epochs_private_verified"] = False
    config["Colab_evidence"]["terminal_text"] = json.dumps(t)
    with pytest.raises(ValueError):
        CTL.build(config)


def adopt_environment(capsule, monkeypatch, fail_download=None):
    root, _, evidence, compiler, policy, _ = capsule
    canonical = root.parent / "account"
    canonical.mkdir()
    monkeypatch.setattr(ADOPT, "LEDGER", canonical / "ledger.sqlite3")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    db = sqlite3.connect(canonical / "ledger.sqlite3")
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    lease = json.loads((root / "control/original-Colab-lease.json").read_text())
    db.execute("INSERT INTO meta VALUES (?,?)", (lease["canonical_ledger_key"], json.dumps(lease)))
    db.commit()
    candidate = compiler.contract.candidate(**evidence)
    prepared = dict(
        bundle_identity=policy.sha(policy.canonical(candidate)),
        models=[candidate],
        all_publications_sha256=candidate["all_publications_sha256"],
    )
    (root / "prepared.json").write_text(json.dumps(prepared))
    (root / "source-permission.txt").write_bytes(evidence["license_bytes"])
    (root / "colab-evidence.private.json").write_text(
        json.dumps({k: v for k, v in evidence.items() if k != "license_bytes"})
    )
    receipts = evidence["epoch_publications"] + [evidence["terminal_publication"]]
    by_slug = {r["slug"]: r for r in receipts}
    calls, downloads = [], []

    class Remote:
        def call(self, service, method, payload):
            calls.append(method)
            if method == "IntrospectToken":
                return {"active": True, "username": "godlydonuts"}
            assert method == "GetDatasetStatus"  # No create/push/submit path exists.
            return {"status": "READY"}

    @contextlib.contextmanager
    def ledger(*_args):
        yield db

    def write(path, value):
        with path.open("x") as f:
            json.dump(value, f)

    def download(_remote, slug, name, dest, proof):
        downloads.append((slug, name))
        assert proof == by_slug[slug]["files"][name]
        if name == fail_download:
            raise ValueError("synthetic partial download")
        body = {
            "run-start.json.bin": evidence["binding_text"].encode(),
            "terminal.json.bin": evidence["terminal_text"].encode(),
            "source-permission.txt.bin": evidence["license_bytes"],
        }.get(name, b"synthetic checked bytes")
        dest.write_bytes(body)
        return dest

    publication = types.SimpleNamespace(
        metadata=lambda _r, s: {k: by_slug[s][k] for k in ("id", "slug", "version", "private")},
        list_files=lambda _r, s: {n: x["bytes"] for n, x in by_slug[s]["files"].items()},
    )
    modules = {
        "compile.py": compiler,
        "prepare.py": PREPARE,
        "workflow.py": types.SimpleNamespace(
            Remote=Remote, ledger=ledger, verify_account=lambda *_: None, new_json=write
        ),
        "publication.py": publication,
        "colab_storage.py": types.SimpleNamespace(download_file=download),
    }
    cpu = types.SimpleNamespace(
        policy=policy,
        private_path=Path,
        funding=lambda *_: None,
        check_controls=lambda _: dict(source_git="synthetic"),
        read=lambda p: json.loads(p.read_text()),
        write=write,
        load=lambda _n, p: modules[p.name],
    )
    v = dict(phase="adopt", root=str(root), source_git="synthetic", token="synthetic test token")
    return cpu, v, db, lease, calls, downloads, prepared


def test_actual_adoption_records_existing_ids_without_reupload(capsule, monkeypatch):
    cpu, v, db, _, calls, downloads, prepared = adopt_environment(capsule, monkeypatch)
    result = ADOPT.execute(cpu, v)
    assert result["id"] == 10006 and result["version"] == 1
    assert result["new_Dataset_uploads_or_GPU_or_entries"] == 0
    assert not result["full_optimizer_downloaded_again"]
    assert "full-state.private.pt.bin" not in {n for _, n in downloads}
    assert len(downloads) == 7
    assert set(calls) == {"IntrospectToken", "GetDatasetStatus"}
    ADOPT.require_adopted(cpu, Path(v["root"]), prepared, db)
    before = list(downloads)
    with pytest.raises(ValueError, match="reconciliation"):
        ADOPT.execute(cpu, v)
    assert downloads == before


def test_partial_download_cannot_be_repeated_or_marked_accepted(capsule, monkeypatch):
    cpu, v, db, _, _, downloads, _ = adopt_environment(
        capsule, monkeypatch, fail_download="model.pt.bin"
    )
    with pytest.raises(ValueError, match="partial"):
        ADOPT.execute(cpu, v)
    root = Path(v["root"])
    assert (root / "adopt-intent.json").exists() and not (root / "dataset-verified.json").exists()
    assert len(list(db.execute("SELECT key FROM meta"))) == 1
    before = list(downloads)
    with pytest.raises(ValueError):
        ADOPT.execute(cpu, v)
    assert downloads == before


def test_wrong_canonical_lease_cannot_read_model_or_mutate(capsule, monkeypatch):
    cpu, v, db, lease, _, downloads, _ = adopt_environment(capsule, monkeypatch)
    lease["code_sha256"]["actual_colab_train.py"] = "0" * 64
    db.execute("UPDATE meta SET value=?", (json.dumps(lease),))
    db.commit()
    with pytest.raises(ValueError, match="lease"):
        ADOPT.execute(cpu, v)
    assert not downloads and not (Path(v["root"]) / "adopt-intent.json").exists()


def test_changed_metadata_fails_before_download(capsule, monkeypatch):
    cpu, v, _, _, _, downloads, _ = adopt_environment(capsule, monkeypatch)
    original_load = cpu.load

    def load_bad(name, path):
        m = original_load(name, path)
        if path.name == "publication.py":
            return types.SimpleNamespace(
                metadata=lambda *_: {"id": 999, "private": False}, list_files=m.list_files
            )
        return m

    cpu.load = load_bad
    with pytest.raises(ValueError, match="identity"):
        ADOPT.execute(cpu, v)
    assert not downloads and not (Path(v["root"]) / "adopt-intent.json").exists()


def test_actual_staged_CPU_prepares_then_compiles_existing_Dataset(capsule, monkeypatch):
    root, config, evidence, compiler, policy, files = capsule
    cpu = load("Colab_staged_CPU_execution_test", root / "control/cpu.py")
    parent = root / "original-parent.py"
    parent.write_text("synthetic exact parent")
    permission = root / "original-permission.txt"
    permission.write_bytes(evidence["license_bytes"])
    compiler.adapter._base.PARENT_SOURCE = policy.sha(parent.read_bytes())
    compiler.contract.PARENT_SOURCE = policy.sha(parent.read_bytes())
    FIX.rewrite_binding(
        evidence, lambda b: b.update(parent_source_sha256=compiler.contract.PARENT_SOURCE)
    )
    monkeypatch.setattr(compiler.implementation.C, "Contract", lambda *_a, **_kw: compiler.adapter)
    original_load = cpu.load
    monkeypatch.setattr(
        cpu,
        "load",
        lambda name, path: compiler if path.name == "compile.py" else original_load(name, path),
    )
    monkeypatch.setattr(cpu, "funding", lambda *_: None)
    monkeypatch.setattr(cpu, "private_path", Path)
    monkeypatch.setattr(cpu, "check_controls", lambda _: {"source_git": "synthetic"})
    # Keep the actual original root gate, but test within the synthetic temp directory.
    real_str = str
    monkeypatch.setattr(
        cpu,
        "str",
        lambda x: (
            CTL.ROOT_PREFIX + "synthetic" if isinstance(x, Path) and x == root else real_str(x)
        ),
        raising=False,
    )
    v = dict(
        config,
        root=real_str(root),
        source_git="synthetic",
        phase="prepare",
        license_path=real_str(permission),
        parent_source_path=real_str(parent),
        Colab_evidence={k: val for k, val in evidence.items() if k != "license_bytes"},
    )
    prepared = cpu.execute(v)
    assert prepared["dataset_slug"] == evidence["epoch_publications"][-1]["slug"]
    assert not (root / "upload").exists() and not (root / "bundle").exists()
    accepted = dict(
        evidence["epoch_publications"][-1],
        status="original_Colab_bytes_verified_existing_Dataset_admitted",
        original_Colab_downloaded_all_bytes=True,
    )
    (root / "dataset-verified.json").write_text(json.dumps(accepted))
    receipt = cpu.execute(dict(v, phase="compile", mode="qualification"))
    assert receipt["dataset"]["id"] == 10006
    assert receipt["model_files"]["model.pt.bin"] == prepared["models"][0]["model"]
    ast.parse((root / "compiled-qualification/source.private.py").read_text())
    with pytest.raises(ValueError, match="reconciliation"):
        cpu.execute(dict(v, phase="compile", mode="qualification"))
