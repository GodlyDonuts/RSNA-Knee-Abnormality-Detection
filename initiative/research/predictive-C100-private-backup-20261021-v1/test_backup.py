import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = load("storage_adapter_test", HERE / "storage_contract.py")
E = load("export_adapter_test", HERE / "export_only.py")
FIX = load(
    "backup_original_model_tests",
    HERE.parent / "predictive-density100-serving-20261020-v1/test_contract.py",
)


def receipt(epoch=6):
    files = {
        n: {"bytes": 10, "sha256": "a" * 64}
        for n in (
            "model.pt",
            "provenance.json",
            "own-source.json",
            "source-permission.txt",
            "manifest.json",
        )
    }
    return {
        "storage_provider": "Kaggle",
        "epoch": epoch,
        "source_git": "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b",
        "GPU_job": "995351",
        "revision": f"Kaggle:123:v1:epoch-{epoch:03d}",
        "files": files,
        "private_backup": {
            "id": 123,
            "slug": "godlydonuts/rsna-c100-backup456-" + "a" * 12,
            "version": 1,
            "private": True,
            "downloaded_all_bytes": True,
            "original_members_verified_before_upload": True,
            "archive": {"bytes": 100, "sha256": "b" * 64},
            "members": {f"epoch-{epoch:03d}/" + n: p for n, p in files.items()},
        },
    }


def test_real_storage_namespace_not_fake_HF_revision():
    assert S.storage_identity({"revision": "a" * 40})
    for epoch in (4, 5, 6):
        assert S.storage_identity(receipt(epoch))
    for key, bad in [
        ("private", False),
        ("version", 2),
        ("downloaded_all_bytes", False),
        ("members", {}),
        ("id", None),
    ]:
        r = receipt()
        r["private_backup"][key] = bad
        assert not S.storage_identity(r)
    r = receipt()
    r["revision"] = "a" * 40
    assert not S.storage_identity(r)


def test_complete_model_contract_still_rejects_wrong_recipe_and_missing_epochs():
    old = HERE.parent / "predictive-density100-serving-20261020-v1/contract.py"
    ns = {"__name__": "actual_storage_extended_contract"}
    exec(compile(S.extend(old.read_text()), "extended_contract", "exec"), ns)
    original_adapter, entry = FIX.case.__wrapped__()
    adapter = ns["Contract"](FIX.SOURCE, **FIX.options())
    adapter._base.LICENSE = original_adapter._base.LICENSE
    # Original six-HF proof remains valid; the numerical/clinical contract is unchanged.
    expected = original_adapter.candidate(**entry)
    assert adapter.candidate(**entry)["model"] == expected["model"]
    bad = copy.deepcopy(entry)
    bad["epoch_publications"].pop()
    with pytest.raises(ValueError):
        adapter.candidate(**bad)
    FIX.rewrite(entry, lambda p: p["recipe"].update(density_weight=0.1))
    with pytest.raises(ValueError):
        adapter.candidate(**entry)


def test_export_only_transformation_preserves_numerical_checks(tmp_path, monkeypatch):
    parent = load(
        "backup_export_original",
        HERE.parent / "predictive-C100-preservation-recovery-20261021-v1/continue.py",
    )
    c = parent.original()
    original = c.derive.build(REPO)["publish_epoch.py"]
    path = tmp_path / "publish_epoch.py"
    path.write_text(original)
    ns = E.load(path, "f" * 64)
    assert path.read_text() == original
    calls = []

    class Common:
        @staticmethod
        def atomic_new(p, value):
            import json

            p.write_text(json.dumps(value, sort_keys=True))
            calls.append(p.name)

    files = {
        n: tmp_path / n
        for n in ("model.pt", "provenance.json", "own-source.json", "source-permission.txt")
    }
    for n, p in files.items():
        p.write_bytes(n.encode())
    r = ns["upload_and_verify"](None, Common, tmp_path, "logical/prefix", files, None, {"epoch": 5})
    assert r["independent_private_copy_verified"] is False and r["objects"] == 5
    assert calls == ["manifest.json"] and not (tmp_path / "verified.json").exists()
    assert r["files"]["model.pt"]["sha256"] == hashlib.sha256(b"model.pt").hexdigest()
    assert (
        ns["validate_state"].__code__.co_consts
        == load("pure_original_export", path).validate_state.__code__.co_consts
    )


def test_single_backup_creation_and_no_billing_or_GPU_mutation():
    tree = ast.parse((HERE / "backup.py").read_text())
    calls = [ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x, ast.Call)]
    assert calls.count("pub.sdk_create_once") == 1 and calls.count("pub.reconcile") == 1
    source = (HERE / "backup.py").read_text()
    assert "coordinator-claim" in source and "common.publication_lock" in source
    assert "Credit" not in source and "CreateCodeSubmission" not in source


def test_mixed_original_HF_and_actual_Kaggle_receipts_pass_complete_contract():
    original_adapter, entry = FIX.case.__wrapped__()
    science = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
    ns = {"__name__": "mixed_private_storage_contract"}
    source = (HERE.parent / "predictive-density100-serving-20261020-v1/contract.py").read_text()
    exec(compile(S.extend(source), "extended", "exec"), ns)
    options = FIX.options()
    options["full_source_git"] = science
    adapter = ns["Contract"](FIX.SOURCE, **options)
    adapter._base.LICENSE = original_adapter._base.LICENSE
    FIX.rewrite(
        entry,
        lambda p: p.update(source_git=science, publisher_source_git=science, GPU_job="995351"),
    )
    entry["full_terminal"].update(
        source_git=science, publisher_source_git=science, GPU_job="995351"
    )
    entry["full_scheduler"]["job_id"] = "995351"
    for r in entry["epoch_publications"]:
        e = r["epoch"]
        r.update(
            source_git=science,
            GPU_job="995351",
            prefix=f"checkpoints/predictive-C100-full3887-{science[:12]}/C/epoch-{e:03d}",
        )
        if e >= 4:
            b = receipt(e)
            b["private_backup"]["members"] = {
                f"epoch-{e:03d}/" + n: p for n, p in r["files"].items()
            }
            for key in ("storage_provider", "revision", "private_backup"):
                r[key] = b[key]
    result = adapter.candidate(**entry)
    assert result["model"] == entry["publication"]["files"]["model.pt"]
    bad = copy.deepcopy(entry)
    bad["epoch_publications"][3]["private_backup"]["downloaded_all_bytes"] = False
    with pytest.raises(ValueError):
        adapter.candidate(**bad)
