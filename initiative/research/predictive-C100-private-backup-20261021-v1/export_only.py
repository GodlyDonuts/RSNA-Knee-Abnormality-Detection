"""Reuse the exact original checked exporter; defer only its unavailable HF transport."""

from pathlib import Path


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen exporter anchor differs")
    return source.replace(old, new)


def load(publisher, adapter_sha256):
    assert len(adapter_sha256) == 64
    source = Path(publisher).read_text()
    old = """        api = HfApi(token=v["token"])
        require(api.repo_info(REPO, repo_type="dataset").private is True, "Private repo required")
        require(
            not list(api.get_paths_info(REPO, paths=[prefix], repo_type="dataset")),
            "Existing HF path requires read-only reconciliation",
        )"""
    source = replace(
        source,
        old,
        "        api = None  # No HF request; export remains private on the owning node.",
    )
    source = replace(
        source,
        '"actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"',
        '"local_original_export_ready_for_private_backup"',
    )
    source = replace(
        source,
        'common.atomic_new(journal / "verified.json", result)',
        'common.atomic_new(journal / "export-ready.json", result)',
    )
    ns = {"__file__": str(publisher), "__name__": "original_export_private_backup"}
    exec(compile(source, str(publisher), "exec"), ns)

    def stage(_api, common, journal, prefix, payload, _operation, receipt):
        table = {
            n: {"bytes": p.stat().st_size, "sha256": ns["digest"](p)} for n, p in payload.items()
        }
        common.atomic_new(journal / "manifest.json", {"files": table, **receipt})
        table["manifest.json"] = {
            "bytes": (journal / "manifest.json").stat().st_size,
            "sha256": ns["digest"](journal / "manifest.json"),
        }
        return {
            **receipt,
            "prefix": prefix,
            "files": table,
            "objects": 5,
            "independent_private_copy_verified": False,
            "export_adapter_sha256": adapter_sha256,
        }

    ns["upload_and_verify"] = stage
    return ns
