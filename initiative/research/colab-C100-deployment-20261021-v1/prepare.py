"""Prepare actual Colab final metadata; original weights already live in Kaggle."""

import json
import os
import shutil


def entry(root):
    value = json.loads((root / "colab-evidence.private.json").read_text())
    value["license_bytes"] = (root / "source-permission.txt").read_bytes()
    return value


def prepare(cpu, v, root, compiler):
    p = cpu.policy
    p.no_prior_intent(root / "prepare-intent.json", {}, "prepare")
    evidence = json.loads(p.canonical(v["Colab_evidence"]))
    license_path = cpu.private_path(v["license_path"])
    parent = cpu.private_path(v["parent_source_path"])
    p.require(
        license_path.is_file()
        and parent.is_file()
        and license_path.stat().st_size < 2**20
        and parent.stat().st_size < 2**20,
        "Original small parent and license required",
    )
    evidence["license_bytes"] = license_path.read_bytes()
    chosen = compiler.contract.candidate(**evidence)
    p.require(
        cpu.digest(parent) == compiler.contract.PARENT_SOURCE, "Original parent source differs"
    )
    identity = p.sha(p.canonical(chosen))
    cpu.write(
        root / "prepare-intent.json",
        dict(
            bundle_identity=identity,
            automatic_retry=False,
            new_dataset_or_GPU_or_entries=0,
        ),
    )
    evidence.pop("license_bytes")
    cpu.write(root / "colab-evidence.private.json", evidence)
    for src, name in ((license_path, "source-permission.txt"), (parent, "parent.private.py")):
        destination = root / name
        p.require(not destination.exists(), "Do not overwrite original prepared source")
        shutil.copyfile(src, destination)
        os.chmod(destination, 0o600)
    result = dict(
        status="actual_Colab_fixed_final_existing_Dataset_prepared_CPU_only",
        source_git=v["source_git"],
        bundle_identity=identity,
        dataset_slug=chosen["model_dataset_ref"],
        dataset_id=chosen["model_dataset_id"],
        files=evidence["epoch_publications"][-1]["files"],
        models=[chosen],
        all_publications_sha256=chosen["all_publications_sha256"],
        new_Dataset_uploads_or_GPU_or_entries=0,
    )
    cpu.write(root / "prepared.json", result)
    return result


def compile_final(cpu, v, root, compiler):
    p = cpu.policy
    prepared = cpu.read(root / "prepared.json")
    dataset = cpu.read(root / "dataset-verified.json")
    p.require(
        dataset["status"] == "original_Colab_bytes_verified_existing_Dataset_admitted"
        and dataset["original_Colab_downloaded_all_bytes"] is True
        and dataset["files"] == prepared["files"]
        and dataset["slug"] == prepared["dataset_slug"]
        and dataset["id"] == prepared["dataset_id"]
        and dataset["private"] is True
        and dataset["version"] == 1,
        "Exact original Colab publication and canonical admission required",
    )
    mode = v["mode"]
    p.require(mode in ("qualification", "C"), "Only the independent Colab candidate")
    target = root / ("compiled-" + mode)
    p.require(not target.exists(), "Existing compilation needs read-only reconciliation")
    qualification = (
        None if mode == "qualification" else cpu.read(root / "qualification-accepted.json")
    )
    source, receipt = compiler.build(
        (root / "parent.private.py").read_text(),
        [entry(root)],
        mode="qualification" if mode == "qualification" else "competition",
        qualification=qualification,
        model_dataset_ref=dataset["slug"],
    )
    target.mkdir(mode=0o700)
    (target / "source.private.py").write_text(source)
    result = {
        **receipt,
        "bundle_identity": prepared["bundle_identity"],
        "dataset": {k: dataset[k] for k in ("id", "slug", "version", "private")},
    }
    cpu.write(target / "compiled.json", result)
    return result
