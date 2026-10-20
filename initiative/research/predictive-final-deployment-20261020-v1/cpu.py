"""Zero-GPU owning-node preparation/compilation, invoked explicitly by the root owner."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = globals().get("_staged_policy") or load("predictive_deployment_policy", D / "policy.py")


def read(path):
    path = Path(path)
    policy.require(path.stat().st_size < 16 * 2**20, "Small immutable receipt/source required")
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def private_path(value):
    path = Path(value)
    policy.require(
        path.is_absolute()
        and str(path).startswith("/tmp/")
        and not path.is_symlink()
        and path.resolve() == path
        and path.stat().st_uid == os.getuid(),
        "Existing owning-node original /tmp asset required",
    )
    return path


def check_controls(root):
    stage = read(root / "staged.json")
    for name, expected in stage["code_sha256"].items():
        policy.require(digest(root / "control" / name) == expected, "Frozen control changed")
    return stage


def funding(v, root):
    policy.require(
        os.uname().nodename == v["owning_node"]
        and not os.environ.get("SLURM_JOB_GPUS")
        and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
        "One CPU allocation on the actual owning node, zero GPUs required",
    )
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(x.split() for x in quota.splitlines() if x.strip().startswith("/home/sa305415"))
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    policy.storage(
        v["funding"],
        node_free=shutil.disk_usage("/tmp").free,
        shared_free=(int(q[2]) - int(q[1])) * 1024,
        shared_files=int(q[6]) - int(q[5]),
        used=used,
    )


def scheduler(job):
    policy.require(str(job).isdecimal(), "Original production Slurm job ID required")
    text = subprocess.check_output(
        ["sacct", "-n", "-P", "-j", str(job), "--format=JobIDRaw,State,ExitCode"], text=True
    )
    rows = [line.split("|") for line in text.splitlines() if line.split("|")[0] == str(job)]
    policy.require(len(rows) == 1, "Unique original parent Slurm terminal required")
    return dict(job_id=rows[0][0], state=rows[0][1], exit_code=rows[0][2])


def entry(root, arm):
    base = root / "bundle" / arm
    return dict(
        publication=read(root / "original" / (arm + "-publication.json")),
        provenance_text=(base / "provenance.json").read_text(),
        own_source_text=(base / "own-source.json").read_text(),
        license_bytes=(base / "source-permission.txt").read_bytes(),
    )


def execute(v):
    os.umask(0o077)
    root = Path(v["root"])
    policy.require(
        str(root).startswith("/tmp/rsna-predictive-final-deployment-20261020-v1-")
        and ".." not in root.parts
        and root.resolve() == root,
        "Distinct explicitly named node-local control root required",
    )
    funding(v, root)
    phase = v["phase"]
    if phase == "stage":
        policy.require(not root.exists(), "Existing stage requires read-only reconciliation")
        sources = v["sources"]
        policy.require(
            {name: policy.sha(text) for name, text in sources.items()} == v["code_sha256"]
            and sum(len(text.encode()) for text in sources.values()) < 16 * 2**20,
            "Exact small frozen source payload required",
        )
        for name in sources:
            policy.require(
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Confined own control paths required",
            )
        policy.require(
            v["code_sha256"]["common/workflow.py"] == policy.WORKFLOW_SHA
            and v["code_sha256"]["common/gpu_capacity.py"] == policy.CAPACITY_SHA,
            "Original canonical workflow and corrected capacity helper required",
        )
        root.mkdir(mode=0o700)
        for name, text in sources.items():
            path = root / "control" / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("x") as handle:
                handle.write(text)
        result = dict(
            status="predictive_final_deployment_controls_staged",
            source_git=v["source_git"],
            code_sha256=v["code_sha256"],
            owning_node=v["owning_node"],
            CPU_job=os.environ["SLURM_JOB_ID"],
        )
        write(root / "staged.json", result)
        return result
    stage = check_controls(root)
    policy.require(stage["source_git"] == v["source_git"], "Pinned launch source required")
    compiler = load("predictive_final_compiler", root / "control/serving/compile.py")
    pub = load("predictive_deployment_publication", root / "control/common/publication.py")
    if phase == "prepare":
        intent = root / "prepare-intent.json"
        policy.no_prior_intent(intent, {}, "prepare")
        policy.require(
            set(v["original_runs"]) == set(policy.ARMS), "All three actual finals required"
        )
        candidates, originals, inputs = [], {}, {}
        for arm in policy.ARMS:
            refs = v["original_runs"][arm]
            run = private_path(refs["run_root"])
            original = run / "publication/epoch-006"
            publication = read(original / "verified.json")
            license_path = Path(refs["license_path"])
            policy.require(
                license_path.is_file()
                and not license_path.is_symlink()
                and license_path.stat().st_uid == os.getuid()
                and license_path.stat().st_size < 2**20,
                "Original owned small license file required",
            )
            value = dict(
                publication=publication,
                provenance_text=(original / "provenance.json").read_text(),
                own_source_text=(original / "own-source.json").read_text(),
                license_bytes=license_path.read_bytes(),
            )
            chosen = compiler.contract.candidate(**value)
            terminal = read(run / "terminal.json")
            actual = scheduler(chosen["provenance"]["GPU_job"])
            policy.final_outcome(chosen, terminal, actual)
            policy.require(chosen["arm"] == arm, "Actual role differs")
            for name, item in publication["files"].items():
                path = license_path if name == "source-permission.txt" else original / name
                policy.require(
                    path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
                    "Exact original five-object publication bytes required",
                )
            candidates.append(chosen)
            inputs[arm] = (original, license_path, value)
            originals[arm] = dict(publication=publication, terminal=terminal, scheduler=actual)
        policy.require(
            len({c["provenance"]["source_git"] for c in candidates}) == 1
            and len({c["provenance"]["initial_shared_tensor_sha256"] for c in candidates}) == 1,
            "Matched source and initialization required",
        )
        parent = private_path(v["parent_source_path"])
        policy.require(
            digest(parent) == compiler.contract.PARENT_SOURCE, "Exact private parent source"
        )
        identity = policy.sha(
            policy.canonical(dict(models=candidates, original_outcomes=originals))
        )
        dataset_slug = "godlydonuts/rsna-predictive-finals-" + identity[:12]
        write(
            intent,
            dict(
                bundle_identity=identity,
                original_outcomes=originals,
                automatic_retry=False,
                new_dataset_or_GPU_or_entry_mutations=0,
            ),
        )
        bundle = root / "bundle"
        bundle.mkdir()
        (root / "original").mkdir()
        files = {}
        for chosen in candidates:
            arm = chosen["arm"]
            original, license_path, value = inputs[arm]
            (bundle / arm).mkdir()
            write(root / "original" / (arm + "-publication.json"), value["publication"])
            for name, item in value["publication"]["files"].items():
                source = license_path if name == "source-permission.txt" else original / name
                destination = chosen["model_filename"] if name == "model.pt" else arm + "/" + name
                shutil.copyfile(source, bundle / destination)
                os.chmod(bundle / destination, 0o600)
                files[destination] = item
        write(
            bundle / "bundle-manifest.json",
            dict(bundle_identity=identity, files=files, models=candidates, private=True),
        )
        files["bundle-manifest.json"] = {
            "bytes": (bundle / "bundle-manifest.json").stat().st_size,
            "sha256": digest(bundle / "bundle-manifest.json"),
        }
        (root / "upload").mkdir()
        archive = pub.build_zip(bundle, files, root / "upload/bundle.zip")
        shutil.copyfile(parent, root / "parent.private.py")
        result = dict(
            status="actual_three_predictive_final_exports_prepared_CPU_only",
            bundle_identity=identity,
            dataset_slug=dataset_slug,
            files=files,
            models=candidates,
            archive=archive,
            original_outcomes=originals,
            source_git=stage["source_git"],
            new_dataset_or_GPU_or_entry_mutations=0,
        )
        write(root / "prepared.json", result)
    elif phase == "compile":
        prepared, dataset = read(root / "prepared.json"), read(root / "dataset-verified.json")
        policy.require(
            dataset["files"] == prepared["files"]
            and dataset["slug"] == prepared["dataset_slug"]
            and dataset["status"] == "all_payload_bytes_verified"
            and dataset["downloaded_all_bytes"] is True
            and dataset["private"] is True
            and dataset["version"] == 1,
            "Actual unique private Dataset fully re-downloaded and verified required",
        )
        policy.require(
            all(
                dataset["remote_names"][c["model_filename"]] == c["model_filename"]
                for c in prepared["models"]
            ),
            "Verified Dataset model mount has a wrapper: bind exact relative paths before compilation",
        )
        mode = v["mode"]
        policy.require(mode == "qualification" or mode in policy.ARMS, "Explicit compile role")
        target = root / ("compiled-" + mode)
        policy.require(not target.exists(), "Existing compilation requires reconciliation")
        qualification = (
            None if mode == "qualification" else read(root / "qualification-accepted.json")
        )
        values = [entry(root, a) for a in policy.ARMS if mode == "qualification" or a == mode]
        source, receipt = compiler.build(
            (root / "parent.private.py").read_text(),
            values,
            mode="qualification" if mode == "qualification" else "competition",
            qualification=qualification,
            model_dataset_ref=dataset["slug"],
        )
        target.mkdir()
        (target / "source.private.py").write_text(source)
        result = {
            **receipt,
            "bundle_identity": prepared["bundle_identity"],
            "dataset": {k: dataset[k] for k in ("id", "slug", "version", "private")},
        }
        write(target / "compiled.json", result)
    else:
        raise ValueError("Only explicit stage, prepare or compile CPU phases are implemented")
    funding(v, root)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin)), allow_nan=False), flush=True)
    except Exception as error:
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(error).__name__, "automatic_retry": False}
            ),
            flush=True,
        )
        raise SystemExit(1) from None
