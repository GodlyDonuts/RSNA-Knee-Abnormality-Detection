"""Local CPU end-to-end component path executed from one exact Git projection.

The projection is built by ``project_code.build`` from a real commit of this
repository's ``rsna/``, ``scripts/kaggle/`` and ``pyproject.toml``. Only the
Kaggle transport/remote, the root release validator and the delta-base model
reconstruction (both separately tested) are faked. Synthetic bytes only: no
network, credential, weights or account mutation. Engineering behaviour only,
not Kaggle runtime, memory, /tmp composition cost or accuracy evidence.
"""

import hashlib
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_kaggle_shared_wheelhouse import proof

import rsna.competitive.infer as inference

ROOT = Path(__file__).resolve().parents[1]
OWNER = "godlydonuts"
LICENSE = "https://github.com/facebookresearch/dinov3/blob/" + "d" * 40 + "/LICENSE.md"
COMMON_SLUG = OWNER + "/rsna-common-encoder-fixture-v1"
WHEELS = OWNER + "/rsna-wheels-fixture"
PROBE = OWNER + "/rsna-wheel-probe-fixture"
PROBE_SOURCE = "synthetic wheel probe source"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashed(path):
    return {"sha256": digest(path), "bytes": Path(path).stat().st_size}


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_projection(tmp, name="projection", mutate=None):
    """Exact projection of a real commit with this checkout's runtime/coordinator sources."""
    src = tmp / (name + "-git")
    for pattern in ("rsna/**/*.py", "scripts/kaggle/*.py", "scripts/kaggle/*.sh"):
        for path in ROOT.glob(pattern):
            target = src / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copyfile(ROOT / "pyproject.toml", src / "pyproject.toml")
    if mutate is not None:
        mutate(src)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(src), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "fixture")
    commit = git("rev-parse", "HEAD")
    builder = load("fixture_project_code", ROOT / "scripts/kaggle/project_code.py")
    receipt = builder.build(src, commit, commit, tmp / (name + "-build"))
    tree = tmp / name
    tree.mkdir()
    with tarfile.open(tmp / (name + "-build/projection.tar")) as archive:
        archive.extractall(tree, filter="data")
    return SimpleNamespace(
        tree=tree,
        manifest=tmp / (name + "-build/projection.json"),
        sha=receipt["projection_sha256"],
        receipt=receipt,
    )


class Response:
    def __init__(self, data):
        self.status_code, self.data, self.url, self.headers = 200, data, "", {}

    def iter_content(self, size):
        for start in range(0, len(self.data), size):
            yield self.data[start : start + size]

    def close(self):
        pass


class Session:
    def __init__(self, kaggle):
        self.kaggle = kaggle
        self.cookies = SimpleNamespace(clear=lambda: None)

    def post(self, url, json, **kwargs):
        assert url.endswith("datasets.DatasetApiService/DownloadDataset")
        assert json["datasetVersionNumber"] == 1
        self.kaggle.downloads += 1
        slug = json["ownerSlug"] + "/" + json["datasetSlug"]
        return Response(self.kaggle.datasets[slug]["files"][json["fileName"]])


class FakeKaggle:
    """Fake remote: owned Dataset listing/files/download, SaveKernel and wheel probe."""

    username = OWNER

    def __init__(self, receipt):
        self.datasets = {WHEELS: {"id": 42, "files": {}}}
        self.saves, self.creates, self.downloads, self.next_id = [], [], 0, 500
        self.receipt = receipt
        self.session = Session(self)

    def publish(self, slug, files):
        assert slug not in self.datasets
        self.datasets[slug] = {"id": self.next_id, "files": files}
        self.next_id += 1

    def publish_zip(self, slug, archive):
        with zipfile.ZipFile(archive) as z:
            self.publish(slug, {name: z.read(name) for name in z.namelist()})

    def call(self, service, method, body):
        if method == "ListDatasets":
            rows = [{"ref": ref} for ref in sorted(self.datasets)] if body["page"] == 1 else []
            return {"datasets": rows}
        if method == "GetDatasetStatus":
            return {"status": "READY"}
        if method == "ListDatasetFiles":
            files = self.datasets[body["ownerSlug"] + "/" + body["datasetSlug"]]["files"]
            return {
                "datasetFiles": [
                    {"name": name, "totalBytes": len(data)} for name, data in sorted(files.items())
                ]
            }
        if method == "SaveKernel":
            self.saves.append(body)
            return {"ref": "/code/" + body["slug"], "versionNumber": 1, "kernelId": 7}
        raise AssertionError(method)

    def dataset(self, slug):
        return {"id": self.datasets[slug]["id"], "isPrivate": True, "currentVersionNumber": 1}

    def kernel(self, method, slug, version):
        assert slug == PROBE and version == 1
        if method == "GetKernelSessionStatus":
            return {"status": "COMPLETE"}
        return {
            "metadata": {
                "isPrivate": True,
                "enableInternet": False,
                "machineShape": "NvidiaTeslaT4",
                "datasetDataSources": [WHEELS],
                "competitionDataSources": [],
            },
            "blob": {"source": PROBE_SOURCE},
        }

    def output_json(self, slug, version, name):
        assert (slug, version, name) == (PROBE, 1, "wheelhouse-gpu-acceptance.json")
        return self.receipt

    def require_absent_kernel(self, slug):
        if any(save["slug"] == slug for save in self.saves):
            raise ValueError("Dedicated notebook already exists; do not overwrite")
        return {"method": "GetKernel404"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10**15))
    proj = build_projection(tmp_path)
    c = load("projection_coordinator", proj.tree / "scripts/kaggle/coordinator.py")
    wf, _ = c.load_workflow(proj.tree, proj.manifest, proj.sha)
    monkeypatch.setattr(wf, "validate_delta_base", lambda *a: None)
    node_a = tmp_path / "node-a-tmp"
    node_a.mkdir()
    monkeypatch.setattr(c, "NODE_SCRATCH", node_a)

    # Encoder, licence, wheelhouse and shared runtime proof (synthetic bytes).
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = inputs / "config.json"
    config.write_text('{"synthetic": true}\n')
    base = inputs / "base.safetensors"
    base.write_bytes(b"synthetic frozen base; not pretrained" * 256)
    spec = inputs / "encoder.json"
    spec.write_text(
        json.dumps(
            {
                "architecture": "dinov3_l16",
                "config_path": str(config),
                "config_sha256": digest(config),
                "weights_path": str(base),
                "weights_sha256": digest(base),
            }
        )
    )
    provenance = {k: v for k, v in json.loads(spec.read_text()).items() if not k.endswith("_path")}
    licence = inputs / "LICENSE.md"
    licence.write_text("Synthetic licensing fixture for local coordinator tests only.\n")
    wheels = inputs / "wheelhouse"
    wheels.mkdir()
    (wheels / "fixture.whl").write_bytes(b"fixture wheel")
    (wheels / "requirements.lock").write_text("fixture==1 --hash=sha256:" + "a" * 64 + "\n")
    pyproject = digest(proj.tree / "pyproject.toml")
    (wheels / "acceptance.json").write_text(
        json.dumps({"status": "cpu_offline_install_passed", "pyproject_sha256": pyproject})
    )
    (wheels / "wheelhouse.json").write_text(
        json.dumps(
            {
                "platform": "linux_x86_64",
                "python": "3.12",
                "cuda": "13.0",
                "offline_install_verified": True,
                "files": {p.name: hashed(p) for p in sorted(wheels.iterdir())},
                "pyproject_sha256": pyproject,
            }
        )
    )
    shared = proof()
    gpu_receipt = {
        "status": "passed",
        "wheelhouse_archive_sha256": shared["archive_sha256"],
        "wheelhouse_manifest_sha256": digest(wheels / "wheelhouse.json"),
        "cuda_tensor_backward": True,
        "random_tiny_dinov3_cuda_forward": True,
    }
    shared.update(
        dataset_slug=WHEELS,
        gpu_probe_slug=PROBE,
        gpu_probe_source_sha256=hashlib.sha256(PROBE_SOURCE.encode()).hexdigest(),
        gpu_receipt_sha256=object_hash(gpu_receipt),
        manifest_sha256=digest(wheels / "wheelhouse.json"),
        pyproject_sha256=pyproject,
    )
    shared_file = inputs / "shared-wheelhouse.json"
    shared_file.write_text(json.dumps(shared))

    # Real projection code hashes; the root release validator is faked.
    binding = {
        "code_sha256": {
            str(p.relative_to(proj.tree / "rsna")): digest(p)
            for p in sorted((proj.tree / "rsna").rglob("*.py"))
        },
        "config": {
            "fold": 0,
            "seed": 42,
            "submission_every_epochs": 2,
            "encoder": provenance,
            "checkpoint_format": "frozen_base_delta",
        },
        "release": {"binding_sha256": "b" * 64},
    }
    monkeypatch.setattr(inference, "validate_inference_checkpoint", lambda *a, **kw: binding)

    def snapshot(epoch):
        directory = tmp_path / f"epoch-{epoch:03d}"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        torch.save(
            {
                "schema_version": 2,
                "scheduled_epoch": True,
                "epoch": epoch,
                "next_update": 0,
                "step": 3,
                "model_storage": {
                    "format": "frozen_base_delta_v1",
                    "encoder_provenance": provenance,
                },
                "synthetic_delta": torch.full((4,), float(epoch)),
            },
            checkpoint,
        )
        sidecar = directory / "snapshot.json"
        sidecar.write_text(
            json.dumps(
                {
                    "status": "scheduled_epoch_snapshot",
                    "epoch": epoch,
                    "submission_every_epochs": 2,
                    "submission_eligible": True,
                    "diagnostic_only": False,
                    "optimizer_steps": 3,
                    "checkpoint_file": checkpoint.name,
                    "checkpoint_sha256": digest(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "binding_sha256": wf.object_hash(binding),
                    "release_binding_sha256": "b" * 64,
                    "fold": 0,
                    "seed": 42,
                }
            )
        )
        return sidecar, digest(checkpoint)[:12]

    def prepare(workflow, sidecar, output, dataset, kernel, common=None, repo=proj.tree):
        return workflow.prepare_bundle(
            sidecar,
            spec,
            wheels,
            output,
            dataset,
            kernel,
            licence,
            LICENSE,
            digest(licence),
            repo,
            shared_wheelhouse=shared_file,
            common_bundle=common,
        )

    def regenerate(workflow, sidecar, repo=proj.tree):
        return workflow.common_table(
            sidecar, spec, wheels, licence, LICENSE, digest(licence), repo, shared_file
        )

    kaggle = FakeKaggle(gpu_receipt)

    # The original standalone payload (an older checkpoint) and its actual
    # full-download private v1 receipt through the fake transport.
    old_sidecar, old_prefix = snapshot(8)
    standalone = tmp_path / "standalone"
    old_slug = OWNER + "/rsna-standalone-" + old_prefix
    prepare(wf, old_sidecar, standalone, old_slug, OWNER + "/rsna-standalone-run-" + old_prefix)
    data = standalone / "dataset"
    expected = c.p.expected_files(data, json.loads((data / "bundle-manifest.json").read_text()))
    kaggle.publish(old_slug, {name: (data / name).read_bytes() for name in expected})
    receipt = tmp_path / "standalone-receipt.json"
    receipt.write_text(json.dumps(c.p.reconcile(kaggle, old_slug, expected)))
    kaggle.downloads = 0

    sidecar, prefix = snapshot(10)
    regenerated = regenerate(wf, sidecar)
    operation_dir = tmp_path / "common-operation"
    operation = wf.components.prepare_common_operation(
        data,
        operation_dir,
        competition=wf.COMPETITION,
        shared_wheelhouse=shared,
        checkpoint_format="frozen_base_delta",
        dataset_slug=COMMON_SLUG,
        source_receipt=receipt,
        regenerated=regenerated,
    )
    ledger_file = tmp_path / "home/rsna-knee/kaggle/account-ledger.sqlite3"
    with wf.ledger(ledger_file, OWNER):
        pass

    state = SimpleNamespace(outcome=None)

    def management(python):
        def create(token, slug, title, upload):
            kaggle.creates.append(slug)
            kaggle.publish_zip(slug, upload / "bundle.zip")
            if state.outcome is not None:
                raise state.outcome("response lost after upload; detail must not be journaled")
            return {"returned": True, "status": "", "final_create_rpc_calls": 1}

        return create

    monkeypatch.setattr(c, "management_create", management)
    monkeypatch.setattr(c, "remote_from_stdin", lambda w: (kaggle, "fixture-not-a-token"))
    common_root = node_a / ("rsna-kaggle-common-journal-" + operation["manifest_sha256"][:12])

    def common(action, root=common_root):
        return SimpleNamespace(
            action=action,
            projection=proj.tree,
            projection_manifest=proj.manifest,
            projection_sha256=proj.sha,
            common_operation=operation_dir / "common-operation.json",
            management_python=Path("/pinned/python"),
            node_local_journal_root=root,
        )

    def checkpoint(action, op_file, sidecar, root):
        return SimpleNamespace(
            action=action,
            projection=proj.tree,
            projection_manifest=proj.manifest,
            projection_sha256=proj.sha,
            operation=op_file,
            snapshot=sidecar,
            publication_revision=False,
            superseded_operation_sha256=None,
            dataset_publication_operation=None,
            management_python=Path("/pinned/python"),
            node_local_upload_root=None,
            node_local_journal_root=root,
        )

    shared_dir = tmp_path / "home/rsna-knee/kaggle/dataset-publications"
    return SimpleNamespace(**locals())


def markers(x):
    return sorted(p.name for p in x.shared_dir.iterdir()) if x.shared_dir.exists() else []


def entries(x):
    with sqlite3.connect(x.ledger_file) as db:
        return [row[0] for row in db.execute("SELECT sha FROM entries")]


def test_full_common_create_reconcile_checkpoint_prepare_and_once_only_push(world):
    x = world
    op = x.operation
    assert op["checkpoint_entry"] is None and op["competition_submission_eligible"] is False
    assert op["source_receipt"]["dataset_slug"] == x.old_slug
    assert op["source_receipt"]["receipt_sha256"] == digest(x.receipt)
    assert op["regenerated_common_sha256"] == op["files_sha256"]
    manifest = json.loads((x.operation_dir / "common/common-manifest.json").read_text())
    assert manifest["competition_submission_eligible"] is False
    assert manifest["files"] == x.regenerated
    assert {n for n in manifest["files"] if n.startswith("code/rsna/")} == {
        "code/rsna/" + n for n in x.binding["code_sha256"]
    }
    assert "encoder/weights.safetensors" in manifest["files"]

    transport = x.c.run(x.common("prepare-common-transport"))
    footprint = transport["footprint"]
    assert footprint["shared_marker_files"] == 2 and footprint["deletion_credit"] is False
    assert footprint["node_local_upload_zip_bytes"] == transport["archive_bytes"]
    assert footprint["node_local_common_tree_bytes"] >= op["total_bytes"]
    assert markers(x) == []

    # Unknown outcome after the upload: never re-created, only reconciled.
    x.state.outcome = TimeoutError
    with pytest.raises(RuntimeError, match="uncertain"):
        x.c.run(x.common("create-common-dataset"))
    assert x.kaggle.creates == [COMMON_SLUG]
    key = "common-" + op["manifest_sha256"]
    assert markers(x) == [key + ".node-intent.json"]
    intent = json.loads((x.shared_dir / (key + ".node-intent.json")).read_text())
    assert intent["retry_prohibited"] is True and intent["checkpoint_entry"] is None
    assert "fixture-not-a-token" not in json.dumps(intent)
    assert "detail" not in (x.common_root / "uncertain.json").read_text()
    with pytest.raises(ValueError, match="reconcile only"):
        x.c.run(x.common("create-common-dataset"))
    assert x.kaggle.creates == [COMMON_SLUG]

    verified = x.c.run(x.common("reconcile-common-dataset"))
    assert verified["status"] == "all_payload_bytes_verified"
    assert verified["downloaded_all_bytes"] is True and verified["private"] is True
    assert x.kaggle.downloads == len(verified["files"]) == op["member_count"] + 1
    assert x.c.run(x.common("reconcile-common-dataset")) == verified
    assert markers(x) == [key + ".node-intent.json", key + ".node-verified.json"]
    assert sum((x.shared_dir / n).stat().st_size for n in markers(x)) < 64 * 1024
    assert entries(x) == []

    receipts = x.tmp_path / "common-receipts"
    receipts.mkdir()
    shutil.copyfile(
        x.operation_dir / "common/common-manifest.json", receipts / "common-manifest.json"
    )
    shutil.copyfile(
        x.shared_dir / (key + ".node-verified.json"), receipts / "common-verification.json"
    )
    slug = OWNER + "/rsna-ckpt-" + x.prefix
    candidate = x.prepare(
        x.wf, x.sidecar, x.tmp_path / "component", slug, OWNER + "/rsna-run-" + x.prefix, receipts
    )
    assert candidate["common_bundle"]["dataset_id"] == x.kaggle.datasets[COMMON_SLUG]["id"]
    payload = x.tmp_path / "component/dataset"
    assert {p.name for p in payload.rglob("*") if p.is_file()} == {
        "checkpoint.pt",
        "snapshot.json",
        "bundle-manifest.json",
        "dataset-metadata.json",
    }

    op_file = x.tmp_path / "component/operation.json"
    root = x.node_a / (
        "rsna-kaggle-journal-"
        + candidate["checkpoint_sha256"][:12]
        + "-"
        + candidate["manifest_sha256"][:12]
    )
    x.state.outcome = None
    x.c.run(x.checkpoint("prepare-transport", op_file, x.sidecar, root))
    x.c.run(x.checkpoint("create-dataset", op_file, x.sidecar, root))
    x.c.run(x.checkpoint("reconcile-dataset", op_file, x.sidecar, root))
    entry = x.c.run(x.checkpoint("push-kernel", op_file, x.sidecar, root))
    assert entry["state"] == "pushed" and entry["kernel_version"] == 1
    assert len(x.kaggle.saves) == 1
    assert x.kaggle.saves[0]["datasetDataSources"] == [slug, COMMON_SLUG, WHEELS]
    with pytest.raises(ValueError, match="Push already attempted"):
        x.c.run(x.checkpoint("push-kernel", op_file, x.sidecar, root))
    assert len(x.kaggle.saves) == 1 and x.kaggle.creates == [COMMON_SLUG, slug]
    assert entries(x) == [candidate["checkpoint_sha256"]]

    # Notebook composition from the bytes the fake remote actually holds.
    n = load("projection_notebook", x.proj.tree / "scripts/kaggle/notebook_entry.py")
    mounted = x.tmp_path / "kaggle-input"
    for name, dataset in (("ckpt", slug), ("common", COMMON_SLUG)):
        for member, data in x.kaggle.datasets[dataset]["files"].items():
            (mounted / name / member).parent.mkdir(parents=True, exist_ok=True)
            (mounted / name / member).write_bytes(data)
    bundle = mounted / "ckpt"
    bound = json.loads((bundle / "bundle-manifest.json").read_text())
    composed, merged = n.compose_common_bundle(bundle, bound, mounted, x.tmp_path / "composed")
    assert merged["files"] == {**x.regenerated, **bound["files"]}
    for member, row in merged["files"].items():
        assert hashed(composed / member) == row


def test_older_payload_is_not_reused_when_regenerated_code_differs(world):
    x = world

    def change(src):
        path = src / "scripts/kaggle/chunked_infer.py"
        path.write_text(path.read_text() + "\n# integrated package differs from old payload\n")

    newer = build_projection(x.tmp_path, "newer", change)
    workflow, _ = x.c.load_workflow(newer.tree, newer.manifest, newer.sha)
    workflow.validate_delta_base = lambda *a: None
    regenerated = x.regenerate(workflow, x.sidecar, newer.tree)
    assert x.wf.components.bootstrap_difference(x.regenerated, regenerated) == [
        "code/chunked_infer.py"
    ]
    target = x.tmp_path / "rejected-common"
    with pytest.raises(ValueError, match="code/chunked_infer.py"):
        workflow.components.prepare_common_operation(
            x.data,
            target,
            competition=workflow.COMPETITION,
            shared_wheelhouse=x.shared,
            checkpoint_format="frozen_base_delta",
            dataset_slug=COMMON_SLUG,
            source_receipt=x.receipt,
            regenerated=regenerated,
        )
    assert not target.exists()


@pytest.mark.parametrize("change", ["row", "slug", "partial", "public"])
def test_common_bootstrap_requires_the_exact_original_standalone_receipt(world, change):
    x = world
    receipt = json.loads(x.receipt.read_text())
    if change == "row":
        receipt["files"]["code/chunked_infer.py"]["bytes"] += 1
    elif change == "slug":
        receipt["slug"] = OWNER + "/another"
    elif change == "partial":
        receipt["downloaded_all_bytes"] = False
    else:
        receipt["private"] = False
    x.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt"):
        x.wf.components.prepare_common_operation(
            x.data,
            x.tmp_path / "rejected",
            competition=x.wf.COMPETITION,
            shared_wheelhouse=x.shared,
            checkpoint_format="frozen_base_delta",
            dataset_slug=COMMON_SLUG,
            source_receipt=x.receipt,
            regenerated=x.regenerated,
        )


def test_a_different_node_cannot_duplicate_the_common_creation(world, monkeypatch):
    x = world
    x.c.run(x.common("prepare-common-transport"))
    node_b = x.tmp_path / "node-b-tmp"
    node_b.mkdir()
    monkeypatch.setattr(x.c, "NODE_SCRATCH", node_b)
    root_b = node_b / x.common_root.name
    x.c.run(x.common("prepare-common-transport", root_b))
    x.c.run(x.common("create-common-dataset", root_b))
    assert x.kaggle.creates == [COMMON_SLUG]
    monkeypatch.setattr(x.c, "NODE_SCRATCH", x.node_a)
    with pytest.raises(ValueError, match="reconcile only"):
        x.c.run(x.common("create-common-dataset"))
    with pytest.raises(ValueError, match="durable shared intent"):
        x.c.run(x.common("prepare-common-transport"))
    assert x.kaggle.creates == [COMMON_SLUG]
    # Node A reconciles node B's create read-only from the shared marker.
    assert (
        x.c.run(x.common("reconcile-common-dataset"))["id"]
        == (x.kaggle.datasets[COMMON_SLUG]["id"])
    )


def test_lost_node_journal_reconciles_from_marker_and_never_recreates(world):
    x = world
    x.c.run(x.common("prepare-common-transport"))
    x.state.outcome = ConnectionError
    with pytest.raises(RuntimeError, match="uncertain"):
        x.c.run(x.common("create-common-dataset"))
    shutil.rmtree(x.common_root)
    with pytest.raises(ValueError, match="missing"):
        x.c.run(x.common("create-common-dataset"))
    with pytest.raises(ValueError, match="durable shared intent"):
        x.c.run(x.common("prepare-common-transport"))
    assert x.c.run(x.common("reconcile-common-dataset"))["downloaded_all_bytes"] is True
    assert x.kaggle.creates == [COMMON_SLUG]


def test_pending_common_intent_blocks_checkpoint_creates_and_absent_result_stays_blocked(world):
    x = world
    x.c.run(x.common("prepare-common-transport"))
    x.state.outcome = TimeoutError
    real_publish = x.kaggle.publish_zip
    x.kaggle.publish_zip = lambda slug, archive: None
    with pytest.raises(RuntimeError, match="uncertain"):
        x.c.run(x.common("create-common-dataset"))
    x.kaggle.publish_zip = real_publish
    assert x.c.pending_dataset_mutation(x.shared_dir.parent, None)
    # Absent remote after an intent is not authority to re-create.
    with pytest.raises(KeyError):
        x.c.run(x.common("reconcile-common-dataset"))
    with pytest.raises(ValueError, match="reconcile only"):
        x.c.run(x.common("create-common-dataset"))
    assert x.kaggle.creates == [COMMON_SLUG] and markers(x) == [
        "common-" + x.operation["manifest_sha256"] + ".node-intent.json"
    ]


@pytest.mark.parametrize("change", ["moved", "member", "account", "ledger", "projection"])
def test_common_phase_guards_fail_before_any_mutation(world, change, monkeypatch):
    x = world
    action = "create-common-dataset"
    if change != "moved":
        x.c.run(x.common("prepare-common-transport"))
    if change == "moved":
        tree = x.operation_dir / "common"
        tree.rename(x.operation_dir / "moved")
        shutil.copytree(x.operation_dir / "moved", tree)
        action = "prepare-common-transport"
    elif change == "member":
        with (x.operation_dir / "common/encoder/weights.safetensors").open("r+b") as handle:
            handle.write(b"X")
    elif change == "account":
        monkeypatch.setattr(x.kaggle, "username", "someone-else")
    elif change == "ledger":
        x.ledger_file.unlink()
    else:
        monkeypatch.setattr(x.c, "COMPONENT_PROJECTION", (*x.c.COMPONENT_PROJECTION, "absent.py"))
    with pytest.raises(ValueError):
        x.c.run(x.common(action))
    assert x.kaggle.creates == [] and not any(n.endswith("intent.json") for n in markers(x))


def test_reconcile_requires_a_durable_intent(world):
    x = world
    x.c.run(x.common("prepare-common-transport"))
    with pytest.raises(ValueError, match="No durable common create intent"):
        x.c.run(x.common("reconcile-common-dataset"))
    assert x.kaggle.downloads == 0


def test_cli_separates_common_and_checkpoint_arguments(world, monkeypatch, capsys):
    base = ["coordinator.py", "--projection", "p", "--projection-manifest", "m"]
    base += ["--projection-sha256", "s"]
    for argv in (
        ["create-common-dataset", "--common-operation", "o", "--snapshot", "s"],
        ["create-common-dataset", "--node-local-journal-root", "/tmp/x"],
        ["create-dataset", "--operation", "o"],
        ["create-dataset", "--operation", "o", "--snapshot", "s", "--common-operation", "c"],
    ):
        monkeypatch.setattr(sys, "argv", base[:1] + argv + base[1:])
        with pytest.raises(SystemExit) as error:
            world.c.main()
        assert error.value.code == 2
