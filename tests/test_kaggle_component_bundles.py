"""CPU tests for optional checkpoint-only payloads over an immutable common bundle.

Synthetic bytes only: no weights, network, credentials or account mutation.
These tests are not Kaggle runtime or accuracy evidence.
"""

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_kaggle_shared_wheelhouse import proof

from scripts.kaggle import coordinator as c
from scripts.kaggle import workflow

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "initiative/audits/kaggle-publicsource-e10-ebeb3c394286-v1"
k = workflow.components
LICENSE = "https://github.com/facebookresearch/dinov3/blob/" + "d" * 40 + "/LICENSE.md"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/kaggle" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = load("component_notebook", "notebook_entry.py")
pub = load("component_publication", "publication.py")


def hashed(path):
    return {"sha256": workflow.digest(path), "bytes": path.stat().st_size}


class Remote:
    username = "owner"

    def __init__(self, listing=None, version=1, private=True, identity=77):
        self.listing, self.version, self.private, self.identity = (
            listing,
            version,
            private,
            identity,
        )

    def call(self, service, method, body):
        if method == "GetDatasetStatus":
            return {"status": "READY"}
        if method == "ListDatasetFiles":
            return {"datasetFiles": [{"name": a, "totalBytes": b} for a, b in self.listing.items()]}
        if method == "ListDatasets":
            return {}
        raise AssertionError(method)

    def dataset(self, slug):
        return {
            "id": self.identity,
            "isPrivate": self.private,
            "currentVersionNumber": self.version,
        }


def build(tmp_path, monkeypatch, *, delta=True):
    """Standalone bundle, its extracted common bundle and a checkpoint-only bundle."""
    repo = tmp_path / "repo"
    (repo / "rsna").mkdir(parents=True)
    (repo / "rsna/__init__.py").write_text("SYNTHETIC = 1\n")
    (repo / "pyproject.toml").write_text("synthetic project fixture\n")
    sidecar = tmp_path / "epoch.json"
    sidecar.write_text("{}")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint fixture, never submitted")
    config = tmp_path / "config.json"
    config.write_text("{}")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"synthetic frozen base; not pretrained" * 64)
    spec = tmp_path / "encoder.json"
    spec.write_text(
        json.dumps(
            {
                "architecture": "dinov3_l16",
                "config_path": str(config),
                "config_sha256": workflow.digest(config),
                "weights_path": str(base) if delta else "/not-copied/pretrained.bin",
                "weights_sha256": workflow.digest(base) if delta else "c" * 64,
            }
        )
    )
    meta = {
        "checkpoint_sha256": workflow.digest(checkpoint),
        "epoch": 10,
        "release_binding_sha256": "b" * 64,
        "binding_sha256": "c" * 64,
    }
    provenance = {a: b for a, b in json.loads(spec.read_text()).items() if not a.endswith("_path")}
    binding = {
        "config": {
            "encoder": provenance,
            "checkpoint_format": "frozen_base_delta" if delta else "full",
        }
    }
    monkeypatch.setattr(workflow, "validate_snapshot", lambda *a, **kw: (meta, binding, checkpoint))
    monkeypatch.setattr(workflow, "validate_delta_base", lambda *a: None)
    monkeypatch.setattr(workflow, "FLOOR", 0)
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "fixture.whl").write_bytes(b"fixture")
    (wheels / "requirements.lock").write_text("fixture==1 --hash=sha256:" + "a" * 64)
    (wheels / "acceptance.json").write_text(
        json.dumps(
            {
                "status": "cpu_offline_install_passed",
                "pyproject_sha256": workflow.digest(repo / "pyproject.toml"),
            }
        )
    )
    files = {p.name: hashed(p) for p in wheels.iterdir()}
    (wheels / "wheelhouse.json").write_text(
        json.dumps(
            {
                "platform": "linux_x86_64",
                "python": "3.12",
                "cuda": "13.0",
                "offline_install_verified": True,
                "files": files,
                "pyproject_sha256": workflow.digest(repo / "pyproject.toml"),
            }
        )
    )
    shared = proof()
    shared.update(
        manifest_sha256=workflow.digest(wheels / "wheelhouse.json"),
        pyproject_sha256=workflow.digest(repo / "pyproject.toml"),
    )
    shared_file = tmp_path / "shared-proof.json"
    shared_file.write_text(json.dumps(shared))
    license_file = tmp_path / "LICENSE.md"
    license_file.write_text("Synthetic licensing fixture for local assembler tests only.")
    prefix = meta["checkpoint_sha256"][:12]

    def prepare(output, dataset, kernel, common=None, licence=license_file):
        return workflow.prepare_bundle(
            sidecar,
            spec,
            wheels,
            output,
            dataset,
            kernel,
            licence,
            LICENSE,
            workflow.digest(licence),
            repo,
            shared_wheelhouse=shared_file,
            common_bundle=common,
        )

    standalone = tmp_path / "standalone"
    prepare(standalone, "owner/rsna-" + prefix, "owner/rsna-" + prefix)
    common = tmp_path / "common"
    common_slug = "owner/rsna-common-encoder-v1"
    k.extract_common(
        standalone / "dataset",
        common,
        competition=workflow.COMPETITION,
        shared_wheelhouse=shared,
        checkpoint_format="frozen_base_delta" if delta else "full",
        dataset_slug=common_slug,
    )
    manifest = json.loads((common / k.COMMON_MANIFEST).read_text())
    expected = pub.expected_files(common, manifest, "common-manifest.json")
    remote = Remote({name: row["bytes"] for name, row in expected.items()})
    verified = pub.reconcile(
        remote,
        common_slug,
        expected,
        stream=lambda r, s, name, row: hashed(common / name),
        manifest_name="common-manifest.json",
    )
    (common / "common-verification.json").write_text(json.dumps(verified))
    receipts = tmp_path / "common-receipts"
    receipts.mkdir()
    for name in (k.COMMON_MANIFEST, "common-verification.json"):
        shutil.copyfile(common / name, receipts / name)
    (common / "common-verification.json").unlink()
    component = tmp_path / "component"
    operation = prepare(
        component, "owner/rsna-ckpt-" + prefix, "owner/rsna-run-" + prefix, receipts
    )
    return SimpleNamespace(
        standalone=standalone,
        common=common,
        receipts=receipts,
        component=component,
        operation=operation,
        prepare=prepare,
        prefix=prefix,
        repo=repo,
        tmp=tmp_path,
        shared=shared,
    )


@pytest.fixture
def bundles(tmp_path, monkeypatch):
    return build(tmp_path, monkeypatch)


def test_checkpoint_only_payload_is_byte_equivalent_to_the_standalone_bundle(bundles):
    standalone = json.loads((bundles.standalone / "dataset/bundle-manifest.json").read_text())
    manifest = json.loads((bundles.component / "dataset/bundle-manifest.json").read_text())
    uploaded = {
        str(p.relative_to(bundles.component / "dataset"))
        for p in (bundles.component / "dataset").rglob("*")
        if p.is_file()
    }
    assert uploaded == {
        "checkpoint.pt",
        "snapshot.json",
        "bundle-manifest.json",
        "dataset-metadata.json",
    }
    assert "encoder/weights.safetensors" not in manifest["files"]
    common = k.check_binding(
        manifest["common_bundle"],
        bundles.receipts / k.COMMON_MANIFEST,
        bundles.receipts / "common-verification.json",
    )
    assert k.merge(manifest["files"], common, manifest["common_bundle"]) == standalone["files"]
    assert workflow.load_operation(bundles.component / "operation.json") == bundles.operation
    meta = json.loads((bundles.component / "kernel/kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == [
        bundles.operation["dataset_slug"],
        "owner/rsna-common-encoder-v1",
        "owner/wheels",
    ]
    assert meta["id"] != bundles.operation["dataset_slug"]
    request = workflow.kernel_request(bundles.component / "operation.json", bundles.operation)
    assert request["datasetDataSources"] == meta["dataset_sources"]
    assert request["newTitle"] != bundles.operation["dataset_slug"].split("/")[1].replace("-", " ")


def test_old_standalone_format_is_unchanged(bundles):
    manifest = json.loads((bundles.standalone / "dataset/bundle-manifest.json").read_text())
    operation = json.loads((bundles.standalone / "operation.json").read_text())
    assert "common_bundle" not in manifest and "common_bundle" not in operation
    assert workflow.load_operation(bundles.standalone / "operation.json") == operation
    assert workflow.dataset_sources(operation) == [operation["dataset_slug"], "owner/wheels"]
    bundle = bundles.standalone / "dataset"
    assert n.compose_common_bundle(bundle, manifest, bundles.tmp, bundles.tmp / "x") == (
        bundle,
        manifest,
    )
    assert not (bundles.tmp / "x").exists()


@pytest.mark.parametrize(
    "dataset,kernel",
    [
        ("owner/rsna-ckpt-{p}", "owner/rsna-ckpt-{p}"),
        ("owner/rsna-common-encoder-v1-{p}", "owner/rsna-common-encoder-v1"),
        ("other/rsna-ckpt-{p}", "other/rsna-run-{p}"),
        ("owner/rsna-ckpt-{p}", "owner/rsna-ckpt_{p}"),
    ],
)
def test_component_identities_must_be_distinct_from_the_start(bundles, dataset, kernel):
    dataset, kernel = (x.replace("{p}", bundles.prefix) for x in (dataset, kernel))
    with pytest.raises(ValueError):
        bundles.prepare(bundles.tmp / "bad", dataset, kernel, bundles.receipts)
    assert not (bundles.tmp / "bad").exists()


@pytest.mark.parametrize("change", ["code", "licence"])
def test_changed_source_or_licence_cannot_reuse_common_bytes(bundles, change):
    licence = bundles.tmp / "LICENSE.md"
    if change == "code":
        (bundles.repo / "rsna/__init__.py").write_text("SYNTHETIC = 2\n")
    else:
        licence = bundles.tmp / "OTHER-LICENSE.md"
        licence.write_text("A different licence text.")
    with pytest.raises(ValueError, match="differs"):
        bundles.prepare(
            bundles.tmp / "bad",
            "owner/rsna-ckpt-" + bundles.prefix,
            "owner/rsna-run-" + bundles.prefix,
            bundles.receipts,
            licence,
        )


def test_full_format_candidate_cannot_mix_with_delta_base(tmp_path, monkeypatch):
    delta = build(tmp_path / "delta", monkeypatch)
    full = build(tmp_path / "full", monkeypatch, delta=False)
    with pytest.raises(ValueError):
        full.prepare(
            tmp_path / "bad",
            "owner/rsna-ckpt-" + full.prefix,
            "owner/rsna-run-" + full.prefix,
            delta.receipts,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("private", False),
        ("version", 2),
        ("downloaded_all_bytes", False),
        ("status", "partial"),
        ("slug", "owner/another-common"),
        ("competition_submission", True),
        ("id", 0),
    ],
)
def test_replaced_or_incomplete_common_proof_fails_closed(bundles, field, value):
    path = bundles.component / "common-verification.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        workflow.load_operation(bundles.component / "operation.json")


@pytest.mark.parametrize("member", ["encoder/weights.safetensors", "code/rsna/__init__.py"])
def test_missing_or_replaced_common_member_in_receipt_fails_closed(bundles, member):
    path = bundles.component / "common-verification.json"
    receipt = json.loads(path.read_text())
    receipt["files"].pop(member)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        workflow.load_operation(bundles.component / "operation.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_version", 2),
        ("dataset_id", 78),
        ("manifest_sha256", "e" * 64),
        ("files_sha256", "e" * 64),
        ("verification_sha256", "e" * 64),
    ],
)
def test_cross_version_or_identity_mixing_is_rejected(bundles, field, value):
    record = copy.deepcopy(bundles.operation["common_bundle"])
    record[field] = value
    with pytest.raises(ValueError):
        k.check_binding(
            record,
            bundles.receipts / k.COMMON_MANIFEST,
            bundles.receipts / "common-verification.json",
        )


@pytest.mark.parametrize(
    "candidate",
    [
        {"code/rsna/__init__.py": {"sha256": "a" * 64, "bytes": 1}},
        {"encoder.json": {"sha256": "a" * 64, "bytes": 1}},
        {"Checkpoint.pt": {"sha256": "a" * 64, "bytes": 1}},
        {"../checkpoint.pt": {"sha256": "a" * 64, "bytes": 1}},
        {"common-manifest.json": {"sha256": "a" * 64, "bytes": 1}},
    ],
)
def test_conflicting_or_unexpected_member_paths_are_rejected(bundles, candidate):
    manifest = json.loads((bundles.component / "dataset/bundle-manifest.json").read_text())
    common = json.loads((bundles.receipts / k.COMMON_MANIFEST).read_text())
    with pytest.raises(ValueError):
        k.merge({**manifest["files"], **candidate}, common, manifest["common_bundle"])
    bundle = bundles.component / "dataset"
    with pytest.raises(ValueError):
        n.compose_common_bundle(
            bundle,
            {**manifest, "files": {**manifest["files"], **candidate}},
            bundles.tmp,
            bundles.tmp / "composed",
        )


@pytest.mark.parametrize(
    "change",
    ["license_sha256", "encoder_spec_sha256", "checkpoint_format", "diagnostic", "checkpoint"],
)
def test_common_manifest_admission_guards(bundles, change):
    common = json.loads((bundles.receipts / k.COMMON_MANIFEST).read_text())
    if change == "checkpoint_format":
        common[change] = "full"
    elif change == "diagnostic":
        common["diagnostic_only"] = True
    elif change == "checkpoint":
        common["files"]["checkpoint.pt"] = {"sha256": "a" * 64, "bytes": 1}
    else:
        common[change] = "e" * 64
    with pytest.raises(ValueError):
        k.validate_common_manifest(common)


def mount(bundles, name="common"):
    inputs = bundles.tmp / "input"
    inputs.mkdir(exist_ok=True)
    shutil.copytree(bundles.component / "dataset", inputs / "rsna-ckpt", dirs_exist_ok=True)
    shutil.copytree(
        bundles.common, inputs / name, ignore=shutil.ignore_patterns("dataset-metadata.json")
    )
    bundle = inputs / "rsna-ckpt"
    return inputs, bundle, json.loads((bundle / "bundle-manifest.json").read_text())


def test_notebook_composes_exact_standalone_tree_and_encoder_arguments(bundles):
    inputs, bundle, manifest = mount(bundles)
    composed, merged = n.compose_common_bundle(bundle, manifest, inputs, bundles.tmp / "composed")
    standalone = json.loads((bundles.standalone / "dataset/bundle-manifest.json").read_text())
    assert merged["files"] == standalone["files"]
    for name, row in merged["files"].items():
        assert hashed(composed / name) == row
        assert (composed / name).read_bytes() == (
            bundles.standalone / "dataset" / name
        ).read_bytes()
    working = bundles.tmp / "working"
    working.mkdir()
    arguments = n.encoder_arguments(composed, working, merged)
    assert arguments[0] == "--encoder-spec"
    spec = json.loads(Path(arguments[1]).read_text())
    assert Path(spec["weights_path"]).is_relative_to(composed)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "replaced", "binding"])
def test_notebook_rejects_missing_duplicate_extra_replaced_or_rebound_common(bundles, change):
    inputs, bundle, manifest = mount(bundles)
    if change == "missing":
        shutil.rmtree(inputs / "common")
    elif change == "duplicate":
        shutil.copytree(bundles.common, inputs / "common-copy")
    elif change == "extra":
        (inputs / "common/encoder/extra.json").write_text("{}")
    elif change == "replaced":
        with (inputs / "common/encoder/weights.safetensors").open("r+b") as handle:
            handle.write(b"X")
    else:
        manifest["common_bundle"]["dataset_version"] = 2
    with pytest.raises(ValueError):
        n.compose_common_bundle(bundle, manifest, inputs, bundles.tmp / "composed")


@pytest.mark.parametrize(
    "remote",
    [
        Remote(version=2),
        Remote(private=False),
        Remote(identity=78),
    ],
)
def test_remote_common_identity_drift_fails_closed(bundles, remote):
    with pytest.raises(ValueError):
        workflow.verify_common_bundle(remote, bundles.operation["common_bundle"])


def test_remote_common_identity_and_account(bundles):
    record = bundles.operation["common_bundle"]
    assert workflow.verify_common_bundle(Remote(), record)["id"] == 77
    other = Remote()
    other.username = "other"
    with pytest.raises(ValueError, match="another account"):
        workflow.verify_common_bundle(other, record)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_kernel_attachment_multiset_includes_common_exactly_once(bundles, order):
    sources = workflow.dataset_sources(bundles.operation)
    rotated = sources[order:] + sources[:order]
    source = "exact code"
    op = dict(bundles.operation, kernel_source_sha256=hashlib.sha256(source.encode()).hexdigest())
    op.pop("runtime_image", None)

    class Kernel:
        def __init__(self, attached):
            self.attached = attached

        def kernel(self, action, slug, version):
            if action == "GetKernelSessionStatus":
                return {"status": "RUNNING"}
            return {
                "metadata": {
                    "isPrivate": True,
                    "enableInternet": False,
                    "machineShape": "NvidiaTeslaT4",
                    "competitionDataSources": [workflow.COMPETITION],
                    "datasetDataSources": self.attached,
                },
                "blob": {"source": source},
            }

    assert workflow.check_kernel(Kernel(rotated), {"operation": op}, 1)["status"] == "RUNNING"
    for wrong in (sources[:1] + sources[2:], sources + [sources[1]]):
        with pytest.raises(ValueError, match="exact prepared"):
            workflow.check_kernel(Kernel(wrong), {"operation": op}, 1)


def test_common_publication_uses_its_own_manifest_name(bundles):
    manifest = json.loads((bundles.common / k.COMMON_MANIFEST).read_text())
    files = pub.expected_files(bundles.common, manifest, "common-manifest.json")
    assert "common-manifest.json" in files and "bundle-manifest.json" not in files
    actual = {name: row["bytes"] for name, row in files.items()}
    assert set(pub.map_members(actual, files, "common-manifest.json")) == set(files)
    with pytest.raises(ValueError):
        pub.map_members(actual, files)
    with pytest.raises((ValueError, FileNotFoundError)):
        pub.expected_files(bundles.common, manifest)


def test_byte_savings_from_existing_public_source_receipt():
    receipt = json.loads((RECEIPT / "Dataset-byte-verification.json").read_text())
    table = {a: b for a, b in receipt["files"].items() if a != "bundle-manifest.json"}
    result = k.savings(table, receipt["files"]["bundle-manifest.json"]["bytes"], candidates=5)
    assert result["standalone_payload_bytes_each"] == 1338824684
    assert result["common_payload_bytes_once"] == 1213254920
    assert result["checkpoint_only_payload_bytes_each"] == 125569764
    assert result["saved_bytes"] == 4 * 1213254920
    assert result["per_additional_candidate_fraction_saved"] > 0.9


# Node-local publication journals for new component operations.

CHECKPOINT = "a" * 64
MANIFEST = "b" * 64
KEY = CHECKPOINT + "-" + MANIFEST[:12]


@pytest.fixture
def coordinated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(c, "NODE_SCRATCH", scratch)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10**15))
    data = tmp_path / "operation/dataset"
    data.mkdir(parents=True)
    (data / "checkpoint.pt").write_bytes(b"synthetic checkpoint")
    (data / "bundle-manifest.json").write_text("{}")
    (data / "dataset-metadata.json").write_text("{}")
    files = c.p.expected_files(data, {"files": {"checkpoint.pt": hashed(data / "checkpoint.pt")}})
    operation = tmp_path / "operation/operation.json"
    operation.write_text("{}")
    op = {
        "checkpoint_sha256": CHECKPOINT,
        "manifest_sha256": MANIFEST,
        "dataset_slug": "godlydonuts/rsna-ckpt-aaaaaaaaaaaa",
        "shared_wheelhouse": {"dataset_slug": "godlydonuts/wheels"},
        "common_bundle": {"dataset_slug": "godlydonuts/rsna-common"},
    }
    state = SimpleNamespace(creates=0, outcome=TimeoutError, ledger=0)

    def ledger(*args):
        state.ledger += 1
        raise RuntimeError("canonical shared ledger reached")

    flow = SimpleNamespace(
        verify_shared_wheelhouse=lambda *a: None,
        verify_common_bundle=lambda *a: None,
        ledger=ledger,
    )
    monkeypatch.setattr(
        c,
        "load_workflow",
        lambda *a: (flow, {"package_git_commit": "x"} | {"training_git_commit": "y"}),
    )
    monkeypatch.setattr(
        c, "validate_operation", lambda *a: (op, {"checkpoint_sha256": CHECKPOINT}, files)
    )
    remote = SimpleNamespace(
        username="godlydonuts",
        call=lambda *a: {},
        dataset=lambda slug: {"id": 9, "isPrivate": True, "currentVersionNumber": 1},
    )
    monkeypatch.setattr(c, "remote_from_stdin", lambda w: (remote, "test-only-not-a-token"))

    def management(python):
        def create(*args):
            state.creates += 1
            raise state.outcome("sensitive detail must not enter journals")

        return create

    monkeypatch.setattr(c, "management_create", management)

    def reconcile(r, slug, expected):
        return {
            "schema_version": 1,
            "status": "all_payload_bytes_verified",
            "id": 9,
            "version": 1,
            "slug": slug,
            "private": True,
            "files": expected,
            "downloaded_all_bytes": True,
            "competition_submission": False,
        }

    monkeypatch.setattr(c.p, "reconcile", reconcile)
    root = scratch / ("rsna-kaggle-journal-" + KEY[:12] + KEY[64:])

    def args(action, journal_root=root):
        return SimpleNamespace(
            action=action,
            projection=tmp_path,
            projection_manifest=tmp_path,
            projection_sha256="p" * 64,
            operation=operation,
            snapshot=tmp_path,
            publication_revision=False,
            superseded_operation_sha256=None,
            dataset_publication_operation=None,
            management_python=Path("/pinned/python"),
            node_local_upload_root=None,
            node_local_journal_root=journal_root,
        )

    shared = home / "rsna-knee/kaggle/dataset-publications"
    return SimpleNamespace(args=args, root=root, shared=shared, state=state, op=op, files=files)


def shared_names(x):
    return sorted(p.name for p in x.shared.iterdir())


def test_node_local_journal_unknown_outcome_reconciles_without_retry(coordinated):
    x = coordinated
    record = c.run(x.args("prepare-transport"))
    assert record["node_local_journal"]["root"] == str(x.root)
    assert shared_names(x) == []
    assert (x.root / "transport.json").exists() and (x.root / "upload/bundle.zip").exists()
    with pytest.raises(RuntimeError, match="uncertain"):
        c.run(x.args("create-dataset"))
    assert x.state.creates == 1
    assert shared_names(x) == [KEY + ".node-intent.json"]
    marker = json.loads((x.shared / (KEY + ".node-intent.json")).read_text())
    assert marker["retry_prohibited"] is True and marker["files"] == x.files
    assert "sensitive" not in (x.root / "uncertain.json").read_text()
    assert "test-only" not in json.dumps(marker)
    with pytest.raises(FileExistsError):
        c.run(x.args("create-dataset"))
    assert x.state.creates == 1
    with pytest.raises(ValueError, match="fall back"):
        c.run(x.args("create-dataset", None))
    assert c.pending_dataset_mutation(x.shared.parent, x.shared / "other")
    # The owning node is lost: reconciliation proceeds read-only from shared markers.
    shutil.rmtree(x.root)
    with pytest.raises(ValueError, match="missing"):
        c.run(x.args("create-dataset"))
    proof = c.run(x.args("reconcile-dataset"))
    assert proof["status"] == "all_payload_bytes_verified"
    assert shared_names(x) == [KEY + ".node-intent.json", KEY + ".node-verified.json"]
    assert c.run(x.args("reconcile-dataset")) == proof
    assert not c.pending_dataset_mutation(x.shared.parent, x.shared / "other")
    with pytest.raises(RuntimeError, match="canonical shared ledger"):
        c.run(x.args("push-kernel"))
    assert x.state.ledger == 1 and x.state.creates == 1
    assert not (x.shared.parent / "coordinator-claim").exists()


def test_node_local_journal_identity_and_scope_guards(coordinated):
    x = coordinated
    c.run(x.args("prepare-transport"))
    transport = (x.root / "transport.json").read_bytes()
    x.root.rename(x.root.with_name("retained-original-journal"))
    x.root.mkdir(mode=0o700)
    (x.root / "transport.json").write_bytes(transport)
    with pytest.raises(ValueError, match="moved"):
        c.run(x.args("create-dataset"))
    assert x.state.creates == 0
    with pytest.raises(ValueError, match="exact private"):
        c.run(x.args("create-dataset", x.root.with_name("rsna-kaggle-journal-other")))
    common = x.op.pop("common_bundle")
    with pytest.raises(ValueError, match="new component"):
        c.run(x.args("create-dataset"))
    x.op["common_bundle"] = common
    (x.shared / CHECKPOINT).mkdir()
    with pytest.raises(ValueError, match="new component"):
        c.run(x.args("create-dataset"))


def test_shared_journal_default_is_unchanged_and_blocked_by_pending_node_intent(coordinated):
    x = coordinated
    x.shared.mkdir(parents=True)
    (x.shared / ("c" * 64 + "-" + "d" * 12 + ".node-intent.json")).write_text("{}")
    x.op.pop("common_bundle")
    c.run(x.args("prepare-transport", None))
    assert (x.shared / CHECKPOINT / "transport.json").exists()
    with pytest.raises(ValueError, match="awaits read-only reconciliation"):
        c.run(x.args("create-dataset", None))
    assert x.state.creates == 0
