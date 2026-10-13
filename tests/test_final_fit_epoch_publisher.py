"""Once-only private preservation of the all-development RSNA-only final fit."""

import contextlib
import copy
import hashlib
import io
import json
import os
from types import SimpleNamespace

import pytest
import torch

from scripts.newton import final_fit_epoch_publisher as pub

T4_LINK = pub.T4_CPU_LINK_SHA


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return pub.sha(path)


def raw_config():
    return {
        "fit_scope": "all_development",
        "fitting_supervision": {
            "kind": "label_source_final_v1",
            "sha256": pub.OVERLAY_SHA,
            "enabled": True,
        },
        "epochs": 10,
        "warmup_updates": 426,
        "seed": 20260922,
        "submission_every_epochs": 1,
        "diagnostic_study_limit": None,
        "fold": 0,
        "checkpoint_format": "frozen_base_delta",
        "precision": "fp16",
        "encoder": dict(pub.ENCODER, weights_path="/w", config_path="/c"),
    }


def binding_for(code):
    return {
        "config": dict(raw_config(), encoder=dict(pub.ENCODER, adapter={"rank": 32})),
        "release": {"manifest_sha256": pub.RELEASE_SHA, "binding_sha256": "b" * 64},
        "folds_sha256": "f" * 64,
        "training_uids_sha256": pub.FITTING_UID_SHA,
        "outer_uids_sha256": pub.EMPTY_ROSTER_SHA,
        "world_size": 1,
        "device_type": "cuda",
        "torch_version": pub.TORCH_VERSION,
        "code_sha256": code,
        "experimental_fitting_supervision": {
            "overlay_sha256": pub.OVERLAY_SHA,
            "release_sha256": pub.RELEASE_SHA,
            "folds_sha256": pub.FOLDS_FILE_SHA,
            "fitting_uid_sha256": pub.FITTING_UID_SHA,
            "outer_uid_sha256": pub.EMPTY_ROSTER_SHA,
            "fitting_studies": 3403,
            "fit_scope": "all_development",
            "final_fit_only": True,
            "internal_validation_available": False,
            "evaluation_masks_changed": False,
            "enabled": True,
            "source": {"csv_sha256": pub.SOURCE_CSV_SHA},
        },
    }


def cpu_receipt():
    return {
        "status": pub.CPU_STATUS,
        "CPU_job": "905196",
        "source_git": pub.SOURCE_GIT,
        "source_modules": {"code/rsna/model.py": h("model\n"), "worker.py": "c" * 64},
        "training_studies": 3403,
        "outer_studies": 0,
        "masked_out_source_studies": 0,
        "fitting_uid_sha256": pub.FITTING_UID_SHA,
        "release_sha256": pub.RELEASE_SHA,
        "folds_sha256": pub.FOLDS_FILE_SHA,
        "overlay_sha256": pub.OVERLAY_SHA,
        "config_sha256": pub.CONFIG_SHA,
        "source_csv_sha256": pub.SOURCE_CSV_SHA,
        "optimizer_updates_per_epoch": 426,
        "fixed_final_optimizer_updates": 4260,
        "observed_shared_quota": "Disk quotas for usr someone (uid 1)",
    }


def t4_receipt():
    return {
        "status": "COMPLETE",
        "actual_remote": {"status": "COMPLETE"},
        "kernel_slug": pub.T4_KERNEL[1],
        "version": 1,
        "exact_source_private_offline_T4_verified": True,
        "actual_recovery": {
            "status": "actual_single_T4_final_public_source_recovery_accepted",
            "science_git": pub.SOURCE_GIT,
            "CPU_receipt_sha256": T4_LINK,
            "exact_complete_state_scaler_RNG_recovery": True,
            "amp_scaler": {"_growth_tracker": 4},
            "complete_state_sha256": "e" * 64,
            "finite_updates": [{"update": i} for i in range(4)],
        },
    }


def state(binding, epoch, **edit):
    value = {
        "schema_version": 2,
        "binding": binding,
        "epoch": epoch,
        "next_update": 0,
        "step": 426 * epoch,
        "optimizer_boundary": True,
        "scheduled_epoch": True,
        "amp_scaler": {"scale": 128.0},
        "rng_by_rank": [{"torch": torch.tensor([epoch])}],
        "adapter": {"w": torch.full((4,), float(epoch))},
    }
    value.update(edit)
    return value


def add_epoch(root, binding, epoch, **edit):
    epochs = root / "production" / "epoch-checkpoints"
    epochs.mkdir(parents=True, exist_ok=True)
    checkpoint = epochs / f"epoch-{epoch:03d}.pt"
    torch.save(state(binding, epoch, **edit), checkpoint)
    write(epochs / f"epoch-{epoch:03d}.json", pub.expected_snapshot(binding, epoch, checkpoint))
    return checkpoint


def make(tmp_path, monkeypatch, *, binding_edit=None, run_edit=None, cpu_edit=None, t4_edit=None):
    root = tmp_path / "control"
    code = root / "code" / "rsna" / "model.py"
    code.parent.mkdir(parents=True)
    code.write_text("model\n")
    license_file = tmp_path / "LICENSE.md"
    license_file.write_text("official DINOv3 license fixture")
    monkeypatch.setattr(pub, "LICENSE", license_file)
    monkeypatch.setattr(pub, "LICENSE_SHA", pub.sha(license_file))
    monkeypatch.setattr(pub, "OVERLAY_SHA", write(root / "overlay.json", {"schema_version": 1}))
    monkeypatch.setattr(pub, "CONFIG_SHA", write(root / "final-config.json", raw_config()))
    # Constants feeding the fixture must be patched before the receipts are built.
    cpu = cpu_receipt()
    (cpu_edit or (lambda v: None))(cpu)
    monkeypatch.setattr(pub, "CPU_RECEIPT_SHA", write(root / "CPU-accepted.json", cpu))
    t4 = t4_receipt()
    (t4_edit or (lambda v: None))(t4)
    monkeypatch.setattr(pub, "T4_RECEIPT_SHA", write(root / "T4-recovery-accepted.json", t4))
    (root / "publish.py").write_bytes(open(pub.__file__, "rb").read())
    binding = binding_for({"model.py": h("model\n")})
    (binding_edit or (lambda v: None))(binding)
    record = {
        "status": "running",
        "binding": binding,
        "diagnostic_run": False,
        "training_studies": 3403,
        "outer_studies": 0,
        "selection_policy": "fixed_epoch_count; no_internal_evaluation",
        "held_out_confirmation_and_gold_used": False,
    }
    (run_edit or (lambda v: None))(record)
    write(root / "production" / "run.json", record)
    write(
        root / "production-admission.json",
        {
            "status": pub.ADMISSION_STATUS,
            "node": "evc40",
            "source_git": pub.SOURCE_GIT,
            "private_prefix": pub.PREFIX,
            "shared_future_bytes": 0,
            "CPU_receipt_sha256": pub.CPU_RECEIPT_SHA,
            "T4_recovery_receipt_sha256": pub.T4_RECEIPT_SHA,
            "final_config_sha256": pub.CONFIG_SHA,
            "overlay_sha256": pub.OVERLAY_SHA,
            "initial_run_binding_sha256": pub.digest(binding),
            "publisher_sha256": pub.sha(root / "publish.py"),
        },
    )
    add_epoch(root, binding, 1)
    return root, binding


class FakeHub:
    def __init__(self, *, private=True, existing=(), corrupt=None, fail_commit=False):
        self.files = {p: b"x" for p in existing}
        self.private, self.corrupt, self.fail_commit = private, corrupt, fail_commit
        self.commits, self.downloads, self.tokens = [], [], []

    def __call__(self, token):
        self.tokens.append(token)
        return self

    def repo_info(self, repo, repo_type, revision=None):
        assert repo == pub.REPO and repo_type == "dataset"
        return SimpleNamespace(private=self.private, sha=f"rev{len(self.commits)}")

    def get_paths_info(self, repo, paths, repo_type, revision):
        return [SimpleNamespace(path=p) for p in paths if p in self.files]

    def create_commit(self, repo, repo_type, parent_commit, commit_message, operations, **_):
        if self.fail_commit:
            raise ConnectionError("network")
        for path, source in operations:
            self.files[path] = open(source, "rb").read()
        self.commits.append(parent_commit)
        return SimpleNamespace(oid=f"rev{len(self.commits)}")

    def hf_hub_download(self, repo, filename, repo_type, revision, local_dir, force_download):
        assert force_download is True
        self.downloads.append(filename)
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.files[filename]
        if self.corrupt and filename.endswith(self.corrupt):
            data = data + b"!"
        open(path, "wb").write(data)
        return path


def placed(monkeypatch, node="evc40", cpus="1"):
    monkeypatch.setattr(pub.os, "uname", lambda: SimpleNamespace(nodename=node))
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", cpus)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)


def run_publish(root, hub, epoch=1, token="hf_secretTOKEN1234567890"):
    return pub.publish(
        epoch,
        root,
        stdin=io.StringIO(json.dumps({"token": token})),
        api_factory=hub,
        operation=lambda path_in_repo, path_or_fileobj: (path_in_repo, path_or_fileobj),
        lock=contextlib.nullcontext(),
    )


def test_full_publication_verifies_all_five_objects_by_forced_redownload(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    placed(monkeypatch)
    hub = FakeHub()
    receipt = run_publish(root, hub)
    prefix = f"{pub.PREFIX}/epoch-001"
    assert receipt["file_count"] == 5 and receipt["revision"] == "rev1"
    assert set(hub.files) == {f"{prefix}/{n}" for n in pub.MEMBERS}
    assert sorted(hub.downloads) == sorted(f"{prefix}/{n}" for n in pub.MEMBERS)
    journal = root / "epoch-publication" / "epoch-001"
    intent = json.loads((journal / "intent.json").read_text())
    assert intent["automatic_retry"] is False and intent["prefix"] == prefix
    provenance = (journal / "source-and-input-provenance.json").read_text()
    assert "observed_shared_quota" not in provenance and "someone" not in provenance
    for path in journal.iterdir():
        assert "hf_secret" not in path.read_text()
    with pytest.raises(ValueError, match="read-only reconciliation"):
        run_publish(root, FakeHub())


def test_pending_epoch_or_admission_is_refused(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="regular"):
        pub.source_files(2, root)
    with pytest.raises(ValueError, match="completed final-fit epoch"):
        pub.source_files(11, root)
    admission = json.loads((root / "production-admission.json").read_text())
    admission["status"] = "prepared_pending_root_admission"
    (root / "production-admission.json").write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="production admission"):
        pub.source_files(1, root)


@pytest.mark.parametrize(
    "edit",
    [
        lambda b: b["config"].update(fit_scope="outer_fold"),
        lambda b: b["config"]["fitting_supervision"].update(kind="label_source_v1"),
        lambda b: b["config"].update(warmup_updates=281),
        lambda b: b["experimental_fitting_supervision"].update(final_fit_only=False),
        lambda b: b.update(outer_uids_sha256="0" * 64),
        lambda b: b["config"]["encoder"].update(weights_sha256="0" * 64),
    ],
)
def test_wrong_scope_recipe_or_base_is_refused(tmp_path, monkeypatch, edit):
    root, _ = make(tmp_path, monkeypatch, binding_edit=edit)
    with pytest.raises(ValueError):
        pub.source_files(1, root)


def test_wrong_scientific_source_is_refused(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    (root / "code" / "rsna" / "model.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        pub.source_files(1, root)
    root2, _ = make(
        tmp_path / "b", monkeypatch, binding_edit=lambda b: b["code_sha256"].update(x="0" * 64)
    )
    with pytest.raises(ValueError, match="source/runtime"):
        pub.source_files(1, root2)
    root3, _ = make(tmp_path / "c", monkeypatch, cpu_edit=lambda c: c.update(source_git="0" * 40))
    with pytest.raises(ValueError, match="CPU905196"):
        pub.source_files(1, root3)


def test_old_or_reused_private_prefix_is_refused(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    monkeypatch.setattr(pub, "PREFIX", pub.OLD_PREFIXES[-2])
    with pytest.raises(ValueError):
        pub.source_files(1, root)


@pytest.mark.parametrize("change", ["bytes", "sidecar", "best", "partial", "steps", "link"])
def test_partial_or_changed_snapshot_is_refused(tmp_path, monkeypatch, change):
    root, binding = make(tmp_path, monkeypatch)
    epochs = root / "production" / "epoch-checkpoints"
    meta = json.loads((epochs / "epoch-001.json").read_text())
    if change == "bytes":
        with (epochs / "epoch-001.pt").open("ab") as f:
            f.write(b"0")
    elif change == "sidecar":
        meta["diagnostic_only"] = True
        (epochs / "epoch-001.json").write_text(json.dumps(meta))
    elif change == "best":
        meta["best_epoch_selected"] = True
        (epochs / "epoch-001.json").write_text(json.dumps(meta))
    elif change == "partial":
        add_epoch(root, binding, 1, next_update=17)
    elif change == "steps":
        add_epoch(root, binding, 1, step=425)
    else:
        target = tmp_path / "elsewhere.pt"
        (epochs / "epoch-001.pt").rename(target)
        (epochs / "epoch-001.pt").symlink_to(target)
    with pytest.raises(ValueError):
        pub.source_files(1, root)


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t["actual_recovery"].update(CPU_receipt_sha256="0" * 64),
        lambda t: t["actual_recovery"].update(science_git="0" * 40),
        lambda t: t["actual_recovery"]["finite_updates"].pop(),
        lambda t: t.update(kernel_slug="godlydonuts/rsna-final-publicsource-recovery-20261013"),
        lambda t: t["actual_recovery"].update(exact_complete_state_scaler_RNG_recovery=False),
    ],
)
def test_recovery_link_must_be_exact_private_T4_receipt(tmp_path, monkeypatch, edit):
    root, _ = make(tmp_path, monkeypatch, t4_edit=edit)
    with pytest.raises(ValueError, match="T4"):
        pub.source_files(1, root)


@pytest.mark.parametrize("kind", ["predictions", "metrics", "diagnostic", "external", "result"])
def test_folded_or_diagnostic_contamination_is_refused(tmp_path, monkeypatch, kind):
    edits = {
        "diagnostic": {"run_edit": lambda r: r.update(diagnostic_run=True)},
        "external": {"binding_edit": lambda b: b.update(external_initialization={"x": 1})},
    }
    root, _ = make(tmp_path, monkeypatch, **edits.get(kind, {}))
    run = root / "production"
    if kind == "predictions":
        (run / "outer-predictions.csv").write_text("x")
    elif kind == "metrics":
        (run / "metrics.json").write_text("{}")
    elif kind == "result":
        write(
            run / "result.json",
            {
                "fit_scope": "all_development",
                "diagnostic_run": False,
                "outer_studies": 0,
                "evaluation_source": None,
                "metrics": {"macro_auc": 0.9},
                "predictions_sha256": None,
                "checkpoint_selection": "fixed_final_epoch_no_internal_selection",
            },
        )
    with pytest.raises(ValueError):
        pub.source_files(1, root)


def test_checkpoint_and_local_obligation_caps(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    size = (root / "production" / "epoch-checkpoints" / "epoch-001.pt").stat().st_size
    monkeypatch.setattr(pub, "CHECKPOINT_CAP", size)
    with pytest.raises(ValueError, match="checkpoint cap"):
        pub.source_files(1, root)
    monkeypatch.setattr(pub, "CHECKPOINT_CAP", 128 * 2**20)
    assert pub.local_obligations(root / "production", 1) < pub.LOCAL_CAP
    monkeypatch.setattr(pub, "LOCAL_CAP", 11 * 128 * 2**20)
    with pytest.raises(ValueError, match="original bound"):
        pub.source_files(1, root)


def test_existing_remote_path_is_refused_without_journal(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    placed(monkeypatch)
    hub = FakeHub(existing=[f"{pub.PREFIX}/epoch-001/manifest.json"])
    with pytest.raises(ValueError, match="never repeat"):
        run_publish(root, hub)
    assert not (root / "epoch-publication").exists() and not hub.commits


def test_uncertain_commit_is_journaled_and_never_retried(tmp_path, monkeypatch):
    root, _ = make(tmp_path, monkeypatch)
    placed(monkeypatch)
    with pytest.raises(RuntimeError, match="uncertain"):
        run_publish(root, FakeHub(fail_commit=True))
    journal = root / "epoch-publication" / "epoch-001"
    assert json.loads((journal / "uncertain.json").read_text()) == {
        "error_type": "ConnectionError",
        "automatic_retry": False,
    }
    hub = FakeHub()
    with pytest.raises(ValueError, match="reconciliation"):
        run_publish(root, hub)
    before = sorted(p.name for p in journal.iterdir())
    report = pub.reconcile(
        1, root, stdin=io.StringIO(json.dumps({"token": "hf_x"})), api_factory=hub
    )
    assert report["status"] == "uncertain_requires_owner_review" and not hub.commits
    assert sorted(p.name for p in journal.iterdir()) == before


@pytest.mark.parametrize("hub", [FakeHub(corrupt="checkpoint.pt"), FakeHub(private=False)])
def test_full_verification_failure_is_not_verified(tmp_path, monkeypatch, hub):
    root, _ = make(tmp_path, monkeypatch)
    placed(monkeypatch)
    if hub.private is False:
        hub.repo_info = lambda *a, revision=None, **k: SimpleNamespace(
            private=revision is None, sha="rev0"
        )
    with pytest.raises(ValueError):
        run_publish(root, hub)
    assert not (root / "epoch-publication" / "epoch-001" / "verified.json").exists()


@pytest.mark.parametrize("payload", [{"token": "hf_a", "extra": 1}, {"token": "not-hf"}, ["hf_a"]])
def test_credential_must_be_single_stdin_token(tmp_path, monkeypatch, payload):
    root, _ = make(tmp_path, monkeypatch)
    placed(monkeypatch)
    with pytest.raises(ValueError, match="credential"):
        pub.publish(1, root, stdin=io.StringIO(json.dumps(payload)), api_factory=FakeHub())


@pytest.mark.parametrize("node,cpus", [("evc31", "1"), ("evc40", "2")])
def test_wrong_placement_is_refused(tmp_path, monkeypatch, node, cpus):
    root, _ = make(tmp_path, monkeypatch)
    placed(monkeypatch, node, cpus)
    with pytest.raises(ValueError, match="one-CPU"):
        run_publish(root, FakeHub())


@pytest.mark.parametrize(
    "value",
    [
        {"study_uid": "x"},
        {"rows": [{"labels": {}}]},
        {"note": "1.2.826.0.1.3680043.8.498.1"},
    ],
)
def test_published_receipts_reject_raw_case_ids_or_clinical_fields(value):
    with pytest.raises(ValueError):
        pub.assert_sanitized(copy.deepcopy(value))
    pub.assert_sanitized({"version": pub.TORCH_VERSION, "sha256": "a" * 64, "studies_n": 3})


def test_regular_rejects_linked_parent_with_matching_file_bytes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "control.json").write_text("same bytes")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="non-linked"):
        pub.regular(root / "linked/control.json", root)
