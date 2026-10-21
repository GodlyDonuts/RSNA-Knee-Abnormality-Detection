"""Actual independent Colab final provenance; never project it onto a Slurm run."""

import hashlib
import json
import math
import re
import types
from datetime import datetime

SCIENCE = "94a3921aefdf333624caf41aff77aa10b4a0c57e"
BASE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
MANIFEST = "bce57e766fe4be00fb5ea41d9be7fd498f4296fbc033f663f68f50e0a2b3570b"
ORIGINAL_BINDING = "f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd"
BRANCH = "Colab-C100-seed20261021"
ORIGINAL_CONTRACT = "ee830d60e578163524c9534064e07badca6e05fbd488df0029c2b6d53c5b1075"
RESUME_CONTROL = "27b87a02e899fda775bb9985841b0354c614805bbdc6ea08ea92855cf6bfea0f"
RESUME_CHECKPOINT = "43c1c75920bd31356f5fad96441e43bd1317eac504a3083613a33f89b6e49fdc"
RESUME2_CONTROL = "9576fb26cea626754bc7cc4d94be0c25e0b545d41bf3ef428066a6b7d6f0eb28"
RESUME2_CHECKPOINT = "0fab361de8da414968391b8f7a596190d68cb1e2966e72e96429bcc8b5dae516"
CODE = {
    "actual_colab_train.py": "15e17223c6eda06bf417df0fe8b64e2fd9729e8a5571ae012274974eb29267a3",
    "colab_storage.py": "0eb47a1edb2c73a113e57e43ccae0253311f08df9a11d1a1bd5a83bd2448c4a6",
    "kaggle_io.py": "83da4bc50a4e21e79741fa7a603bf7a04de28093914ca177af3ae3a371ded7f5",
    "components.py": "24e2571bfe8830907bb30a23cdf1bdcde90c4e78b93cb330ae8b8557c81c8d80",
    "publication.py": "e426dfe2dc39a82b36d7c26fd3c300f9dcd7efe310052158d214b701ce7b1336",
    "workflow.py": "2d3074fbf677285687e80c0e6b58721407bc61ddea39a990797da0e1d4e3b900",
}


MODULE_SHA256 = {
    "rsna/competitive/predictive/__init__.py": "51104641623e3b4bd4eb7263d011dd7a048542d706f6d1b96741e876b8bfd9b9",
    "rsna/competitive/predictive/inputs.py": "e773ce9f2e2e2f775cf991365867b6ae4340d55b3097541c37080a938a62c459",
    "rsna/competitive/predictive/cases.py": "5f7c81fd270c072fa16ed486735049bb72070a704304b6ee1adc2c4f6d4f3d09",
    "rsna/competitive/predictive/supports.py": "74cc5b71a9372b0a7970ac3507484fe96d6d1d4e4e710fc8177121f72a95597b",
    "rsna/competitive/predictive/density.py": "b5f56767dd277af65864983f3a5cd58d89d4cc8153990e9139e6f81b870e64a0",
    "rsna/competitive/predictive/model.py": "38c9fc7c2e7e79ea90972f1b14c6e47491aba2f1641496277534e7cd20856f08",
    "rsna/competitive/predictive/infer.py": "1f547a54778cd2e57caaaf0513362e21a85b8119f9f1b188985a17ee9fd162f8",
    "rsna/competitive/predictive/dicom.py": "0980f1958bd9f5c929bbd85fb67cd9e148b6506608af76f0a5f5e52d518fdb94",
    "rsna/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "rsna/competitive/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(v):
    return hashlib.sha256(v.encode() if isinstance(v, str) else v).hexdigest()


def proof(v):
    require(
        set(v) == {"bytes", "sha256"}
        and type(v["bytes"]) is int
        and v["bytes"] > 0
        and re.fullmatch("[0-9a-f]{64}", v["sha256"]),
        "Complete byte/hash proof required",
    )


def matches(v, contents):
    require(
        v == {"bytes": len(contents), "sha256": digest(contents)},
        "Original downloaded object differs",
    )


def publication(v, key, names):
    slug = "godlydonuts/rsna-colab-c100-" + key + "-" + MANIFEST[:12]
    require(
        v["status"] == "private_epoch_full_download_size_SHA_verified"
        and v["private"] is True
        and type(v["id"]) is int
        and v["id"] > 0
        and v["version"] == 1
        and v["slug"] == slug
        and v["key"] == key
        and v["revision"] == f"Kaggle:{v['id']}:v1"
        and v["new_GPU_or_competition_calls"] == 0,
        "Exact private original Colab publication required",
    )
    require(set(v["files"]) == names, "All original publication objects required")
    for p in v["files"].values():
        proof(p)
    require(
        sum(p["bytes"] for p in v["files"].values()) < 2 * 2**30,
        "Original private Dataset cap exceeded",
    )


def execution_continuation(terminal, binding, first_publication):
    """Accept only the executed epoch001 recovery, without inventing weight provenance."""
    if "execution_continuation" not in terminal:
        return None
    original = "/content/rsna-independent-C100-20261021-v1/run-C100-seed20261021"
    actual = terminal["execution_continuation"]
    if actual.get("status") == "exact_epoch002_resume_with_network_independent_local_preservation":
        prior = actual["previous_continuation"]
        require(
            prior.get("status") == "exact_epoch001_resume_after_publication_observation_failure",
            "One exact prior recovery",
        )
        execution_continuation({"execution_continuation": prior}, binding, first_publication)
        expected2 = dict(
            status="exact_epoch002_resume_with_network_independent_local_preservation",
            source_checkpoint_sha256=RESUME2_CHECKPOINT,
            resume_update=972,
            discarded_logged_unsaved_updates=[973, 974, 975],
            possible_unlogged_update_replayed=976,
            original_run=original,
            prior_run=original + "-recovery-v1",
            resumed_run=original + "-recovery-v2",
            control_path="/content/rsna-colab-epoch002-recovery-20261021-v1/resume.py",
            control_sha256=RESUME2_CONTROL,
            previous_recovery_sha256=RESUME_CONTROL,
            previous_continuation=prior,
            original_control_sha256=CODE["actual_colab_train.py"],
            original_ten_hour_deadline=prior["original_ten_hour_deadline"],
            new_account_mutations=0,
            remote_publication_deferred=True,
            numerical_functions_or_recipe_changed=False,
        )
        require(actual == expected2, "Exact executed epoch002 recovery required")
        return actual
    expected = dict(
        status="exact_epoch001_resume_after_publication_observation_failure",
        source_checkpoint_sha256=RESUME_CHECKPOINT,
        resume_update=486,
        discarded_unsaved_updates=[487, 488, 489],
        original_updates_preserved=True,
        original_run=original,
        resumed_run=original + "-recovery-v1",
        control_path="/content/rsna-colab-epoch001-recovery-20261021-v1/resume.py",
        control_sha256=RESUME_CONTROL,
        original_control_sha256=CODE["actual_colab_train.py"],
        original_ten_hour_deadline=(
            datetime.fromisoformat(binding["started_UTC"]).timestamp() + 36000
        ),
        initial_epoch_Dataset_id=12756621,
        first_epoch_reuploaded=False,
        numerical_functions_or_recipe_changed=False,
    )
    require(terminal["execution_continuation"] == expected, "Exact executed recovery required")
    require(
        first_publication["id"] == 12756621
        and first_publication["files"]["full-state.private.pt.bin"]["sha256"] == RESUME_CHECKPOINT,
        "Recovery must use the original privately verified epoch001 checkpoint",
    )
    return expected


def terminal_storage_state(terminal):
    return terminal.get("all_epochs_private_verified") is True or (
        terminal.get("all_epochs_private_verified") is False
        and terminal.get("all_epochs_local_immutable") is True
        and terminal.get("execution_continuation", {}).get("control_sha256") == RESUME2_CONTROL
    )


def epoch_continuation(continuation, epoch):
    if continuation is None or epoch == 1:
        return None
    if continuation.get("control_sha256") == RESUME2_CONTROL and epoch == 2:
        return continuation["previous_continuation"]
    return continuation


def recovery_checkpoint_binding(continuation, publications):
    if continuation and continuation.get("control_sha256") == RESUME2_CONTROL:
        require(
            publications[1]["files"]["full-state.private.pt.bin"]["sha256"] == RESUME2_CHECKPOINT,
            "Recovered epoch002 must have actual independent private byte proof",
        )


class Contract:
    def __init__(self, original, *, module_sha256):
        require(
            digest(original) == ORIGINAL_CONTRACT, "Pinned unchanged inference contract required"
        )
        self._base = types.ModuleType("Colab_original_inference_contract")
        exec(compile(original, "original_predictive_contract", "exec"), self._base.__dict__)
        require(
            module_sha256 == MODULE_SHA256 and set(module_sha256) == set(self._base.MODULES),
            "Every exact numerical serving module required",
        )
        self.module_sha256 = dict(module_sha256)

    def candidate(
        self,
        *,
        binding_text,
        terminal_text,
        epoch_publications,
        terminal_publication,
        own_source_text,
        license_bytes,
    ):
        b, t, own = json.loads(binding_text), json.loads(terminal_text), json.loads(own_source_text)
        base = self._base
        recipe = {**base.recipe("C"), "density_weight": 100.0, "seed": 20261021}
        expected = dict(
            branch=BRANCH,
            source_git=SCIENCE,
            base_science_git=BASE,
            original_binding_sha256=ORIGINAL_BINDING,
            input_manifest_sha256=MANIFEST,
            input_private_revision="Kaggle:12754716:v1+original_HF_revisions",
            fitting_order_sha256=base.ROSTER,
            recipe=recipe,
            parent_weight_sha256=base.PARENT,
            parent_source_sha256=base.PARENT_SOURCE,
            targets_sha256=base.TARGETS,
            license_sha256=base.LICENSE,
            torch="2.11.0+cu130",
            timm="1.0.29",
            cuda="13.0",
            GPU="NVIDIA A100-SXM4-40GB",
            storage_control_git="HEAD",
            storage_control_sha256=CODE,
            new_seed_and_runtime_not_strict_causal_comparison=True,
            protected_labels_or_external_source_data_used=False,
            clinical_accuracy_claim=False,
        )
        require(
            all(b.get(k) == v for k, v in expected.items()),
            "Exact actual Colab source/recipe/input binding required",
        )
        require(
            set(b)
            == set(expected)
            | {"initial_complete_state_sha256", "numerical_code_sha256", "started_UTC"},
            "Do not substitute or invent Newton scheduler/publication provenance",
        )
        require(
            re.fullmatch("[0-9a-f]{64}", b["initial_complete_state_sha256"]),
            "Actual independent initial learned state required",
        )
        require(digest(license_bytes) == base.LICENSE, "Original inherited MRI permission required")
        require(
            t["status"] == "actual_full_C100_seed20261021_complete"
            and t["binding"] == b
            and t["updates"] == 2916
            and t["epochs"] == 6
            and terminal_storage_state(t)
            and t["clinical_or_Kaggle_score"] is None
            and math.isfinite(t["elapsed_seconds"])
            and t["elapsed_seconds"] > 0,
            "Actual full six-epoch Colab terminal required",
        )
        require(len(epoch_publications) == 6, "Six ordered original epoch publications required")
        continuation = execution_continuation(t, b, epoch_publications[0])
        recovery_checkpoint_binding(continuation, epoch_publications)
        names = {
            n + ".bin"
            for n in (
                "model.pt",
                "full-state.private.pt",
                "run-start.json",
                "source-permission.txt",
                "colab_train.py",
                "colab_storage.py",
                "kaggle_io.py",
            )
        }
        identities = []
        for epoch, receipt in enumerate(epoch_publications, 1):
            executed = epoch_continuation(continuation, epoch)
            recovered = executed is not None
            recovery_names = (
                {"resume-control.py.bin", "continuation.json.bin"} if recovered else set()
            )
            publication(receipt, f"epoch-{epoch:03d}", names | recovery_names)
            files = receipt["files"]
            if recovered:
                require(
                    files["resume-control.py.bin"]["sha256"] == executed["control_sha256"],
                    "Published recovery source must equal the actually executed source",
                )
                matches(files["continuation.json.bin"], canonical(executed) + b"\n")
            matches(files["run-start.json.bin"], binding_text.encode())
            matches(files["source-permission.txt.bin"], license_bytes)
            for name in ("colab_train.py", "colab_storage.py", "kaggle_io.py"):
                key = "actual_colab_train.py" if name == "colab_train.py" else name
                require(
                    files[name + ".bin"]["sha256"] == CODE[key], "Actual executed source differs"
                )
            require(
                280_000_000 < files["model.pt.bin"]["bytes"] < 340_000_000
                and 800_000_000 < files["full-state.private.pt.bin"]["bytes"] < 1_200_000_000,
                "Actual complete model and optimizer checkpoint required",
            )
            identities.append(receipt["id"])
        publication(terminal_publication, "terminal", {"terminal.json.bin"})
        matches(terminal_publication["files"]["terminal.json.bin"], terminal_text.encode())
        require(
            len(set(identities + [terminal_publication["id"]])) == 7,
            "Never reuse another epoch or terminal Dataset",
        )
        for n, h in self.module_sha256.items():
            require(
                digest(own[n]) == h and b["numerical_code_sha256"]["code/" + n] == h,
                "Serving modules must equal the actual Colab training modules",
            )
        last = epoch_publications[-1]
        return {
            "arm": "C",
            "branch": BRANCH,
            "architecture": {**base.architecture("C"), "branch": BRANCH},
            "provenance": {**b, "epoch": 6, "completed_updates": 2916},
            "model_schema_sha256": base.SCHEMAS["C"],
            "parameter_count": base.PARAMETERS["C"],
            "model_filename": "model.pt.bin",
            "model": last["files"]["model.pt.bin"],
            "model_dataset_ref": last["slug"],
            "model_dataset_id": last["id"],
            "publication_revision": last["revision"],
            "terminal_sha256": digest(terminal_text),
            "all_publications_sha256": digest(
                canonical(epoch_publications + [terminal_publication])
            ),
            "module_sha256": dict(self.module_sha256),
            # Model exports retain their original scientific binding. Execution
            # recovery is disclosed separately and included in candidate identity.
            **({"execution_continuation": continuation} if continuation is not None else {}),
        }

    def validate_export(self, export, expected, initialized):
        import torch

        require(
            expected["branch"] == BRANCH
            and set(export) == {"model", "architecture", "provenance"}
            and export["architecture"] == expected["architecture"]
            and export["provenance"] == expected["provenance"],
            "Actual Colab export binding differs",
        )
        state, template = export["model"], initialized.state_dict()
        require(set(state) == set(template), "Complete learned tensors and buffers required")
        for n, t in template.items():
            v = state[n]
            require(
                isinstance(v, torch.Tensor)
                and v.device.type == "cpu"
                and v.shape == t.shape
                and v.dtype == t.dtype
                and bool(torch.isfinite(v).all()),
                "Learned tensor schema or finite values differ",
            )
        schema = {n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in state.items()}
        require(
            digest(canonical(schema)) == expected["model_schema_sha256"]
            and sum(p.numel() for p in initialized.parameters()) == expected["parameter_count"],
            "Complete predictive model schema/count differs",
        )
        initialized.load_state_dict(state, strict=True)
        return initialized.eval()

    def validate_qualification(self, receipt, chosen):
        require(chosen["branch"] == BRANCH, "Independent Colab candidate required")
        return self._base.validate_qualification(receipt, chosen)

    def check_predictions(self, uids, rows):
        return self._base.check_predictions(uids, rows)
