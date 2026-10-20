"""Separate C100 final-evidence adapter; no files, networks, jobs or account calls.

Pass the exact original contract text and explicit accepted source/publisher/pilot
pins. This returns an isolated validator, not a deployment or entry authorization.
The original prediction, complete-state and T4 qualification functions are reused.
"""

import hashlib
import json
import re
import types

ORIGINAL_CONTRACT_SHA = "ee830d60e578163524c9534064e07badca6e05fbd488df0029c2b6d53c5b1075"
ORIGINAL_FULL_SOURCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PILOT_NUMERICAL_SOURCE = "39e315c65546aa5205518713bc27349ec63e7538"
PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
PILOT_JOB = "981620"
PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
INITIAL = "dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b"
PILOT_ORIGINAL_RECEIPT = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
OBJECTIVE_RECEIPT = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
PUBLICATION_STATUS = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
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
FINAL_MATCH_FIELDS = (
    "arm", "branch", "GPU_job", "source_git", "binding_sha256", "recipe",
    "base_science_git", "original_numerical_source_git",
    "density100_pilot_sha256", "density100_pilot_source_git", "density100_pilot_GPU_job",
    "original_portable_admission_sha256", "original_C0_binding_sha256",
    "parent_weight_sha256", "parent_source_sha256", "source_revision",
    "renderer_sha256", "accepted_renderer_sha256", "renderer_provenance_counts",
    "alias_regression_receipt_sha256", "input_admission_sha256",
    "runtime_versions", "parameter_count", "model_schema_sha256",
    "context_index_sha256", "source_context_index_sha256", "targets_sha256",
    "fitting_order_sha256", "license_sha256", "runtime_receipt_sha256",
    "engineering_decision_sha256", "funding_sha256", "initial_shared_tensor_sha256",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hexadecimal(value, length):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def completed(scheduler, job):
    require(
        isinstance(job, str) and job.isdecimal()
        and str(scheduler.get("job_id")) == job
        and scheduler.get("state") == "COMPLETED"
        and scheduler.get("exit_code") == "0:0"
        and scheduler.get("queue_empty") is True
        and scheduler.get("node") == "evc27",
        "Actual original evc27 COMPLETED 0:0 and empty queue required",
    )


def accepted_pilot(text, expected_sha, scheduler):
    require(isinstance(text, str) and hexadecimal(expected_sha, 64), "Explicit pilot bytes and pin required")
    require(digest(text) == expected_sha, "Actual pilot acceptance bytes differ")
    pilot = json.loads(text)
    require(
        pilot.get("status") == "actual_C100_full_stop_resume_engineering_complete"
        and pilot.get("branch") == "C100"
        and pilot.get("original_numerical_source_git") == PILOT_NUMERICAL_SOURCE
        and pilot.get("original_engineering_receipt_sha256") == PILOT_ORIGINAL_RECEIPT
        and pilot.get("objective_measurement_sha256") == OBJECTIVE_RECEIPT
        and pilot.get("full_stop_resume_complete_state_and_prediction_bytes_equal") is True
        and pilot.get("clinical_or_held_out_accuracy_claim") is False
        and pilot.get("automatic_full_training_promotion") is False
        and pilot.get("primary_B_C_C0_pipeline_unchanged") is True
        and pilot.get("source_git") == PILOT_SOURCE
        and pilot.get("GPU_job") == PILOT_JOB
        and pilot.get("binding_sha256") == PILOT_BINDING,
        "Actual separately accepted C100 engineering recovery required",
    )
    completed(scheduler, pilot.get("GPU_job"))
    terminal = pilot["terminal"]
    require(
        terminal.get("arm") == "C" and terminal.get("phase") == "full"
        and terminal.get("steps") == 24 and terminal.get("density_weight") == 100
        and terminal.get("source_studies") == 12
        and terminal.get("disjoint_engineering_prediction_studies") == 4
        and terminal.get("production_LR_horizon_updates") == 2916
        and terminal.get("clinical_accuracy_or_submission_candidate") is False
        and [r["step"] for r in pilot["real_optimizer_history"]] == list(range(1, 25)),
        "Pilot cannot be a partial run or production candidate",
    )
    return pilot


class Contract:
    """C100-only evidence gate with unchanged original serving checks."""

    def __init__(
        self, original_source, *, full_source_git=None, publisher_sha256=None,
        pilot_receipt_text=None, pilot_receipt_sha256=None, pilot_scheduler=None,
    ):
        require(digest(original_source) == ORIGINAL_CONTRACT_SHA, "Original contract source differs")
        require(
            hexadecimal(full_source_git, 40) and full_source_git != ORIGINAL_FULL_SOURCE
            and hexadecimal(publisher_sha256, 64),
            "Distinct pinned C100 full source and derived publisher are required",
        )
        self.pilot = accepted_pilot(pilot_receipt_text, pilot_receipt_sha256, pilot_scheduler or {})
        require(full_source_git != self.pilot["source_git"], "Pilot source cannot stand in for full fitting")
        # Only the declared recipe and distinct publication prefix change.
        changed = original_source
        for old, new in (
            ("density_weight=0 if arm == \"C0\" else 0.1,", "density_weight=100.0,"),
            ("checkpoints/predictive-full3887-", "checkpoints/predictive-C100-full3887-"),
        ):
            require(changed.count(old) == 1, "Original contract derivation point differs")
            changed = changed.replace(old, new)
        self._base = types.ModuleType("isolated_C100_final_contract")
        exec(compile(changed, "isolated_C100_final_contract", "exec"), self._base.__dict__)
        require(publisher_sha256 != self._base.PUBLISHER, "Original publisher does not admit C100")
        self._base.PUBLISHER = publisher_sha256
        self.full_source_git = full_source_git
        self.publisher_sha256 = publisher_sha256
        self.pilot_receipt_sha256 = pilot_receipt_sha256
        self.pilot_scheduler_sha256 = digest(canonical(pilot_scheduler))
        self.source_identity = {
            "original_contract_sha256": ORIGINAL_CONTRACT_SHA,
            "derived_recipe_and_prefix_source_sha256": digest(changed),
            "full_source_git": full_source_git,
            "publisher_sha256": publisher_sha256,
            "pilot_receipt_sha256": pilot_receipt_sha256,
        }

    def candidate(
        self, publication, provenance_text, own_source_text, license_bytes, *,
        epoch_publications, full_terminal, full_scheduler,
    ):
        p = json.loads(provenance_text)
        identity = {
            "arm": "C", "branch": "C100", "source_git": self.full_source_git,
            "base_science_git": ORIGINAL_FULL_SOURCE,
            "density100_pilot_sha256": self.pilot_receipt_sha256,
            "density100_pilot_source_git": self.pilot["source_git"], "density100_pilot_GPU_job": self.pilot["GPU_job"],
        }
        for name in ("original_portable_admission_sha256", "original_C0_binding_sha256"):
            require(hexadecimal(p.get(name), 64), "Exact original portable input provenance required")
            identity[name] = p[name]
        require(all(p.get(k) == v for k, v in identity.items()), "C100 branch/source/pilot binding differs")
        require(p.get("publisher_source_git") == self.full_source_git, "Derived publisher source differs")
        require(p["GPU_job"] != self.pilot["GPU_job"], "Pilot job cannot be a full fit")
        require(p.get("initial_shared_tensor_sha256") == INITIAL, "Original common initialization differs")
        require(
            all(p.get("code_sha256", {}).get(n) == h for n, h in MODULE_SHA256.items()),
            "C100 must keep the exact original inference and rendering modules",
        )
        chosen = self._base.candidate(publication, provenance_text, own_source_text, license_bytes)
        require(
            len(epoch_publications) == 6
            and [r.get("epoch") for r in epoch_publications] == list(range(1, 7))
            and epoch_publications[-1] == publication,
            "All six ordered original private publications, including this exact final, are required",
        )
        revisions = []
        for epoch, receipt in enumerate(epoch_publications, 1):
            require(
                receipt.get("status") == PUBLICATION_STATUS
                and receipt.get("objects") == 5
                and receipt.get("canonical_shared_publication_lock_used") is True
                and all(receipt.get(k) == v for k, v in identity.items())
                and receipt.get("GPU_job") == p["GPU_job"]
                and receipt.get("prefix") == (
                    f"checkpoints/predictive-C100-full3887-{self.full_source_git[:12]}/C/epoch-{epoch:03d}"
                )
                and hexadecimal(receipt.get("revision"), 40),
                "A private epoch is unverified or belongs to another run",
            )
            files = receipt["files"]
            require(set(files) == set(publication["files"]), "Exact five-object epoch required")
            require(
                all(type(v.get("bytes")) is int and v["bytes"] > 0 and hexadecimal(v.get("sha256"), 64)
                    for v in files.values())
                and 280_000_000 < files["model.pt"]["bytes"] < 340_000_000
                and files["source-permission.txt"] == publication["files"]["source-permission.txt"]
                and files["own-source.json"] == publication["files"]["own-source.json"],
                "Private epoch sizes/source/license proofs differ",
            )
            revisions.append(receipt["revision"])
        require(len(set(revisions)) == 6, "Six original publication revisions required")
        completed(full_scheduler, p["GPU_job"])
        require(
            full_terminal.get("status") == "actual_fixed_final_full3887_predictive_arm_complete"
            and full_terminal.get("epochs") == 6 and full_terminal.get("updates") == 2916
            and full_terminal.get("study_exposures") == 3887 * 6
            and full_terminal.get("retained_full_checkpoints") == 7
            and full_terminal.get("immutable_epoch_sidecars") == 6
            and full_terminal.get("all_original_fitting_labels_masks_unchanged") is True
            and full_terminal.get("protected_or_gold_cases_read") is False
            and hexadecimal(full_terminal.get("final_checkpoint_sha256"), 64)
            and type(full_terminal.get("final_checkpoint_bytes")) is int
            and full_terminal["final_checkpoint_bytes"] > 0
            and all(k in p and full_terminal.get(k) == p[k] for k in FINAL_MATCH_FIELDS),
            "Actual complete C100 final and preserved epoch-six provenance must agree",
        )
        # final.pt and epoch-006.pt intentionally have different serializations.
        return {
            **chosen, "branch": "C100", "model_filename": "predictive-C100-epoch-006.pt",
            "C100_contract": dict(self.source_identity),
            "pilot_scheduler_sha256": self.pilot_scheduler_sha256,
            "full_terminal_sha256": digest(canonical(full_terminal)),
            "full_scheduler_sha256": digest(canonical(full_scheduler)),
            "six_publications_sha256": digest(canonical(epoch_publications)),
        }

    def validate_export(self, export, expected, initialized):
        require(expected.get("branch") == "C100", "C100 candidate required")
        return self._base.validate_export(export, expected, initialized)

    def validate_qualification(self, receipt, chosen):
        require(chosen.get("branch") == "C100", "C100 candidate required")
        return self._base.validate_qualification(receipt, chosen)

    def check_predictions(self, uids, rows):
        return self._base.check_predictions(uids, rows)
