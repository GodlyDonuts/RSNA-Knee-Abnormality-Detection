"""Exact source derivation and runtime admission for the separate C100 branch."""

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

BASE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
INITIAL = "dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b"
WORKER_SHA = "0282984d70cd47f7b6259e382720bacb99d73edfdd175d612a62b5f5a3e8f81a"
PUBLISHER_SHA = "cbfc161bf17381c816832329f892da5584ac4586138234e6901965f7b09cd795"
ORIGINAL = "initiative/research/predictive-full-fitting-20261020-v2/"
ROOT = "/tmp/rsna-predictive-density100-full-20261020-v1"
RUN = "/tmp/rsna-predictive-full-C100-20261020-v1"
PILOT_PATH = "/tmp/rsna-predictive-density100-pilot-20261020-v1/GPU-accepted.json"
NUMERICAL_SHA = {
    "alias_provenance.py": "68b046f909d01323172140675e846b317ef7e30cab17e6d6618a858f23fb6025",
    "rsna/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "rsna/competitive/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "rsna/competitive/predictive/__init__.py":
        "51104641623e3b4bd4eb7263d011dd7a048542d706f6d1b96741e876b8bfd9b9",
    "rsna/competitive/predictive/inputs.py":
        "e773ce9f2e2e2f775cf991365867b6ae4340d55b3097541c37080a938a62c459",
    "rsna/competitive/predictive/cases.py":
        "5f7c81fd270c072fa16ed486735049bb72070a704304b6ee1adc2c4f6d4f3d09",
    "rsna/competitive/predictive/supports.py":
        "74cc5b71a9372b0a7970ac3507484fe96d6d1d4e4e710fc8177121f72a95597b",
    "rsna/competitive/predictive/density.py":
        "b5f56767dd277af65864983f3a5cd58d89d4cc8153990e9139e6f81b870e64a0",
    "rsna/competitive/predictive/cache.py":
        "f82c478e0d4632ed31b491e4915066a47adf8275a7c2ba6fb42b9ca99871f944",
    "rsna/competitive/predictive/context_cache.py":
        "bf9d59b26bd8a28f4533648d5b10d6c77bd5caf8d2f69de0b77968ce4d4ea3b5",
    "rsna/competitive/predictive/model.py":
        "38c9fc7c2e7e79ea90972f1b14c6e47491aba2f1641496277534e7cd20856f08",
    "rsna/competitive/predictive/train.py":
        "066dc5eda8ca8a793548e6fde62c2c6a06480f052d9034aa3fe2e08a5004dc58",
    "rsna/competitive/predictive/infer.py":
        "1f547a54778cd2e57caaaf0513362e21a85b8119f9f1b188985a17ee9fd162f8",
    "rsna/competitive/predictive/dicom.py":
        "0980f1958bd9f5c929bbd85fb67cd9e148b6506608af76f0a5f5e52d518fdb94",
    "rsna/competitive/predictive/source_resume.py":
        "51a42fd5d5b178d36db84c6d94344ad584545459fef83a7d1de3bf1f24e9ef45",
}
PROVENANCE_KEYS = (
    "branch", "base_science_git", "density100_pilot_sha256", "density100_pilot_source_git",
    "density100_pilot_GPU_job", "original_portable_admission_sha256", "original_C0_binding_sha256",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def recipe():
    return dict(arm="C", epochs=6, effective_batch=8, encoder_lr=3e-5, head_lr=3e-5,
                weight_decay=0.02, warmup_fraction=0.05, end_lr_fraction=0.1,
                density_weight=100.0, head_only_updates=3, clip=1.0,
                seed=20261020, schedule_updates=2916)


def read_json(ref):
    path = Path(ref["path"])
    before = path.stat()
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and before.st_uid == os.getuid() and not before.st_mode & 0o077,
            "Owner-only original C100 reference required")
    require(before.st_size <= 4 * 2**20, "Small admission reference required")
    raw = path.read_bytes()
    after = path.stat()
    require(sha(raw) == ref["sha256"]
            and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "C100 reference changed")
    return json.loads(raw)


def validate(binding, pilot, decision, funding, admitted):
    """Validate real referenced evidence; returning fields does not authorize dispatch."""
    require(binding["arm"] == "C" and binding["branch"] == "C100"
            and binding["base_science_git"] == BASE and binding["node"] == "evc27"
            and binding["run_root"] == RUN and binding["code_root"] == ROOT + "/code"
            and binding["expected_initial_shared_tensor_sha256"] == INITIAL
            and re.fullmatch("[0-9a-f]{40}", binding["source_git"])
            and binding["source_git"] != BASE, "Distinct exact C100 binding required")
    require(binding["density100_pilot"]["path"] == PILOT_PATH
            and binding["density100_pilot_source_git"] == PILOT_SOURCE,
            "Exact original C100 pilot required")
    require(all(binding["code_sha256"].get(n) == h for n, h in NUMERICAL_SHA.items()),
            "Original full numerical modules changed")
    require(pilot["status"] == "actual_C100_full_stop_resume_engineering_complete"
            and pilot["GPU_job"] == "981620" and pilot["source_git"] == PILOT_SOURCE
            and pilot["binding_sha256"] == PILOT_BINDING and pilot["branch"] == "C100"
            and pilot["full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
            and pilot["automatic_full_training_promotion"] is False
            and pilot["clinical_or_held_out_accuracy_claim"] is False
            and pilot["primary_B_C_C0_pipeline_unchanged"] is True,
            "Actual completed C100 engineering acceptance required")
    require(pilot["original_numerical_source_git"] ==
            "39e315c65546aa5205518713bc27349ec63e7538"
            and pilot["objective_measurement_sha256"] ==
            "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
            and pilot["original_engineering_receipt_sha256"] ==
            "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10",
            "Original pilot/measurement lineage differs")
    history = pilot["real_optimizer_history"]
    require(len(history) == 24 and [r["step"] for r in history] == list(range(1, 25))
            and [r["studies"] for r in history] == [8, 4] * 12
            and [r["encoder_updated"] for r in history] == [False] * 3 + [True] * 21
            and pilot["terminal"]["density_weight"] == 100
            and pilot["terminal"]["steps"] == 24, "Complete exact pilot history required")
    require(all(math.isfinite(r[k]) for r in history for k in
                ("loss", "exam_loss", "density_loss", "encoder_gradient_norm",
                 "head_gradient_norm", "gradient_norm")), "Nonfinite pilot arithmetic")
    scheduler = decision["pilot_scheduler"]
    require(scheduler["GPU_job"] == "981620" and scheduler["state"] == "COMPLETED"
            and scheduler["exit_code"] == "0:0" and scheduler["node"] == "evc27",
            "Actual pilot scheduler completion required")
    require(decision["status"] == "accepted_full_predictive_fitting_recipe"
            and decision["pilot_acceptance_sha256"] == binding["density100_pilot"]["sha256"]
            and decision["source_git"] == binding["source_git"]
            and decision["code_sha256"] == binding["code_sha256"]
            and decision["recipe"] == recipe()
            and decision["pilot_warnings_reviewed"] == pilot["engineering_warnings"]
            and decision["observed_C_C0_gradient_risk_reviewed"] is True
            and decision["primary_B_C_C0_unchanged"] is True
            and decision["primary_Kaggle_reserve_seconds"] == 3360
            and decision["engineering_predictions_used_for_selection"] is False
            and decision["automatic_recipe_retuning"] is False
            and isinstance(decision["agent_reasoning"], str)
            and len(decision["agent_reasoning"]) >= 80, "Actual separate full-run decision required")
    original_decision = {
        k: v for k, v in decision.items() if k != "full_C100_agent_decision_sha256"
    }
    original_decision["status"] = "accepted_separate_full_C100_after_actual_pilot_review"
    require(sha(canonical(original_decision)) == decision["full_C100_agent_decision_sha256"],
            "Original separate agent decision differs")
    require(funding["branch"] == admitted["branch"] == "C100"
            and funding["arm"] == "C" and funding["run_root"] == RUN
            and funding["pilot_scope_retained"] is True
            and admitted["source_git"] == binding["source_git"]
            and admitted["code_sha256"] == binding["code_sha256"]
            and admitted["new_MRI_transferred_bytes"] == 0,
            "Original C0 admission/funding cannot substitute for C100")
    for name in ("original_portable_admission_sha256", "original_C0_binding_sha256"):
        require(re.fullmatch("[0-9a-f]{64}", binding[name])
                and admitted[name] == binding[name], "Original input reuse provenance differs")
    return {
        "branch": "C100", "base_science_git": BASE,
        "density100_pilot_sha256": binding["density100_pilot"]["sha256"],
        "density100_pilot_source_git": PILOT_SOURCE, "density100_pilot_GPU_job": pilot["GPU_job"],
        **{key: binding[key] for key in PROVENANCE_KEYS[-2:]},
    }


def validate_references(binding):
    return validate(binding, *(read_json(binding[name]) for name in
                    ("density100_pilot", "engineering_decision", "funding", "input_admission")))


def replace_once(text, old, new):
    require(text.count(old) == 1, "Pinned source anchor changed")
    return text.replace(old, new)


def derive_worker(raw):
    require(sha(raw) == WORKER_SHA, "Exact 44cc worker required")
    text = raw.decode()
    substitutions = [
        ('and arm in ("B", "C", "C0"),', 'and arm == "C",'),
        ('        "worker.py",\n',
         '        "worker.py",\n        "density100_contract.py",\n        "density100_telemetry.py",\n'),
        ('    from alias_provenance import accepted_renderers, renderer_counts, validate_terminal\n',
         '    from alias_provenance import accepted_renderers, renderer_counts, validate_terminal\n'
         '    from density100_contract import validate_references\n'
         '    branch_provenance = validate_references(binding)\n'),
        ('density_weight=0 if arm == "C0" else 0.1,', 'density_weight=100.0,'),
        ('    receipt_base = {\n', '    receipt_base = {\n        **branch_provenance,\n'),
        ('    class Cases:\n',
         '    from density100_telemetry import Monitor\n'
         '    monitor = Monitor(model)\n\n    class Cases:\n'),
        ('            self.loads += 1\n',
         '            self.loads += 1\n'
         '            monitor.begin_case(len(history) + 1, labels[index:index + 1], '
         'masks[index:index + 1])\n'),
        ('        with (root / "updates.jsonl").open("a") as handle:\n',
         '        telemetry.update(monitor.finish_update(record["step"], record["studies"]))\n'
         '        with (root / "updates.jsonl").open("a") as handle:\n'),
        ('    torch.cuda.synchronize()\n    require(\n        state["step"]',
         '    monitor.close()\n    torch.cuda.synchronize()\n    require(\n        state["step"]'),
    ]
    for old, new in substitutions:
        text = replace_once(text, old, new)
    return text


_PUBLISHER_HELPER = '''
def c100_provenance(binding):
    code = Path(binding["code_root"])
    helper = reference({"path": str(code / "density100_contract.py"),
                        "sha256": binding["code_sha256"]["density100_contract.py"]})
    spec = importlib.util.spec_from_file_location("frozen_density100_contract", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_references(binding)

'''


def derive_publisher(raw):
    require(sha(raw) == PUBLISHER_SHA, "Exact cbfc full publisher required")
    text = raw.decode()
    substitutions = [
        ('density_weight=0 if arm == "C0" else 0.1,', 'density_weight=100.0,'),
        ('def validate_binding(binding, start, side, epoch, binding_sha):',
         _PUBLISHER_HELPER + 'def validate_binding(binding, start, side, epoch, binding_sha):'),
        ('        arm in PARAMETERS and type(epoch)',
         '        arm == "C" and type(epoch)'),
        ('    checks = dict(\n', '    branch = c100_provenance(binding)\n    checks = dict(\n'),
        ('    for name in (\n        "context_index",',
         '    checks.update(branch)\n    for name in (\n        "context_index",'),
        ('    root, epoch = Path(binding["run_root"]), v["epoch"]\n',
         '    branch = c100_provenance(binding)\n'
         '    root, epoch = Path(binding["run_root"]), v["epoch"]\n'),
        ('        return previous\n',
         '        require(all(previous.get(k) == x for k, x in branch.items()),\n'
         '                "Original C100 publication provenance differs")\n        return previous\n'),
        ('    require(funding["arm"] == binding["arm"] and funding["epoch"] == epoch,',
         '    require(funding["branch"] == "C100" and funding["arm"] == binding["arm"] '
         'and funding["epoch"] == epoch,'),
        ('checkpoints/predictive-full3887-', 'checkpoints/predictive-C100-full3887-'),
        ('    provenance.update(\n', '    provenance.update(branch)\n    provenance.update(\n'),
        ('            {\n                "arm": binding["arm"],\n',
         '            {\n                **branch,\n                "arm": binding["arm"],\n'),
    ]
    for old, new in substitutions:
        text = replace_once(text, old, new)
    return text


def build(repo):
    """Local preparation only; no source, GPU, publication, or entry mutation."""
    repo = Path(repo)

    def git(path):
        return subprocess.check_output(["git", "show", BASE + ":" + path], cwd=repo)

    files = {}
    for name, expected in NUMERICAL_SHA.items():
        # Match original 44cc prepare_once.py and actual CPU979667, including
        # its empty import shims; repository initializers are not runtime bytes.
        raw = (b"" if name in ("rsna/__init__.py", "rsna/competitive/__init__.py")
               else git(name if name.startswith("rsna/") else ORIGINAL + name))
        require(sha(raw) == expected, "Pinned numerical source mismatch")
        files[name] = raw.decode()
    files["worker.py"] = derive_worker(git(ORIGINAL + "worker.py.txt"))
    files["publish_epoch.py"] = derive_publisher(git(ORIGINAL + "publish_epoch.py.txt"))
    files["density100_contract.py"] = Path(__file__).read_text()
    files["density100_telemetry.py"] = Path(__file__).with_name("telemetry.py").read_text()
    for name, source in files.items():
        compile(source, name, "exec")
    return files
