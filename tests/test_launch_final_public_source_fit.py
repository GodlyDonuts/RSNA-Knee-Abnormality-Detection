import json
import sys

import pytest
from test_final_public_source_t4_recovery import final_config

from scripts.newton import launch_final_public_source_fit as launch


def save(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, sort_keys=True, indent=2))
    return p


def admitted(tmp_path):
    values = final_config()
    overlay = save(tmp_path, "overlay.json", {"scope": "synthetic engineering fixture"})
    values["fitting_supervision"]["sha256"] = launch.sha(overlay)
    config = save(tmp_path, "final-config.json", values)
    save(
        tmp_path,
        "input-accepted.json",
        dict(
            status="actual_final_public_source_CPU_roster_and_16MRI_inputs_verified",
            source_git="1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9",
            training_studies=3403,
            outer_studies=0,
            config_sha256=launch.sha(config),
            overlay_sha256=launch.sha(overlay),
        ),
    )
    code = tmp_path / "code/rsna/train.py"
    code.parent.mkdir(parents=True)
    code.write_text("unchanged source")
    save(tmp_path, "source.json", {"files": {"code/rsna/train.py": launch.sha(code)}})
    save(
        tmp_path,
        "full-cache-transfer-verified.json",
        dict(
            members=37812,
            bytes=140626483351,
            registry_sha256="73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770",
            all_original_member_size_SHA_verified_during_transfer=True,
        ),
    )
    save(
        tmp_path,
        "T4-recovery-accepted.json",
        {
            "actual_recovery": dict(
                status="actual_single_T4_final_public_source_recovery_accepted",
                science_git="1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9",
                exact_complete_state_scaler_RNG_recovery=True,
                amp_scaler={"_growth_tracker": 4},
            )
        },
    )
    recovery_path = tmp_path / "T4-recovery-accepted.json"
    recovery = json.loads(recovery_path.read_text())
    cpu = json.loads((tmp_path / "input-accepted.json").read_text())
    recovery["actual_recovery"]["CPU_receipt_sha256"] = (
        __import__("hashlib")
        .sha256((json.dumps(cpu, sort_keys=True, indent=2) + "\n").encode())
        .hexdigest()
    )
    recovery_path.write_text(json.dumps(recovery))
    save(
        tmp_path,
        "trainer-inputs.json",
        dict(
            cache="/original/cache",
            studies="/original/studies",
            folds="/original/folds",
            series="/original/series",
            **{"artifact-root": "/original/artifacts"},
        ),
    )
    return tmp_path, values


def test_original_full_budget_preserved_and_only_cache_path_relocated(tmp_path):
    root, original = admitted(tmp_path)
    values, inputs = launch.validate_admission(root)
    assert (
        values == original
        and values["epochs"] == 10
        and values["warmup_updates"] == 426
        and values["diagnostic_study_limit"] is None
    )
    assert inputs["cache"] == str(root / "cache") and inputs["studies"] == "/original/studies"
    cmd = launch.command(root, inputs)
    assert (
        cmd[0] == sys.executable
        and "--resume" not in cmd
        and cmd[cmd.index("--config") + 1] == str(root / "final-config.json")
    )
    assert cmd[cmd.index("--output") + 1] == str(root / "production")


@pytest.mark.parametrize("kind", ["cache", "source", "recovery", "overlay"])
def test_incomplete_or_changed_admission_never_launches(kind, tmp_path):
    root, _ = admitted(tmp_path)
    if kind == "cache":
        p = root / "full-cache-transfer-verified.json"
        v = json.loads(p.read_text())
        v["members"] = 16
        p.write_text(json.dumps(v))
    elif kind == "source":
        (root / "code/rsna/train.py").write_text("changed")
    elif kind == "recovery":
        p = root / "T4-recovery-accepted.json"
        v = json.loads(p.read_text())
        v["actual_recovery"]["amp_scaler"]["_growth_tracker"] = 3
        p.write_text(json.dumps(v))
    else:
        (root / "overlay.json").write_text("changed")
    with pytest.raises(ValueError):
        launch.validate_admission(root)
    assert not (root / "production").exists()
