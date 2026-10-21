"""One phase of the exact faster final preparation; same canonical coordinator."""

import importlib.util
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = HERE.parent / "predictive-density100-deployment-20261021-v1"
UPGRADE = HERE.parent / "predictive-selected-frame-preparation-20261021-v1"
BASE = "c9fbbb20"
ROLE = "Newton"


def git(pin, path):
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(REPO).as_posix()], cwd=REPO, text=True
    )


def load(body, path, name):
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    m.__file__ = str(path)
    exec(compile(body, str(path), "exec"), m.__dict__)
    return m


def namespace():
    parent = load(
        git(BASE, ORIGINAL / "run_once.py"), ORIGINAL / "run_once.py", "original_fast_phase"
    )
    ns = parent.namespace()
    baseline = ns["controls"]
    ns["__file__"] = str(HERE / "run_once.py")
    ns["D"] = HERE

    def controls(pin, value):
        paths = [
            HERE / "run_once.py",
            UPGRADE / "serving_upgrade.py",
            UPGRADE / "selected_frames.py",
            HERE.parent / "predictive-C100-private-backup-20261021-v1/storage_contract.py",
            HERE.parent / "predictive-descriptor-batch-20261021-v1/descriptor_batch.py",
        ]
        for path in paths:
            if git(pin, path) != path.read_text():
                raise ValueError("Execute exact committed faster final controls")
        upgrade = load(
            git(pin, UPGRADE / "serving_upgrade.py"),
            UPGRADE / "serving_upgrade.py",
            "accepted_faster_final_preparation",
        )
        if ROLE == "Colab":
            # The actual second recovery's immutable terminal truthfully says
            # local-only. Its independent seven publication proofs are admitted
            # by the existing updated contract; never rewrite that terminal.
            body = git(BASE, ORIGINAL / "controls.py")
            body = upgrade.replace(
                body,
                'and terminal["all_epochs_private_verified"] is True',
                "and compiler.C.terminal_storage_state(terminal)",
            )
            ctl = load(body, ORIGINAL / "controls.py", "Colab_local_terminal_controls")
            for name in ("controls.py", "prepare.py", "adopt.py"):
                if git(BASE, ORIGINAL / name) != git(pin, ORIGINAL / name):
                    raise ValueError("Original Colab control changed")
            files = ctl.build(value)
        else:
            files = baseline(pin, value)
        storage_path = (
            HERE.parent / "predictive-C100-private-backup-20261021-v1/storage_contract.py"
        )
        storage = load(git(pin, storage_path), storage_path, "C100_actual_private_storage_contract")
        files["serving/C100_contract.py"] = storage.extend(files["serving/C100_contract.py"])
        return upgrade.capsule(files)

    ns["controls"] = controls
    return ns


if __name__ == "__main__":
    namespace()["main"]()
