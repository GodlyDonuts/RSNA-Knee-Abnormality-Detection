"""Continue after the original Dataset is independently verified; never recreate it."""

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[2]
J = R / "artifacts" / D.name
PIN = "01cbe205080487b3fb39fbf7c0d28e112fcd7d77"


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    spec = importlib.util.spec_from_file_location("original_pointwise_phase", D / "run_once.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.controls(PIN)
    old = json.loads((J / "owner-intent.json").read_text())
    assert old["source_git"] == PIN and old["PID"] == 71909
    assert (J / "stopped.json").exists()
    observed = subprocess.run(
        ["ps", "-p", str(old["PID"]), "-o", "command="], capture_output=True, text=True
    )
    assert "pointwise-final-deployment-20261020-v1/follow_once.py" not in observed.stdout
    actual = json.loads((D / "actual-created-dataset-readonly.json").read_text())
    assert actual["CPU_job"] == "979849" and not actual["claim_present"]
    assert actual["GetDatasetStatus"] == {"status": "READY"}
    assert actual["GetDataset"]["id"] == 12731583
    assert actual["GetDataset"]["currentVersionNumber"] == 1
    assert actual["GetDataset"]["isPrivate"] is True
    verified = json.loads((J / "dataset-reconcile-qualification-result.json").read_text())
    assert verified["status"] == "all_payload_bytes_verified"
    assert verified["id"] == 12731583 and verified["version"] == 1
    assert verified["private"] is True and verified["downloaded_all_bytes"] is True
    assert verified["slug"] == "godlydonuts/rsna-pointwise-final-e3-da61c5348093"
    assert not any(J.glob("prepare-*-intent.json")) and not any(J.glob("push-*-intent.json"))
    save(D / "actual-dataset-verified.json", verified)
    save(
        J / "continuation-intent.json",
        dict(
            PID=os.getpid(),
            original_source_git=PIN,
            original_owner_stopped=True,
            original_Dataset_id=12731583,
            new_Dataset_calls=0,
            maximum_new_GPU_sessions=2,
            maximum_new_entries=1,
            automatic_mutation_retry=False,
        ),
    )
    deadline = time.monotonic() + 12 * 3600

    def action(phase, mode="qualification"):
        while time.monotonic() < deadline:
            value = runner.run(phase, mode, PIN, J / "config.json", "/tmp/rsna-research-20261001-stable.sock")
            (J / "continuation-last-action.json").write_text(
                json.dumps(dict(phase=phase, mode=mode, actual=value), indent=2) + "\n"
            )
            if value["status"] not in runner.WAITING:
                return value
            time.sleep(45)
        raise RuntimeError("Finite continuation deadline; reconcile the original state")

    try:
        assert action("prepare")["status"] == "actual_preserved_epoch_serving_prepared_without_model_reupload"
        for mode in ("qualification", "competition"):
            pushed = action("push", mode)
            assert pushed["status"] == "one_exact_target_epoch_" + mode + "_pushed"
            save(D / ("actual-" + mode + "-pushed.json"), pushed)
            while time.monotonic() < deadline:
                result = action("status", mode)
                if result["status"] == "COMPLETE":
                    break
                assert result["status"] not in ("ERROR", "FAILED", "CANCELLED", "CANCEL_ACKNOWLEDGED")
                time.sleep(45)
            else:
                raise RuntimeError("Finite qualification/public observation deadline")
            save(D / ("actual-" + mode + "-accepted.json"), result)
        entry = action("submit", "competition")
        assert entry["status"] == "one_exact_target_epoch_code_submission_accepted"
        save(D / "actual-entry.json", entry)
        while time.monotonic() < deadline:
            result = action("outcome", "competition")
            if result["outcome"] in ("scored", "failed"):
                save(D / "actual-hidden-outcome.json", result)
                save(J / "complete.json", result)
                return
            time.sleep(60)
        raise RuntimeError("Finite hidden-outcome observation deadline")
    except BaseException as exc:
        save(J / "continuation-stopped.json", dict(error_type=type(exc).__name__, automatic_retry=False))
        raise


if __name__ == "__main__":
    main()
