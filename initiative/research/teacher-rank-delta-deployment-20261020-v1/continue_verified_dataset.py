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
PIN = "dd5659ee940709443b577852d6558d0ee0cf492e"


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location("original_rank_delta_phase", D / "run_once.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.controls(PIN)
    old = json.loads((J / "owner-intent.json").read_text())
    assert old["source_git"] == PIN and old["PID"] == 19114
    assert (J / "stopped.json").exists()
    observed = subprocess.run(
        ["ps", "-p", str(old["PID"]), "-o", "command="], capture_output=True, text=True
    )
    assert "teacher-rank-delta-deployment-20261020-v1/follow_once.py" not in observed.stdout
    verified = json.loads((J / "dataset-reconcile-competition-result.json").read_text())
    assert verified["status"] == "all_payload_bytes_verified"
    assert verified["id"] == 12739137 and verified["version"] == 1
    assert verified["private"] is True and verified["downloaded_all_bytes"] is True
    assert verified["slug"] == "godlydonuts/rsna-teacher-rank-final-e3-d478a502e6fd"
    assert verified["files"]["epoch-003.pt"]["sha256"] == "d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca"
    assert not (J / "continuation-intent.json").exists()
    assert not any(J.glob("prepare-*-intent.json")) and not any(J.glob("push-*-intent.json"))
    assert not any(J.glob("submit-*-intent.json"))
    save(D / "actual-dataset-verified.json", verified)
    save(
        J / "continuation-intent.json",
        dict(
            PID=os.getpid(),
            original_source_git=PIN,
            original_owner_stopped=True,
            original_Dataset_id=12739137,
            new_Dataset_calls=0,
            maximum_new_GPU_sessions=1,
            maximum_new_entries=1,
            automatic_mutation_retry=False,
        ),
    )
    remaining = (J / "owner-intent.json").stat().st_mtime + 12 * 3600 - time.time()
    assert remaining > 0, "Original finite owner deadline has passed"
    deadline = time.monotonic() + remaining

    def action(phase, mode="competition"):
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
        for mode in ("competition",):
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
