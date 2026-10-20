"""Finite sole pointwise account owner; never trains, recompiles or republishes to HF."""

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[2]


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ssh-socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("pointwise_single_phase", D / "run_once.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.controls(args.source_git)
    journal = R / "artifacts" / D.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not (journal / "owner-intent.json").exists()
    assert not list(journal.glob("*-intent.json")), (
        "Preexisting account phase requires explicit reconciliation"
    )
    source_zero = R / "artifacts/source-zero-final-serving-20261020-v2"
    assert (source_zero / "complete.json").is_file()
    old = json.loads((source_zero / "owner-intent.json").read_text())
    observed = subprocess.run(
        ["ps", "-p", str(old["PID"]), "-o", "command="], capture_output=True, text=True
    )
    assert "source-zero-final-serving-20261020-v2/follow_once.py" not in observed.stdout
    config = json.loads(args.config.read_text())
    with (journal / "config.json").open("x") as handle:
        json.dump(config, handle, indent=2)
    fixed_config = journal / "config.json"
    with (journal / "owner-intent.json").open("x") as handle:
        json.dump(
            dict(
                PID=os.getpid(),
                source_git=args.source_git,
                maximum_new_Datasets=1,
                maximum_new_GPU_sessions=2,
                maximum_new_entries=1,
                training_compilation_HF_publication_repeated=False,
                automatic_mutation_retry=False,
                maximum_seconds=12 * 3600,
            ),
            handle,
            indent=2,
        )
    deadline = time.monotonic() + 12 * 3600

    def action(phase, mode="qualification"):
        while time.monotonic() < deadline:
            out = runner.run(phase, mode, args.source_git, fixed_config, args.ssh_socket)
            (journal / "last-action.json").write_text(
                json.dumps(dict(phase=phase, mode=mode, actual=out), indent=2) + "\n"
            )
            if out["status"] not in runner.WAITING:
                return out
            time.sleep(45)
        raise RuntimeError("Finite deadline; preserve original phase for reconciliation")

    try:
        assert (
            action("inspect")["status"]
            == "actual_pointwise_compiled_and_absent_route_reconciled_read_only"
        )
        assert action("stage")["status"] == "actual_twoT4_epoch_serving_account_control_staged"
        assert action("dataset")["status"] == "pointwise_dataset_create_returned"
        verified = action("dataset-reconcile")
        assert verified["status"] == "all_payload_bytes_verified"
        (D / "actual-dataset-verified.json").write_text(json.dumps(verified, indent=2) + "\n")
        assert (
            action("prepare")["status"]
            == "actual_preserved_epoch_serving_prepared_without_model_reupload"
        )
        for mode in ("qualification", "competition"):
            pushed = action("push", mode)
            assert pushed["status"] == "one_exact_target_epoch_" + mode + "_pushed"
            (D / ("actual-" + mode + "-pushed.json")).write_text(
                json.dumps(pushed, indent=2) + "\n"
            )
            while time.monotonic() < deadline:
                outcome = action("status", mode)
                if outcome["status"] == "COMPLETE":
                    break
                assert outcome["status"] not in (
                    "ERROR",
                    "FAILED",
                    "CANCELLED",
                    "CANCEL_ACKNOWLEDGED",
                )
                time.sleep(45)
            else:
                raise RuntimeError("Finite public acceptance deadline reached")
            (D / ("actual-" + mode + "-accepted.json")).write_text(
                json.dumps(outcome, indent=2) + "\n"
            )
        submitted = action("submit", "competition")
        assert submitted["status"] == "one_exact_target_epoch_code_submission_accepted"
        (D / "actual-entry.json").write_text(json.dumps(submitted, indent=2) + "\n")
        while time.monotonic() < deadline:
            outcome = action("outcome", "competition")
            if outcome["outcome"] in ("scored", "failed"):
                (D / "actual-hidden-outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
                (journal / "complete.json").write_text(json.dumps(outcome, indent=2) + "\n")
                return
            time.sleep(60)
        raise RuntimeError("Finite hidden-outcome observation deadline reached")
    except BaseException as exc:
        (journal / "stopped.json").write_text(
            json.dumps(dict(error_type=type(exc).__name__, automatic_retry=False), indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
