"""Private evc33→evc27 stream inside Newton; one immutable transfer intent."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    assert re.fullmatch("[0-9a-f]{40}", args.source_git)
    os.umask(0o077)
    for name in ("transfer.py", "transfer_once.py", "actual-contexts-prepared.json", "actual-contexts-terminal.json"):
        path = HERE / name
        assert subprocess.check_output(["git", "show", args.source_git + ":" + path.relative_to(REPO).as_posix()], cwd=REPO) == path.read_bytes()
    accepted = json.loads((HERE / "actual-contexts-prepared.json").read_text())
    terminal = json.loads((HERE / "actual-contexts-terminal.json").read_text())
    assert (terminal["job_id"], terminal["state"], terminal["exit_code"]) == ("984645", "COMPLETED", "0:0")
    assert accepted["CPU_job"] == "984645" and accepted["all_roundtrips_bitwise_equal"]
    actual_body = json.dumps(accepted, sort_keys=True, allow_nan=False) + "\n"
    source = (HERE / "transfer.py").read_text()
    payload = {"source_git": args.source_git, "accepted": accepted,
               "remote_actual_sha256": hashlib.sha256(actual_body.encode()).hexdigest()}

    def command(node, mode):
        python = ("/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python" if node == "evc33"
                  else "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python")
        return shlex.join(["srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=" + node,
            "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=2G", "--time=00:10:00", "--immediate=60",
            "--job-name=rsna-contrast16-" + mode, "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1",
            "PYTHONDONTWRITEBYTECODE=1", python, "-B", "-c", source, mode, json.dumps(payload)])

    journal = REPO / "artifacts" / (HERE.name + "-transfer")
    journal.mkdir(mode=0o700, exist_ok=False)
    (journal / "intent.json").write_text(json.dumps({**payload, "transfer_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "private_pixels_in_cluster_pipe_only": True, "automatic_retry": False}, sort_keys=True) + "\n")
    shell = "set -o pipefail\n" + command("evc33", "send") + " | " + command("evc27", "receive")
    with (journal / "stdout.private").open("x") as out, (journal / "stderr.private").open("x") as err:
        result = subprocess.run(["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton",
                                 shlex.join(["bash", "-c", shell])], stdout=out, stderr=err, timeout=720, check=False)
    (journal / "returned.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    assert result.returncode == 0, "Inspect original sender/receiver and retained output; do not retry"
    receipt = json.loads((journal / "stdout.private").read_text())
    assert receipt["status"] == "actual_contrast_matched16_private_transfer_accepted"
    assert receipt["source_git"] == args.source_git and receipt["source_actual_sha256"] == payload["remote_actual_sha256"]
    assert receipt["source_index_sha256"] == accepted["private_index_sha256"]
    assert receipt["context_files"] == 32 and receipt["bytes"] == accepted["cache_bytes"]
    with (HERE / "actual-private-transfer.json").open("x") as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
