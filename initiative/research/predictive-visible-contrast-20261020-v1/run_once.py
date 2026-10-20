"""One bounded zero-GPU observation; no new remote files or model changes."""

import argparse
import hashlib
import importlib.util
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
    for name in ("collect.py", "run_once.py"):
        p = HERE / name
        assert subprocess.check_output(["git", "show", args.source_git + ":" + p.relative_to(REPO).as_posix()], cwd=REPO) == p.read_bytes()
    source = (HERE / "collect.py").read_text()
    payload = dict(source_git=args.source_git, collector_sha256=hashlib.sha256(source.encode()).hexdigest())
    spec = importlib.util.spec_from_file_location("contrast_transport", REPO / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    argv = ["srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc27",
            "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=2G", "--time=00:05:00",
            "--immediate=60", "--job-name=rsna-predictive-visible-contrast", "env", "CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python", "-B", "-c", transport.COMMAND]
    journal = REPO / "artifacts" / HERE.name
    journal.mkdir(mode=0o700, exist_ok=False)
    (journal / "intent.json").write_text(json.dumps({**payload, "CPU": 1, "GPU": 0,
        "maximum_seconds": 300, "new_remote_files": 0, "automatic_retry": False}) + "\n")
    with (journal / "stdout.private").open("x") as out, (journal / "stderr.private").open("x") as err:
        result = subprocess.run(["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
                                input=transport.envelope(source, payload), text=True, stdout=out, stderr=err,
                                check=False, timeout=420)
    (journal / "returned.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    assert result.returncode == 0, "Observe original failure, never silently retry"
    receipt = json.loads((journal / "stdout.private").read_text())
    assert receipt["status"] == "actual_twelve_fitting_case_rendered_contrast_readonly"
    assert all(receipt[k] == value for k, value in payload.items())
    assert receipt["fitting_studies"] == 12 and receipt["excluded_engineering_studies"] == 4
    assert not receipt["clinical_labels_models_or_evaluation_inputs_read"]
    assert receipt["new_remote_files_or_changed_cache_bytes"] == receipt["optimizer_updates"] == 0
    with (HERE / "actual-result.json").open("x") as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
