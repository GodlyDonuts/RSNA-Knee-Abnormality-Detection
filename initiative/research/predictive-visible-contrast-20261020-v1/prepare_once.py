"""Single original pilot-cache preparation; no GPU, transfer, or automatic retry."""

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
    candidate = REPO / "rsna/competitive/predictive/visible_contrast.py"
    paths = (candidate, HERE / "prepare_once.py", HERE / "prepare_contexts.py",
             HERE / "actual-candidate-input-result.json", HERE / "actual-candidate-terminal.json")
    for path in paths:
        assert subprocess.check_output(["git", "show", args.source_git + ":" + path.relative_to(REPO).as_posix()], cwd=REPO) == path.read_bytes()
    terminal = json.loads((HERE / "actual-candidate-terminal.json").read_text())
    assert (terminal["job_id"], terminal["state"], terminal["exit_code"]) == ("984583", "COMPLETED", "0:0")
    acceptance_body = (HERE / "actual-candidate-input-result.json").read_text()
    acceptance = json.loads(acceptance_body)
    source = (HERE / "prepare_contexts.py").read_text()
    payload = {"source_git": args.source_git, "prepare_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "candidate_source": candidate.read_text(), "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
               "scientific_modules": acceptance["scientific_modules"], "input_acceptance_body": acceptance_body,
               "input_acceptance_sha256": hashlib.sha256(acceptance_body.encode()).hexdigest()}
    assert payload["candidate_sha256"] == acceptance["candidate_sha256"]
    spec = importlib.util.spec_from_file_location("contrast_transport", REPO / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    argv = ["srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc33",
            "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=12G", "--time=00:15:00",
            "--immediate=60", "--job-name=rsna-visible-contrast16", "env", "CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python", "-B", "-c", transport.COMMAND]
    journal = REPO / "artifacts" / (HERE.name + "-contexts")
    journal.mkdir(mode=0o700, exist_ok=False)
    (journal / "intent.json").write_text(json.dumps({k: value for k, value in payload.items()
        if k not in ("candidate_source", "input_acceptance_body")}, sort_keys=True) + "\n")
    with (journal / "stdout.private").open("x") as out, (journal / "stderr.private").open("x") as err:
        result = subprocess.run(["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
                                input=transport.envelope(source, payload), text=True, stdout=out, stderr=err,
                                check=False, timeout=1020)
    (journal / "returned.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    assert result.returncode == 0, "Reconcile original private output; no automatic retry"
    receipt = json.loads((journal / "stdout.private").read_text())
    assert receipt["status"] == "actual_contrast_matched16_CPU_contexts_accepted"
    for key in ("source_git", "candidate_sha256", "input_acceptance_sha256"):
        assert receipt[key] == payload[key]
    assert receipt["context_files"] == 32 and receipt["all_roundtrips_bitwise_equal"]
    assert receipt["label_encoder_optimizer_GPU_reads_or_updates"] == 0
    with (HERE / "actual-contexts-prepared.json").open("x") as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
