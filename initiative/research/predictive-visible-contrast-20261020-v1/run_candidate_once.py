"""One CPU-only original-input experiment, source via stdin, zero remote writes."""

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
    for path in (candidate, HERE / "probe_candidate.py", HERE / "run_candidate_once.py", HERE / "visible_pixels.py"):
        assert subprocess.check_output(["git", "show", args.source_git + ":" + path.relative_to(REPO).as_posix()], cwd=REPO) == path.read_bytes()
    source = (HERE / "probe_candidate.py").read_text()
    original = json.loads((HERE / "actual-exact-visible-result.json").read_text())
    payload = {"source_git": args.source_git, "collector_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "candidate_source": candidate.read_text(), "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
               "measurement_source": (HERE / "visible_pixels.py").read_text(),
               "measurement_sha256": hashlib.sha256((HERE / "visible_pixels.py").read_bytes()).hexdigest(),
               "scientific_modules": original["scientific_modules"],
               "prior_visible_summary": original["equal_context_then_study_summary"]}
    spec = importlib.util.spec_from_file_location("contrast_transport", REPO / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    argv = ["srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc33",
            "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=12G", "--time=00:10:00",
            "--immediate=60", "--job-name=rsna-visible-contrast-candidate", "env", "CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python", "-B", "-c", transport.COMMAND]
    journal = REPO / "artifacts" / (HERE.name + "-candidate")
    journal.mkdir(mode=0o700, exist_ok=False)
    (journal / "intent.json").write_text(json.dumps({k: value for k, value in payload.items()
        if not k.endswith("_source")}, sort_keys=True) + "\n")
    with (journal / "stdout.private").open("x") as out, (journal / "stderr.private").open("x") as err:
        result = subprocess.run(["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
                                input=transport.envelope(source, payload), text=True, stdout=out, stderr=err,
                                check=False, timeout=720)
    (journal / "returned.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    assert result.returncode == 0, "Inspect the original result; never repeat an uncertain dispatch"
    receipt = json.loads((journal / "stdout.private").read_text())
    assert receipt["status"] == "actual_twelve_fitting_MRI_visible_only_contrast_candidate"
    for key in ("source_git", "collector_sha256", "candidate_sha256", "measurement_sha256", "scientific_modules"):
        assert receipt[key] == payload[key]
    assert receipt["fitting_studies"] == 12 and receipt["excluded_engineering_studies"] == 4
    assert receipt["new_remote_files"] == receipt["optimizer_updates"] == 0
    assert not receipt["model_optimizer_labels_evaluation_inputs_read"]
    with (HERE / "actual-candidate-input-result.json").open("x") as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "CPU_job", "elapsed_seconds",
                     "equal_context_then_study_summary", "studies_below_point05_span")}), flush=True)


if __name__ == "__main__":
    main()
