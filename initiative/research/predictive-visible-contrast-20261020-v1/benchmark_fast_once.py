"""Run the new CPU timing comparison once; original pilot/cache jobs stay untouched."""

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
    original = REPO / "rsna/competitive/predictive/visible_contrast.py"
    fast = REPO / "rsna/competitive/predictive/visible_contrast_fast.py"
    for path in (original, fast, HERE / "benchmark_fast.py", HERE / "benchmark_fast_once.py",
                 HERE / "actual-candidate-input-result.json"):
        assert subprocess.check_output(["git", "show", args.source_git + ":" + path.relative_to(REPO).as_posix()], cwd=REPO) == path.read_bytes()
    payload = {"source_git": args.source_git, "original_source": original.read_text(),
               "fast_source": fast.read_text(), "fast_sha256": hashlib.sha256(fast.read_bytes()).hexdigest(),
               "scientific_modules": json.loads((HERE / "actual-candidate-input-result.json").read_text())["scientific_modules"]}
    source = (HERE / "benchmark_fast.py").read_text()
    spec = importlib.util.spec_from_file_location("contrast_transport", REPO /
        "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    journal = REPO / "artifacts" / (HERE.name + "-fast-timing")
    journal.mkdir(mode=0o700, exist_ok=False)
    (journal / "intent.json").write_text(json.dumps({"source_git": args.source_git,
        "fast_sha256": payload["fast_sha256"], "benchmark_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "new_GPU_jobs": 0, "new_remote_files": 0, "automatic_retry": False}) + "\n")
    argv = ["srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc33",
            "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=12G", "--time=00:10:00",
            "--immediate=60", "--job-name=rsna-contrast-fast-timing", "env", "CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python", "-B", "-c", transport.COMMAND]
    result = subprocess.run(["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
                            input=transport.envelope(source, payload), text=True, capture_output=True, timeout=720, check=False)
    (journal / "transport.private.json").write_text(json.dumps({"returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    assert result.returncode == 0, "Inspect original job; do not repeat an uncertain comparison"
    receipt = json.loads(result.stdout)
    assert receipt["source_git"] == args.source_git and receipt["fast_source_sha256"] == payload["fast_sha256"]
    assert receipt["status"] == "actual_twelve_fitting_MRI_equivalent_contrast_geometry_timing"
    assert receipt["all_context_and_support_fields_bitwise_equal"] and receipt["all_visible_scales_exactly_equal"]
    with (HERE / "actual-fast-timing.json").open("x") as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
