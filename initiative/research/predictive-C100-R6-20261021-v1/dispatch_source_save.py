"""Explicit single source-save invocation; no automatic polling, GPU run or entry."""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOURNAL = ROOT / "artifacts" / HERE.name / "source-only"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-git", required=True)
    a = p.parse_args()
    pin = subprocess.check_output(["git", "rev-parse", a.source_git], cwd=ROOT, text=True).strip()

    def frozen(path):
        return subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT, text=True)

    rel = str(HERE.relative_to(ROOT))
    for name in ("dispatch_source_save.py", "source_save.py"):
        assert (HERE / name).read_text() == frozen(rel + "/" + name)
    assert not (ROOT / ".env").stat().st_mode & 0o077
    values = {
        k.strip(): x.strip().strip("\"'")
        for line in (ROOT / ".env").read_text().splitlines()
        for k, sep, x in [line.partition("=")]
        if sep and not line.lstrip().startswith("#")
    }
    payload = dict(
        pin=pin,
        token=values["KAGGLE_API_TOKEN"],
        compiler_source=subprocess.check_output(
            ["git", "show", "0e7ef257b04aef117f1b4173d1a09a824d243c2a:" + rel + "/compile.py"],
            cwd=ROOT,
            text=True,
        ),
        workflow_source=subprocess.check_output(
            ["git", "show", "0e7ef257b04aef117f1b4173d1a09a824d243c2a:scripts/kaggle/workflow.py"],
            cwd=ROOT,
            text=True,
        ),
    )
    program = (
        frozen(rel + "/source_save.py")
        + "\nimport sys\nprint(json.dumps(main(json.load(sys.stdin))),flush=True)\n"
    )
    s = importlib.util.spec_from_loader("original_stdin_envelope", loader=None)
    t = importlib.util.module_from_spec(s)
    exec(
        compile(
            frozen("initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"),
            "original_stdin_envelope",
            "exec",
        ),
        t.__dict__,
    )
    os.umask(0o077)
    JOURNAL.mkdir(mode=0o700, exist_ok=True)
    attempt = JOURNAL / str(time.time_ns())
    attempt.mkdir(mode=0o700)
    (attempt / "intent.json").write_text(
        json.dumps(
            dict(
                source_git=pin,
                phase="conditional_source_only",
                GPU_run=False,
                competition_entry=False,
                automatic_retry=False,
            )
        )
    )
    cmd = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc27",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=1G",
        "--time=00:08:00",
        "--job-name=rsna-C100-R6-source-save",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        t.COMMAND,
    ]
    x = subprocess.run(
        [
            "ssh",
            "-S",
            "/tmp/rsna-research-20261001-stable.sock",
            "-o",
            "BatchMode=yes",
            "newton",
            shlex.join(cmd),
        ],
        input=t.envelope(program, payload),
        text=True,
        capture_output=True,
        timeout=600,
    )
    (attempt / "transport.private.json").write_text(
        json.dumps(dict(returncode=x.returncode, stdout=x.stdout, stderr=x.stderr))
    )
    assert x.returncode == 0, "Inspect original source-save result/claim; no automatic retry"
    result = json.loads(x.stdout)
    (attempt / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
