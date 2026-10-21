"""Explicit one-time CPU notebook push or read-only observation; credentials via stdin."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAYLOAD = ROOT / "artifacts" / HERE.name
JOURNAL = PAYLOAD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("push", "status"))
    p.add_argument("--source-git", required=True)
    args = p.parse_args()
    args.source_git = subprocess.check_output(
        ["git", "rev-parse", args.source_git], cwd=ROOT, text=True
    ).strip()
    os.umask(0o077)

    def frozen(path):
        return subprocess.check_output(
            ["git", "show", args.source_git + ":" + path], cwd=ROOT, text=True
        )

    for name in ("run_once.py", "account.py", "benchmark.py", "compile.py", "descriptor_batch.py"):
        assert (HERE / name).read_text() == frozen(str((HERE / name).relative_to(ROOT)))
    assert (ROOT / ".env").stat().st_mode & 0o077 == 0
    values = {
        k.strip(): v.strip().strip("\"'")
        for line in (ROOT / ".env").read_text().splitlines()
        for k, sep, v in [line.partition("=")]
        if sep and not line.lstrip().startswith("#")
    }
    source = (PAYLOAD / "candidate.py").read_text()
    compiled = json.loads((PAYLOAD / "compiled.json").read_text())
    JOURNAL.mkdir(mode=0o700, exist_ok=True)
    assert (
        compiled["binding"]["module_sha256"]["benchmark.py"]
        == hashlib.sha256((HERE / "benchmark.py").read_bytes()).hexdigest()
    )
    assert hashlib.sha256(source.encode()).hexdigest() == compiled["source_sha256"]
    payload = {
        "phase": args.phase,
        "pin": args.source_git,
        "source": source,
        "compiled": compiled,
        "token": values["KAGGLE_API_TOKEN"],
        "common": {n: frozen("scripts/kaggle/" + n) for n in ("workflow.py", "components.py")},
    }
    program = (
        frozen(str((HERE / "account.py").relative_to(ROOT)))
        + "\nimport sys\nprint(json.dumps(main(json.load(sys.stdin))),flush=True)\n"
    )
    path = ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_loader("CPU_preparation_stdin", loader=None)
    t = importlib.util.module_from_spec(spec)
    exec(compile(frozen(str(path.relative_to(ROOT))), str(path), "exec"), t.__dict__)
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
        "--job-name=rsna-CPU-prepare-" + args.phase,
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        t.COMMAND,
    ]
    marker = "push" if args.phase == "push" else "status-" + str(time.time_ns())
    intent = JOURNAL / (marker + "-intent.json")
    with intent.open("x") as f:
        json.dump(
            {
                "phase": args.phase,
                "source_git": args.source_git,
                "source_sha256": compiled["source_sha256"],
                "GPU": False,
                "competition_entry": False,
                "automatic_retry": False,
            },
            f,
        )
    result = subprocess.run(
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
        capture_output=True,
        text=True,
        timeout=600,
    )
    with (JOURNAL / (marker + "-transport.private.json")).open("x") as f:
        json.dump(
            {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, f
        )
    assert result.returncode == 0, (
        "CPU coordinator failed; inspect original result before any new action"
    )
    parsed = json.loads(result.stdout)
    with (JOURNAL / (marker + "-result.json")).open("x") as f:
        json.dump(parsed, f, indent=2)
    print(json.dumps({k: v for k, v in parsed.items() if k != "receipt"}))


if __name__ == "__main__":
    main()
