"""One immutable CPU-only private-source preparation; credentials through stdin."""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOURNAL = ROOT / "artifacts" / HERE.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-git", required=True)
    args = p.parse_args()
    pin = subprocess.check_output(
        ["git", "rev-parse", args.source_git], cwd=ROOT, text=True
    ).strip()
    os.umask(0o077)

    def frozen(path):
        return subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT, text=True)

    relative = str(HERE.relative_to(ROOT))
    for name in ("run_once.py", "prepare.py", "compile.py"):
        assert (HERE / name).read_text() == frozen(relative + "/" + name)
    assert (ROOT / ".env").stat().st_mode & 0o077 == 0
    values = {
        k.strip(): value.strip().strip("\"'")
        for line in (ROOT / ".env").read_text().splitlines()
        for k, sep, value in [line.partition("=")]
        if sep and not line.lstrip().startswith("#")
    }
    prior = "initiative/research/predictive-R6-additive-20261020-v1/"
    payload = dict(
        pin=pin,
        token=values["KAGGLE_API_TOKEN"],
        files={
            "compile.py": frozen(relative + "/compile.py"),
            **{name: frozen(prior + name) for name in ("supervisor.py", "fusion.py")},
            **{name: frozen("scripts/kaggle/" + name) for name in ("workflow.py", "components.py")},
        },
    )
    for field, name in (
        ("qualification", "actual-qualified.json"),
        ("public", "actual-public.json"),
    ):
        payload[field] = json.loads(
            frozen("initiative/research/predictive-C100-final-entry-20261021-v1/" + name)
        )["receipt"]
    program = (
        frozen(relative + "/prepare.py")
        + "\nimport sys\nprint(json.dumps(main(json.load(sys.stdin))),flush=True)\n"
    )
    spec = importlib.util.spec_from_loader("private_source_stdin", loader=None)
    t = importlib.util.module_from_spec(spec)
    exec(
        compile(
            frozen("initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"),
            "stdin_transport",
            "exec",
        ),
        t.__dict__,
    )
    JOURNAL.mkdir(mode=0o700, exist_ok=True)
    with (JOURNAL / "intent.json").open("x") as f:
        json.dump(
            dict(source_git=pin, new_GPU=False, new_account_mutations=0, automatic_retry=False), f
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
        "--time=00:05:00",
        "--job-name=rsna-C100-R6-CPU-prepare",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        t.COMMAND,
    ]
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
        text=True,
        capture_output=True,
        timeout=420,
    )
    with (JOURNAL / "transport.private.json").open("x") as f:
        json.dump(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), f)
    assert result.returncode == 0, "Inspect original preparation; no automatic retry"
    r = json.loads(result.stdout)
    with (JOURNAL / "result.json").open("x") as f:
        json.dump(r, f, indent=2)
    print(json.dumps({k: v for k, v in r.items() if k not in {"binding", "original_R6_public"}}))


if __name__ == "__main__":
    main()
