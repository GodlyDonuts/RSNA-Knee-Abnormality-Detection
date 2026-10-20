"""Explicit single CPU phase. No daemon, polling owner, implicit dispatch or retries."""

import argparse
import importlib.util
import json
import shlex
import subprocess
import time
from pathlib import Path

R = Path(__file__).resolve().parents[3]
D = Path(__file__).resolve().parent
CPU_PHASES = ("stage", "prepare", "compile")
ACCOUNT_PHASES = ("dataset", "dataset-reconcile", "push", "status", "submit", "outcome")


def frozen(pin, path):
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=R, text=True)


def controls(pin):
    own = D.relative_to(R).as_posix()
    serving = "initiative/research/predictive-final-serving-20261020-v1"
    files = {name: frozen(pin, own + "/" + name) for name in ("policy.py", "cpu.py", "account.py")}
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", pin, serving], cwd=R, text=True
    ).splitlines()
    for name in names:
        if name.endswith((".py", ".py.txt")):
            files["serving/" + Path(name).name] = frozen(pin, name)
    for name in (
        "workflow.py",
        "components.py",
        "publication.py",
        "gpu_capacity.py",
        "output_csv.py",
    ):
        files["common/" + name] = frozen(pin, "scripts/kaggle/" + name)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=CPU_PHASES + ACCOUNT_PHASES)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Private token-free owning-node inputs, fresh funding/runtime plan",
    )
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument(
        "--mode", choices=("qualification", "B", "C", "C0"), default="qualification"
    )
    args = parser.parse_args()
    # Match this executable to the frozen owner commit, never silently mix later working edits.
    assert Path(__file__).read_text() == frozen(
        args.source_git, Path(__file__).relative_to(R).as_posix()
    )
    value = json.loads(args.config.read_text())
    assert not any("token" in k.lower() or "password" in k.lower() for k in value)
    value.update(phase=args.phase, mode=args.mode, source_git=args.source_git)
    own = controls(args.source_git)
    if args.phase == "stage":
        import hashlib

        value.update(
            sources=own,
            code_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in own.items()},
        )
    elif args.phase in ACCOUNT_PHASES:
        env = R / ".env"
        assert env.stat().st_mode & 0o077 == 0
        values = {
            k.strip(): z.strip().strip("\"'")
            for line in env.read_text().splitlines()
            for k, sep, z in [line.partition("=")]
            if sep and not line.lstrip().startswith("#")
        }
        value["token"] = values["KAGGLE_API_TOKEN"]
    transport_path = "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_loader("predictive_transport_frozen", loader=None)
    transport = importlib.util.module_from_spec(spec)
    exec(
        compile(frozen(args.source_git, transport_path), transport_path, "exec"), transport.__dict__
    )
    if args.phase == "stage":
        program = (
            "import json,sys,types\nv=json.load(sys.stdin)\n"
            "p=types.ModuleType('predictive_stage_policy')\n"
            "exec(compile(v['sources']['policy.py'],'frozen_policy','exec'),p.__dict__)\n"
            "g={'__name__':'predictive_stage_CPU','__file__':v['root']+'/control/cpu.py','_staged_policy':p}\n"
            "exec(compile(v['sources']['cpu.py'],'frozen_CPU','exec'),g)\n"
            "print(json.dumps(g['execute'](v)),flush=True)\n"
        )
    else:
        target = "cpu.py" if args.phase in CPU_PHASES else "account.py"
        program = (
            "import json,sys,runpy,io\nv=json.load(sys.stdin)\n"
            "sys.stdin=io.StringIO(json.dumps(v))\n"
            f"runpy.run_path(v['root']+'/control/{target}',run_name='__main__')\n"
        )
    command = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=" + value["owning_node"],
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=2G",
        "--time=00:30:00",
        "--job-name=rsna-predictive-final-" + args.phase,
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        transport.COMMAND,
    ]
    journal = R / "artifacts" / D.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = args.phase + "-" + args.mode + "-" + str(time.time_ns())
    with (journal / (attempt + "-intent.json")).open("x") as handle:
        json.dump(
            dict(
                phase=args.phase,
                mode=args.mode,
                source_git=args.source_git,
                config_path=str(args.config),
                automatic_retry=False,
            ),
            handle,
        )
    safe = (
        "try:\n"
        + "".join("    " + line + "\n" for line in program.splitlines())
        + "except Exception as e:\n import json\n"
        + " print(json.dumps({'status':'blocked','error_type':type(e).__name__}),flush=True)\n"
        + " raise SystemExit(1)\n"
    )
    result = subprocess.run(
        ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "newton", shlex.join(command)],
        input=transport.envelope(safe, value),
        capture_output=True,
        text=True,
        timeout=1900,
    )
    with (journal / (attempt + "-transport.json")).open("x") as handle:
        json.dump(
            dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), handle
        )
    assert result.returncode == 0, "Inspect original remote intent; never retry uncertain mutation"
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
