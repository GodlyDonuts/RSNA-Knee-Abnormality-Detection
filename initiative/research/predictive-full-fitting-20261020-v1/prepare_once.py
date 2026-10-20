"""Stage small full-fitting controls once on the original owning node, without a GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REMOTE = "/tmp/rsna-predictive-full-fitting-20261020-v1"
PYTHON = "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python"
SOCKET = "/tmp/rsna-research-20261001-stable.sock"


def main():
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = {
        path.relative_to(ROOT).as_posix(): path
        for path in (ROOT / "rsna/competitive/predictive").glob("*.py")
    }
    paths.update(
        {
            "worker.py": CONTROL / "worker.py.txt",
            "publish_epoch.py": CONTROL / "publish_epoch.py.txt",
            "admit_cpu.py": CONTROL / "admit_cpu.py.txt",
        }
    )
    for path in (
        *paths.values(),
        CONTROL / "engineering-decision.json",
        Path(__file__),
    ):
        assert path.read_bytes() == subprocess.check_output(
            ["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT
        )
    files = {name: path.read_text() for name, path in paths.items()}
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    decision = (CONTROL / "engineering-decision.json").read_text()
    payload = {
        "mode": "prepare",
        "root": REMOTE,
        "source_git": pin,
        "files": files,
        "sha256": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()},
        "engineering_decision_text": decision,
        "engineering_decision_sha256": hashlib.sha256(decision.encode()).hexdigest(),
    }
    # Reuse paid original caches, never double-count them as still-unpaid. The
    # conservative60GiB reserve includes the original24GiB B/C model scope and
    # the additional12GiB C0 scope alongside older retained obligations.
    program = """
import hashlib,json,os,runpy,shutil,sys
from pathlib import Path
v=json.load(sys.stdin);os.umask(0o077)
raw=Path('/tmp/rsna-predictive-full-raw-cache-20261020-v1')
context=Path('/tmp/rsna-predictive-full-context-cache-20261020-v1')
paid_raw=json.loads(Path('/tmp/rsna-predictive-raw16-cache-20261020-v1/actual.json').read_text())['cache_bytes']+sum(p.stat().st_size for p in (raw/'cache').glob('*.npz'))
paid_context=json.loads(Path('/tmp/rsna-predictive-context16-cache-20261020-v1/actual.json').read_text())['cache_bytes']+sum(p.stat().st_size for p in context.glob('context-*.npz'))
v['funding']={'other_unpaid_node_liabilities_bytes':60*2**30+566*2**20+max(0,320*2**30-paid_raw)+max(0,96*2**30-paid_context),'shared_unpaid_bytes':92*2**20,'shared_unpaid_files':368}
namespace={'__name__':'cpu_admission'}
exec(compile(v['files']['admit_cpu.py'],'admit_cpu.py','exec'),namespace)
result=namespace['main'](v)
root=Path(v['root'])
assert hashlib.sha256(v['engineering_decision_text'].encode()).hexdigest()==v['engineering_decision_sha256']
with (root/'engineering-decision.json').open('x') as handle:
 handle.write(v['engineering_decision_text'])
result.update(funding=v['funding'],engineering_decision_sha256=v['engineering_decision_sha256'],prepared_private_sha256=namespace['sha'](root/'prepared.private.json'),new_GPU_reservations=0)
namespace['write_once'](root/'actual-preparation.json',result)
print(json.dumps(result),flush=True)
"""
    local = ROOT / "artifacts" / CONTROL.name
    local.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (local / "prepare-intent.json").open("x") as handle:
        json.dump(
            {
                "source_git": pin,
                "code_sha256": payload["sha256"],
                "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                "new_GPU_reservations": 0,
                "new_pixel_weight_runtime_copies": 0,
                "automatic_retry": False,
            },
            handle,
        )
    transport_path = (
        ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    )
    spec = importlib.util.spec_from_file_location("full_prepare_transport", transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    args = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc33",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=8G",
        "--time=00:15:00",
        "--job-name=rsna-predictive-full-CPU-prepare",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        PYTHON,
        "-B",
        "-c",
        transport.COMMAND,
    ]
    result = subprocess.run(
        ["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "newton", shlex.join(args)],
        input=transport.envelope(program, payload),
        capture_output=True,
        text=True,
        timeout=1500,
    )
    with (local / "prepare-transport.json").open("x") as handle:
        json.dump(
            dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
            handle,
            indent=2,
        )
    assert result.returncode == 0, (
        "Reconcile original CPU intent read-only; never repeat preparation"
    )
    actual = json.loads(result.stdout)
    assert actual["status"] == "actual_predictive_full_CPU_source_parent_initialization_accepted"
    with (CONTROL / "actual-cpu-prepared.json").open("x") as handle:
        handle.write(json.dumps(actual, indent=2) + "\n")
    print(json.dumps(actual), flush=True)


if __name__ == "__main__":
    main()
