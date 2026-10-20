"""Stage small full-fitting controls once on the original owning node, without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REMOTE = "/tmp/rsna-predictive-full-fitting-20261020-v2"
PYTHON = "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python"
SOCKET = "/tmp/rsna-research-20261001-stable.sock"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    pin = args.source_git
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)
    original = json.loads(
        (
            ROOT
            / "initiative/research/predictive-full-fitting-20261020-v1/actual-cpu-prepared.json"
        ).read_text()
    )
    failure = ROOT / "initiative/research/predictive-alias-failure-20261020-v1"
    regression_text = (failure / "actual-regression.json").read_text()
    stopped_text = (failure / "actual-old-fitting-owner-stopped.json").read_text()
    helper = {}
    exec(
        compile((CONTROL / "alias_provenance.py").read_text(), "alias_provenance.py", "exec"),
        helper,
    )
    helper["validate_owner_stop"](json.loads(stopped_text))
    assert (
        hashlib.sha256(regression_text.encode()).hexdigest()
        == "946911a47151af80225cf28921876e88e089c35c86e4c11061ca141bbe2e463e"
    )
    prior = subprocess.run(["ps", "-p", "59538", "-o", "command="], capture_output=True, text=True)
    assert "predictive-full-fitting-20261020-v1/owner.py" not in prior.stdout
    old_journal = ROOT / "artifacts/predictive-full-fitting-20261020-v1/finite-owner"
    assert {p.name for p in old_journal.iterdir()} == {
        "launcher.json",
        "owner-intent.json",
        "stdout",
        "stderr",
    }

    paths = {name: ROOT / name for name in original["code_sha256"] if name.startswith("rsna/")}
    paths.update(
        {
            "worker.py": CONTROL / "worker.py.txt",
            "publish_epoch.py": CONTROL / "publish_epoch.py.txt",
            "admit_cpu.py": CONTROL / "admit_cpu.py.txt",
            "alias_provenance.py": CONTROL / "alias_provenance.py",
        }
    )
    for path in (
        *paths.values(),
        ROOT / "initiative/research/predictive-full-fitting-20261020-v1/engineering-decision.json",
        Path(__file__),
    ):
        assert path.read_bytes() == subprocess.check_output(
            ["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT
        )
    files = {name: path.read_text() for name, path in paths.items()}
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    master_path = (
        ROOT / "initiative/research/predictive-full-fitting-20261020-v1/engineering-decision.json"
    )
    master = json.loads(master_path.read_text())
    hashes = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
    accepted = helper["accepted_renderers"](
        original["code_sha256"], hashes, json.loads(regression_text)
    )
    master.update(
        original_exact_numerical_module_sha256=master["exact_numerical_module_sha256"].copy(),
        exact_numerical_module_sha256={
            name: hashes[name] for name in master["exact_numerical_module_sha256"]
        },
        original_engineering_decision_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest(),
        alias_regression_receipt_sha256=hashlib.sha256(regression_text.encode()).hexdigest(),
        accepted_renderer_sha256=accepted,
        operational_successor_source_git=pin,
        numerical_optimizer_model_recipe_initialization_unchanged=True,
    )
    decision = json.dumps(master, sort_keys=True, indent=2) + "\n"
    payload = {
        "mode": "prepare",
        "root": REMOTE,
        "source_git": pin,
        "original_preparation": original,
        "alias_regression": json.loads(regression_text),
        "alias_regression_text": regression_text,
        "alias_regression_sha256": hashlib.sha256(regression_text.encode()).hexdigest(),
        "old_owner_stopped": json.loads(stopped_text),
        "old_owner_stopped_text": stopped_text,
        "old_owner_stopped_sha256": hashlib.sha256(stopped_text.encode()).hexdigest(),
        "files": files,
        "sha256": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()},
        "engineering_decision_text": decision,
        "engineering_decision_sha256": hashlib.sha256(decision.encode()).hexdigest(),
    }
    # Reuse paid original caches, never double-count them as still-unpaid. The
    # conservative62GiB reserve includes all three funded model scopes and older
    # obligations; 594MiB includes this successor, with 4GiB scratch retained.
    program = """
import hashlib,json,os,runpy,shutil,sys
from pathlib import Path
v=json.load(sys.stdin);os.umask(0o077)
raw=Path('/tmp/rsna-predictive-full-raw-cache-20261020-v1')
context=Path('/tmp/rsna-predictive-full-context-cache-20261020-v1')
paid_raw=json.loads(Path('/tmp/rsna-predictive-raw16-cache-20261020-v1/actual.json').read_text())['cache_bytes']+sum(p.stat().st_size for p in (raw/'cache').glob('*.npz'))
paid_context=json.loads(Path('/tmp/rsna-predictive-context16-cache-20261020-v1/actual.json').read_text())['cache_bytes']+sum(p.stat().st_size for p in context.glob('context-*.npz'))
v['funding']={'other_unpaid_node_liabilities_bytes':66*2**30+594*2**20+max(0,320*2**30-paid_raw)+max(0,96*2**30-paid_context),'shared_unpaid_bytes':121*2**20,'shared_unpaid_files':376}
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
