"""One finite zero-GPU bridge; no credential files on Newton or automatic retry."""

import argparse
import importlib.util
import json
import os
import shlex
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JOURNAL = REPO / "artifacts" / HERE.name
ORIGINAL = REPO / "artifacts/colab-independent-C100-20261021-v1/bridge-v3"


def main(pin, socket):
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location("private_Colab_transport", HERE / "derive.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    for name in ("derive.py", "kaggle_io.py", "dispatch.py", "private-storage-observed.json"):
        path = HERE / name
        assert d.frozen(path.relative_to(REPO).as_posix(), pin) == path.read_text()
    assert json.loads((ORIGINAL / "exit.json").read_text())["returncode"] == 1
    rows = [json.loads(s) for s in (ORIGINAL / "stdout.jsonl").read_text().splitlines()]
    assert rows[-1]["http_status"] == 403
    assert sum(r.get("phase") == "private_input_archive_uploaded" for r in rows) == 7
    actual = subprocess.check_output(
        [
            "ssh",
            "-S",
            socket,
            "-o",
            "BatchMode=yes",
            "newton",
            "sacct -X -j 995557 --noheader --parsable2 --format=JobIDRaw,State,ExitCode",
        ],
        text=True,
    )
    assert actual.strip().rstrip("|") == "995557|FAILED|1:0"
    env = REPO / ".env"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO, check=True)
    values = {
        k.strip(): v.strip().strip("\"'")
        for line in env.read_text().splitlines()
        for k, sep, v in [line.partition("=")]
        if sep and not line.lstrip().startswith("#")
    }
    assert values["HF_TOKEN"].startswith("hf_") and values["KAGGLE_API_TOKEN"].startswith("KGAT_")
    controls = d.sources()
    controls["private-storage-observed.json"] = (HERE / "private-storage-observed.json").read_text()
    program = d.bridge()
    JOURNAL.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (JOURNAL / "dispatch-intent.json").open("x") as f:
        json.dump(
            {
                **d.source_receipt(),
                "control_git": pin,
                "created_UTC": datetime.now(timezone.utc).isoformat(),
                "CPUs": 2,
                "GPUs": 0,
                "node": "evc27",
                "original_failed_CPU": "995557",
                "reuses_original_80GiB_node_scope": True,
                "new_shared_files": 0,
                "Kaggle_private_transport_and_checkpoints_reserved_bytes": 80 * 2**30,
                "automatic_retry": False,
            },
            f,
        )
    bootstrap = """import hashlib,importlib.util,io,json,os,sys
from pathlib import Path
e=json.load(sys.stdin)
os.umask(0o077)
sys.path.append('/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/lib/python3.12/site-packages')
root=Path('/tmp/rsna-colab-private-kaggle-control-20261021-v1')
assert not root.exists(), 'Reconcile original control; no duplicate'
root.mkdir(mode=0o700)
for name,body in e['controls'].items():
    assert '/' not in name
    (root/name).write_text(body)
spec=importlib.util.spec_from_file_location('private_Colab_Kaggle_IO',root/'kaggle_io.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.stdin=io.StringIO(json.dumps(e['payload']))
exec(compile(e['program'],'original_archive_private_Kaggle_transport','exec'),
    {'__name__':'__main__','create_private_inputs':module.create_private_inputs})
"""
    remote = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc27",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--gres=none",
        "--mem=8G",
        "--time=02:00:00",
        "--immediate=60",
        "--job-name=rsna-colab-private-Kaggle",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=2",
        "MKL_NUM_THREADS=2",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        bootstrap,
    ]
    payload = {
        "source_git": d.SCIENCE,
        "bridge_control_git": pin,
        "token": values["HF_TOKEN"],
        "Kaggle_token": values["KAGGLE_API_TOKEN"],
        "use_private_Kaggle_storage": True,
        "own_files": d.original_files(),
    }
    with (JOURNAL / "stdout.jsonl").open("x") as out, (JOURNAL / "stderr.txt").open("x") as err:
        result = subprocess.run(
            ["ssh", "-S", socket, "-o", "BatchMode=yes", "newton", shlex.join(remote)],
            input=json.dumps({"program": program, "controls": controls, "payload": payload}),
            text=True,
            stdout=out,
            stderr=err,
            timeout=7500,
            check=False,
        )
    with (JOURNAL / "exit.json").open("x") as f:
        json.dump(
            {"returncode": result.returncode, "ended_UTC": datetime.now(timezone.utc).isoformat()},
            f,
        )
    if result.returncode:
        raise SystemExit("Inspect original job and account intent; no automatic retry")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-git", required=True)
    p.add_argument("--socket", required=True)
    args = p.parse_args()
    main(args.source_git, args.socket)
