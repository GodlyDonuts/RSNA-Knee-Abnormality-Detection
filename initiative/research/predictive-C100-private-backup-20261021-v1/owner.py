"""Finite continuation of the original fit with explicit private storage recovery."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JOURNAL = REPO / "artifacts" / HERE.name
SOURCE = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
END = 1792670400  # October22 noon UTC; same finite final-entry deadline.


def load(path):
    spec = importlib.util.spec_from_file_location("backup_original_C100_owner", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def git(pin, path):
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(REPO).as_posix()], cwd=REPO, text=True
    )


def token():
    p = REPO / ".env"
    assert p.stat().st_mode & 0o077 == 0
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO).returncode == 0
    values = {
        k.strip(): z.strip().strip("\"'")
        for line in p.read_text().splitlines()
        for k, s, z in [line.partition("=")]
        if s and not line.lstrip().startswith("#")
    }
    return values["KAGGLE_API_TOKEN"]


def once(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-git", required=True)
    p.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    for name in ("owner.py", "backup.py", "export_only.py"):
        assert git(args.source_git, HERE / name) == (HERE / name).read_text()
    parent = load(HERE.parent / "predictive-C100-preservation-recovery-20261021-v1/continue.py")
    c = parent.original()
    a = SimpleNamespace(source_git=SOURCE, socket=args.socket)
    frozen = c.frozen(a)
    if args.preflight_only:
        observed = c.call(a, frozen, "status")
        print(json.dumps(observed))
        return
    assert all(
        subprocess.run(["ps", "-p", str(pid), "-o", "pid="], capture_output=True).returncode != 0
        for pid in (67205, 76338, 86015)
    )
    JOURNAL.mkdir(mode=0o700, parents=True, exist_ok=True)
    once(
        JOURNAL / "owner-intent.json",
        {
            "source_git": args.source_git,
            "PID": os.getpid(),
            "original_GPU_job": "995351",
            "new_GPU_jobs": 0,
            "automatic_retry": False,
        },
    )

    def call(program, value, phase, python, mem):
        once(
            JOURNAL / (phase + "-intent.json"),
            {
                "source_git": args.source_git,
                "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                "automatic_retry": False,
            },
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
            "--mem=" + mem,
            "--time=00:40:00",
            "--job-name=rsna-C100-backup-" + phase,
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            python,
            "-B",
            "-c",
            frozen[2].COMMAND,
        ]
        r = subprocess.run(
            ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(cmd)],
            input=frozen[2].envelope(program, value),
            text=True,
            capture_output=True,
            timeout=2500,
        )
        once(
            JOURNAL / (phase + "-transport.private.json"),
            {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr},
        )
        assert r.returncode == 0, "Uncertain mutation stops; never retry"
        result = json.loads(r.stdout)
        once(JOURNAL / (phase + "-result.json"), result)
        return result

    try:
        while time.time() < END:
            observed = c.observe(a, frozen, END)
            assert observed is not None and observed["dispatch"]["GPU_job"] == "995351"
            if observed["full_scheduler"]["state"] in (
                "RUNNING",
                "PENDING",
                "COMPLETING",
                "CONFIGURING",
            ):
                time.sleep(min(120, max(0, END - time.time())))
                continue
            assert (
                observed["full_scheduler"]["state"] == "COMPLETED"
                and observed["full_scheduler"]["exit_code"] == "0:0"
            )
            assert observed["terminal"]["updates"] == 2916 and observed["ready_epochs"] == [
                1,
                2,
                3,
                4,
                5,
                6,
            ]
            break
        else:
            raise RuntimeError("Original completion deadline elapsed")
        adapter = (HERE / "export_only.py").read_text()
        operations = git(
            SOURCE, HERE.parent / "predictive-density100-full-20261020-v1/operations.py"
        )
        anchor = 'ns = runpy.run_path(str(publisher), run_name="distinct_C100_epoch_publisher")'
        assert operations.count(anchor) == 1
        operations = operations.replace(
            anchor, "ns = export_adapter.load(publisher, EXPORT_ADAPTER_SHA)"
        )
        admission = git(SOURCE, HERE.parent / "predictive-density100-full-20261020-v1/admission.py")
        program = (
            "import sys,json,types\n"
            + "a=types.ModuleType('admission');sys.modules['admission']=a\n"
            + f"exec(compile({admission!r},'original_admission','exec'),a.__dict__)\n"
            + "e=types.ModuleType('export_adapter')\n"
            + f"exec(compile({adapter!r},'export_adapter','exec'),e.__dict__)\n"
            + "o=types.ModuleType('operations');o.export_adapter=e\n"
            + f"o.EXPORT_ADAPTER_SHA={hashlib.sha256(adapter.encode()).hexdigest()!r}\n"
            + f"exec(compile({operations!r},'original_operations_transport_only','exec'),o.__dict__)\n"
            + "print(json.dumps(o.main(json.load(sys.stdin))))\n"
        )
        for epoch in (5, 6):
            assert str(epoch) not in observed["publications"], (
                "Existing export/publication requires reconciliation"
            )
            value = {
                **frozen[0],
                "phase": "publish",
                "epoch": epoch,
                "token": "",
                "original_dispatched_jobs": c.original_jobs(),
                "primary_control": c.a.primary_control(),
            }
            result = call(program, value, f"export-{epoch:03d}", c.a.PYTHON, "8G")
            assert result["status"] == "local_original_export_ready_for_private_backup"
        helpers = {
            n: git("fb9e3adf2d4f2f873adf10733bb310e7afa73be8", REPO / "scripts/kaggle" / n)
            for n in ("workflow.py", "components.py", "publication.py")
        }
        result = call(
            (HERE / "backup.py").read_text(),
            {"helpers": helpers, "source_git": args.source_git, "token": token()},
            "private-dataset",
            c.MANAGEMENT,
            "2G",
        )
        assert result["status"] == "actual_C100_epochs456_privately_download_SHA_verified"
        final = c.call(a, frozen, "status")
        assert len(final["publications"]) == 6 and all(
            x["status"].endswith("private_all_objects_download_size_SHA_verified")
            for x in final["publications"].values()
        )
        once(JOURNAL / "owner-result.json", final)
        print(
            json.dumps(
                {
                    "phase": "original_C100_all_epochs_preserved_and_final_ready",
                    "new_GPU_jobs": 0,
                    "Dataset": result["Dataset"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        once(
            JOURNAL / "stopped.json",
            {"error_type": type(error).__name__, "message": str(error), "automatic_retry": False},
        )
        raise


if __name__ == "__main__":
    os.umask(0o077)
    main()
