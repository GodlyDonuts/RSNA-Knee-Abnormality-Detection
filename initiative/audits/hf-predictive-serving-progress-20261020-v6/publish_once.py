"""Explicit one-time publication of the reviewed v6 small-source manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exclusive(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--execute-reviewed-publication", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    prep = load(directory / "prepare.py", "v6_source_manifest")
    relative = directory.relative_to(root).as_posix()
    transport_name = "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    for name in (
        relative + "/prepare.py", relative + "/publish_once.py",
        relative + "/program.py.txt", transport_name,
    ):
        assert (root / name).read_bytes() == prep.git("show", args.source_git + ":" + name)
    files, expected = prep.select(args.source_git)
    manifest_body = (directory / "manifest.json").read_bytes()
    assert json.loads(manifest_body) == expected
    files["source-manifest.json"] = manifest_body
    assert len(files) < 200 and sum(map(len, files.values())) < 2**20
    env = root / ".env"
    assert env.stat().st_mode & 0o077 == 0
    token = next(
        value.strip().strip("\"'")
        for line in env.read_text().splitlines()
        for key, sep, value in [line.partition("=")]
        if sep and key.strip() == "HF_TOKEN"
    )
    assert token.startswith("hf_")
    journal = root / "artifacts/hf-predictive-serving-progress-20261020-v6"
    journal.mkdir(mode=0o700, exist_ok=True)
    assert not journal.is_symlink() and not journal.stat().st_mode & 0o077
    exclusive(journal / "intent.json", {
        "source_git": args.source_git, "previous_verified_source_git": expected["previous_verified_source_git"],
        "objects": len(files), "bytes": sum(map(len, files.values())),
        "manifest_sha256": prep.sha(manifest_body),
        "new_node_control_cap_bytes": 4 * 2**20,
        "canonical_shared_lock_required": True, "automatic_retry": False,
    })
    transport = load(root / transport_name, "v6_stdin_transport")
    program = (directory / "program.py.txt").read_text()
    program = (
        "import sys\ntry:\n"
        + "".join("    " + line + "\n" for line in program.splitlines())
        + "except BaseException as error:\n"
        + "    print(type(error).__name__,file=sys.stderr)\n    raise SystemExit(1)\n"
    )
    command = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc33",
        "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=2G", "--time=00:15:00",
        "--immediate=60", "--job-name=rsna-predictive-v6-source-milestone",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
        "OMP_NUM_THREADS=1", "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python",
        "-B", "-c", transport.COMMAND,
    ]
    result = subprocess.run(
        [
            "ssh", "-S", "/tmp/rsna-research-20261001-stable.sock",
            "-o", "BatchMode=yes", "newton", shlex.join(command),
        ],
        input=transport.envelope(program, {
            "source_git": args.source_git,
            "previous_verified_source_git": expected["previous_verified_source_git"],
            "files": {name: body.decode() for name, body in files.items()}, "token": token,
        }),
        capture_output=True, text=True, timeout=1200,
    )
    exclusive(journal / "transport.json", {
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    })
    assert result.returncode == 0, "Uncertain original publication; reconcile read-only, never retry"
    receipt = json.loads(result.stdout)
    assert receipt["source_git"] == args.source_git
    assert receipt["objects"] == len(files) and receipt["canonical_shared_publication_lock_used"]
    exclusive(directory / "actual-private-verified.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
