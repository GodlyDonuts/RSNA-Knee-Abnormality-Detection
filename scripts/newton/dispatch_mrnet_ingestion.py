"""Single private CPU-only Stanford acquisition, credential streamed over stdin."""

from __future__ import annotations

import argparse
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

from scripts.newton.ingest_mrnet import admit_sources, save_new, sha

ROOT = Path(__file__).resolve().parents[2]
HOST = "sa305415@newton.ist.ucf.edu"
CONTROL = "/home/sa305415/rsna-knee/mrnet-original-research-ingestion-20261001-v1"
LOCAL = ROOT / "artifacts/external-knee-research-20261001-v1/mrnet-dispatch"
SOURCE = ROOT / "artifacts/external-knee-research-20261001-v1/mrnet-original-api"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-control", type=Path, required=True)
    args = parser.parse_args()
    if not args.ssh_control.is_socket():
        raise ValueError("live_SSH_control_required")
    admit_sources(SOURCE)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scripts = [
        ROOT / "scripts/newton" / name
        for name in ("ingest_mrnet.py", "dispatch_mrnet_ingestion.py")
    ]
    for path in scripts:
        committed = subprocess.check_output(
            ["git", "show", revision + ":" + str(path.relative_to(ROOT))], cwd=ROOT
        )
        if committed != path.read_bytes():
            raise ValueError("committed_immutable_ingestion_source_required")
    env = ROOT / ".env"
    if env.stat().st_mode & 0o077:
        raise ValueError("private_local_credentials_required")
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    token = values.get("REDIVIS_ACCESS_TOKEN", "")
    if not token or "\n" in token:
        raise ValueError("read_only_Redivis_token_required")
    prefix = [
        "ssh",
        "-S",
        str(args.ssh_control),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        HOST,
    ]
    probe = r"""
import json,subprocess,sys
from pathlib import Path
c=Path(sys.argv[1])
q=subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'],text=True).splitlines()[-1].split()
used,soft,files,limit=[int(q[i]) for i in (1,2,5,6)]
# Native outstanding checkpoints, conservative sampling atomic/metadata remnants,
# and this entire small-source/control liability. No original deletion credit.
liability=5246914048+12*1024**2+272*1024**2+64*1024**2
gib=((soft-used)*1024-liability)/1024**3
remaining=limit-files-190-400-160
assert gib>=20 and remaining>=5000,'funded_shared_reserve_required'
assert not c.exists(),'existing_MRNet_control_requires_readonly_reconciliation'
print(json.dumps({'projected_shared_reserve_gib':gib,'projected_shared_file_reserve':remaining,'node_local_raw_bytes_funded':7740871424,'node_local_reserve_gib':20,'shared_small_source_cap_bytes':64*1024**2,'shared_small_source_cap_files':160,'GPUs':0,'CPUs':1}))
"""
    precheck = subprocess.run(
        prefix + [shlex.join(["/usr/bin/python3.12", "-B", "-c", probe, CONTROL])],
        capture_output=True,
        timeout=60,
    )
    if precheck.returncode:
        raise RuntimeError("MRNet_readonly_prerequisite_check_failed")
    funding = json.loads(precheck.stdout)
    LOCAL.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = LOCAL / "dispatch-intent.json"
    if intent.exists():
        raise FileExistsError("existing_MRNet_intent_requires_readonly_reconciliation")
    files = {"ingest_mrnet.py": scripts[0]}
    for name in (
        "downloads.json",
        "train.csv",
        "valid.csv",
        *(
            f"{split}-{target}.csv"
            for split in ("train", "valid")
            for target in ("acl", "meniscus", "abnormal")
        ),
    ):
        files["original-api/" + name] = SOURCE / name
    manifest = {
        "source_git": revision,
        "source_manifest_sha256": sha(SOURCE / "downloads.json"),
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha(path)}
            for name, path in files.items()
        },
        "funding": funding,
        "scope": "verbatim_private_noncommercial_research_acquisition_only",
        "model_fitting": False,
        "raw_redistribution": False,
    }
    save_new(
        intent,
        {
            "status": "single_private_MRNet_ingestion_intent",
            "control": CONTROL,
            "source": manifest,
            "automatic_retry": False,
        },
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
        raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("source.json")
        info.size, info.mode = len(raw), 0o600
        archive.addfile(info, io.BytesIO(raw))
    stage = r"""
import hashlib,json,os,sys,tarfile
from pathlib import Path
os.umask(0o077);c=Path(sys.argv[1]);c.mkdir(mode=0o700)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for m in archive:
  p=Path(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  target=c/p;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(archive.extractfile(m).read())
manifest=json.loads((c/'source.json').read_text())
for name,row in manifest['files'].items():
 p=c/name;assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print('private_original_source_stage_verified')
"""
    result = subprocess.run(
        prefix + [shlex.join(["/usr/bin/python3.12", "-B", "-c", stage, CONTROL])],
        input=buffer.getvalue(),
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError("MRNet_stage_uncertain_reconcile_readonly")
    save_new(
        LOCAL / "stage-result.json",
        {"status": "private_original_source_stage_verified", "source": manifest},
    )
    command = [
        "srun",
        "--job-name=rsna-MRNet-verbatim-private",
        "--account=skattel",
        "--partition=normal",
        "--nodes=1",
        "--nodelist=evc32",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=02:00:00",
        "/usr/bin/python3.12",
        "-B",
        CONTROL + "/ingest_mrnet.py",
        "--source-directory",
        CONTROL + "/original-api",
        "--manifest-sha256",
        manifest["source_manifest_sha256"],
        "--control",
        CONTROL,
    ]
    # Srun forwards stdin to the sole task; no secret appears in argv or files.
    process = subprocess.Popen(
        prefix + [shlex.join(command)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write((token + "\n").encode())
    process.stdin.close()
    token, values = "", {}
    assert process.stdout is not None
    with (LOCAL / "progress.jsonl").open("x") as log:
        for line in process.stdout:
            message = line.decode()
            log.write(message)
            log.flush()
            print(message, end="", flush=True)
    code = process.wait()
    save_new(
        LOCAL / "process-result.json",
        {"exit_code": code, "automatic_retry": False, "control": CONTROL},
    )
    if code:
        raise SystemExit("MRNet_ingestion_nonzero_reconcile_private_receipts")


if __name__ == "__main__":
    main()
