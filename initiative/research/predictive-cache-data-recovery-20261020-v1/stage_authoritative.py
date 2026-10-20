"""Stage the two existing authoritative files once on the owning compute node."""

import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REMOTE = "/tmp/rsna-predictive-authoritative-source-overlay-20261020-v1"
PROGRAM = r'''
import base64,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
v=json.load(sys.stdin)
assert os.uname().nodename=="evc33" and not os.environ.get("SLURM_JOB_GPUS")
os.umask(0o077)
root=Path(v["root"])
assert not root.exists(), "Existing intent requires read-only reconciliation"
payload={h:base64.b64decode(x["body"],validate=True) for h,x in v["files"].items()}
assert len(payload)==2 and sum(map(len,payload.values()))==1168268
assert all(hashlib.sha256(b).hexdigest()==h and len(b)==v["files"][h]["bytes"] for h,b in payload.items())
roots={role:[Path("/tmp/rsna-predictive-full-"+role+"-"+tail) for tail in
    ("cache-20261020-v1","alias-recovery-20261020-v3","capacity-recovery-20261020-v1","http-recovery-20261020-v1")]
    for role in ("raw","context")}
def physical(paths):
    seen=set(); total=0
    for path in paths:
        if not path.exists(): continue
        items=path.rglob("*") if path.is_dir() else (path,)
        for p in items:
            if p.is_file() and not p.is_symlink():
                s=p.stat(); key=(s.st_dev,s.st_ino)
                if key not in seen: total+=s.st_size; seen.add(key)
    return total
raw=physical([p for r in roots["raw"] for p in (r/"cache").glob("*.npz")]+list(Path("/tmp/rsna-predictive-raw16-cache-20261020-v1").rglob("*.npz")))
context=physical([p for r in roots["context"] for p in r.glob("context-*.npz")]+list(Path("/tmp/rsna-predictive-context16-cache-20261020-v1").glob("context-*.npz")))
scratch=physical([r/"scratch" for r in roots["raw"]])
# Capped funded cache totals include paid originals; no deletion credit.
future=max(0,320*2**30-raw)+max(0,96*2**30-context)+max(0,6*2**30-scratch)
future+=62*2**30+594*2**20+16*2**20
free=shutil.disk_usage("/tmp").free
assert free-future>=20*2**30, "Fresh node reserve unavailable"
quota=subprocess.check_output(["lfs","quota","-u",str(os.getuid()),"/home/sa305415"],text=True)
row=next(s.split() for s in quota.splitlines() if s.strip().startswith("/home/sa305415"))
shared=(int(row[2])-int(row[1].rstrip("*")))*1024
files=int(row[6])-int(row[5].rstrip("*"))
assert shared-123*2**20>=20*2**30 and files-376>=5000
root.mkdir(mode=0o700)
intent={k:v[k] for k in ("root","plan_sha256","receipt_sha256","regression_sha256")}
(root/"intent.json").write_text(json.dumps(intent,sort_keys=True))
refs={}
for h,b in payload.items():
    path=root/(h+".dcm")
    with path.open("xb") as f: f.write(b); f.flush(); os.fsync(f.fileno())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==h
    refs[h]={"path":str(path),"sha256":h,"bytes":len(b)}
result=dict(status="actual_existing_authoritative_source_bytes_staged_once",CPU_job=os.environ["SLURM_JOB_ID"],
    **intent,corrected_sources=refs,source_bytes=1168268,source_HTTP_requests=0,
    original_files_changed=0,shared_new_files=0,shared_future_bytes=123*2**20,shared_future_files=376,
    node_free_before=free,node_future_reserved=future,node_reserve_after=free-future,
    raw_paid_bytes=raw,context_paid_bytes=context,retained_source_bytes=scratch,
    shared_free_bytes=shared,shared_free_files=files,all_files_owner_only=True)
(root/"actual.json").write_text(json.dumps(result,sort_keys=True))
print(json.dumps(result),flush=True)
'''


def main():
    journal = ROOT / "artifacts" / "predictive-authoritative-source-overlay-20261020-v1"
    journal.mkdir(mode=0o700, exist_ok=True)
    intent = journal / "intent.json"
    assert not intent.exists(), "Read-only reconciliation required"
    audit = ROOT / "initiative/audits/source-pixel-representation-v1"
    plan = audit / "source_correction_plan.json"
    receipt = audit / "hf-source-correction-receipt.json"
    regression = ROOT / "initiative/research/predictive-source-decode-failure-20261020-v1/actual-authoritative-overlay-reproduction.json"
    def h(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    assert h(plan) == "68b44c1412ced183db223220f39cd4a6e071d40515df4c7b75b40a96ba6d0f3d"
    assert h(receipt) == "2c88c4f53cb3d2332e8079e1cd353eede5a601172febf0ab0b74ce8161f9fbb4"
    files = {}
    for change in json.loads(plan.read_text())["changes"]:
        info = change["after"]
        path = audit / "authoritative-kaggle" / Path(change["path"]).name
        body = path.read_bytes()
        assert h(path) == info["sha256"] and len(body) == info["bytes"]
        files[info["sha256"]] = {"bytes": len(body), "body": base64.b64encode(body).decode()}
    payload = dict(root=REMOTE, files=files, plan_sha256=h(plan),
                   receipt_sha256=h(receipt), regression_sha256=h(regression))
    module_path = ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_file_location("overlay_transport", module_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    command = ["srun", "--account=skattel", "--partition=normal", "--qos=normal",
               "--nodelist=evc33", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
               "--gres=none", "--mem=4G", "--time=00:05:00", "--immediate=60",
               "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
               "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python", "-c", transport.COMMAND]
    with intent.open("x") as f:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(), root=REMOTE,
                       program_sha256=hashlib.sha256(PROGRAM.encode()).hexdigest(),
                       plan_sha256=h(plan), receipt_sha256=h(receipt),
                       files={k: {"sha256": k, "bytes": x["bytes"]} for k, x in files.items()}), f)
    run = subprocess.run(
        ["ssh", "-S", "/tmp/rsna-research-20261001-stable.sock", "-o", "BatchMode=yes",
         "newton", shlex.join(command)], input=transport.envelope(PROGRAM, payload),
        text=True, capture_output=True, timeout=360)
    (journal / "stdout").write_text(run.stdout)
    (journal / "stderr").write_text(run.stderr)
    assert run.returncode == 0, "Inspect original intent/output read-only; never retry"
    result = json.loads(run.stdout.strip().splitlines()[-1])
    out = HERE / "actual-authoritative-source-resident.json"
    with out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
