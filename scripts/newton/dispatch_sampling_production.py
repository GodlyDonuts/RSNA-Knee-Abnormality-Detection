"""Inspect, stage, or dispatch each accepted full-cohort sampling arm exactly once."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "initiative/audits/stratified-slices-production-launch-v1"
CONTROL = "/home/sa305415/rsna-knee/sampling-production-7ac7a97-v1"
HOST = "sa305415@newton.ist.ucf.edu"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def ssh(socket, code, payload):
    result = subprocess.run(
        [
            "ssh",
            "-S",
            str(socket),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            HOST,
            shlex.join(["/usr/bin/python3.12", "-B", "-c", code]),
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(
            f"Sampling remote phase failed ({result.returncode}); reconcile before retry"
        )
    return json.loads(result.stdout)


PROBE = r"""
import hashlib,json,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);c=Path(p['control']);m=p['manifest'];b=Path('/home/sa305415/rsna-knee')
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
def terminal(j):return subprocess.check_output(['sacct','-X','--noheader','-j',j,'--format=JobIDRaw,State,ExitCode','--parsable2'],text=True).strip()
assert terminal('866624')=='866624|COMPLETED|0:0' and terminal('866694')=='866694|COMPLETED|0:0'
pilot=b/'stratified-pilot-7ac7a97-v1';prep=b/'stratified-production-7ac7a97-preparation-v1'
assert sha(prep/'cpu-prerequisites-verified.json')==m['CPU_proof_sha256'];assert sha(pilot/'terminal-verified.json')==m['pilot_proof_sha256']
proof=json.loads((prep/'cpu-prerequisites-verified.json').read_text());assert proof['status']=='full_cohort_source_assets_and_sampling_CPU_preparation_verified' and proof['data']['full_cohort_metadata']['studies']==3403 and proof['data']['full_cohort_metadata']['series']==18906
preserved=json.loads((pilot/'private-preservation/verified.json').read_text());assert preserved['status']=='private_full_download_size_sha_verified' and preserved['revision']=='906732feec5fe3b67e61b94cec6ebce7d3dd4bec' and preserved['file_count']==26
for path,row in proof['assets'].items():
 q=Path(path);assert not q.is_symlink() and q.is_file();s=q.stat();assert {'dev':s.st_dev,'ino':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}==row['stat']
queue=subprocess.check_output(['squeue','-h','-u','sa305415','-o','%i|%T|%j|%b|%N'],text=True).splitlines();gpu=[r for r in queue if 'gpu' in r.split('|')[3]]
q=subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'],text=True).splitlines()[-1].split();used,soft,n,limit=[int(q[i]) for i in (1,2,5,6)]
lanes=b/'native-full-cache-two-lanes-v1';done=sum((lanes/f'shard-{i:03d}/result.json').is_file() for i in range(1,75));native_files=(74-done)*20+190
charged={}
for run in m['runs'].values():
 r=Path(run)
 for f in [r/'checkpoint.pt',*list((r/'epoch-checkpoints').glob('*.pt'))]:
  if f.is_file():
   assert not f.is_symlink();s=f.stat();assert s.st_size<128*1024**2;charged[(s.st_dev,s.st_ino)]=s.st_size
future_bytes=5246914048+12*1024**2+22*128*1024**2+16*1024**2-sum(charged.values())
reserve=((soft-used)*1024-future_bytes)/1024**3;files=limit-n-native_files-400
staged=(c/'files.json').is_file()
if staged:
 assert sha(c/'files.json')==p['manifest_sha256']
 for name,row in m['files'].items():
  f=c/name;assert not f.is_symlink() and f.is_file() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
roles={k:{'output_exists':Path(v).exists(),'intent_exists':(c/(k+'-dispatch-intent.json')).exists(),'result':json.loads((c/(k+'-dispatch-result.json')).read_text()) if (c/(k+'-dispatch-result.json')).is_file() else None} for k,v in m['runs'].items()}
print(json.dumps({'status':'accepted_inputs_and_funded_pair' if reserve>=20 and files>=5000 else 'waiting_for_funded_pair','staged':staged,'projected_remaining_gib':reserve,'projected_remaining_files':files,'actual_gpu_jobs':gpu,'roles':roles,'completed_native_future_shards':done,'full_sampling_runs_launched':sum(r['result'] is not None for r in roles.values())}))
"""


def require_dispatch(state, policy):
    if state.get("status") != "accepted_inputs_and_funded_pair" or not state.get("staged"):
        raise ValueError("Actual accepted source/cache/pilot and storage funding are required")
    if (
        state["projected_remaining_gib"] < 20
        or state["projected_remaining_files"] < 5000
        or len(state["actual_gpu_jobs"]) >= 2
    ):
        raise ValueError("Live byte/file reserve and at most two GPU nodes are required")
    role = state["roles"][policy]
    if role["output_exists"] or role["intent_exists"] or role["result"] is not None:
        raise FileExistsError("Role already attempted; reconcile scheduler and outputs read-only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "stage", "uniform", "stratified"))
    parser.add_argument("--ssh-control", type=Path, required=True)
    parser.add_argument("--revision", choices=("original", "uniform-evc31-v2"), default="original")
    args = parser.parse_args()
    audit = AUDIT if args.revision == "original" else AUDIT / "uniform-evc31-recovery-v2"
    control = (
        CONTROL
        if args.revision == "original"
        else "/home/sa305415/rsna-knee/sampling-production-7ac7a97-uniform-evc31-v2"
    )
    if args.revision != "original" and args.mode == "stratified":
        raise ValueError("The unchanged stratified role already exists in the original control")
    if not args.ssh_control.is_socket():
        raise ValueError("Live SSH control socket required")
    manifest_path = audit / "files.json"
    manifest = json.loads(manifest_path.read_text())
    request = dict(control=control, manifest=manifest, manifest_sha256=sha(manifest_path))
    state = ssh(args.ssh_control, PROBE, request)
    if args.mode == "check":
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    if args.mode == "stage":
        if state["staged"] or state["status"] != "accepted_inputs_and_funded_pair":
            raise ValueError("Stage requires accepted/funded inputs and an unused unique control")
        entries = {}
        for name, row in manifest["files"].items():
            path = ROOT / row["local_path"]
            if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Local production source changed")
            entries[name] = base64.b64encode(path.read_bytes()).decode()
        entries["files.json"] = base64.b64encode(manifest_path.read_bytes()).decode()
        save(
            audit / "stage-intent.json",
            dict(request=request, accepted_state=state, automatic_retry=False),
        )
        code = r"""
import base64,hashlib,json,os,sys
from pathlib import Path
r=json.load(sys.stdin);c=Path(r['control']);assert not c.exists();c.mkdir(mode=0o700)
for name,data in r['entries'].items():
 assert Path(name).name==name;raw=base64.b64decode(data)
 if name!='files.json':assert hashlib.sha256(raw).hexdigest()==r['manifest']['files'][name]['sha256']
 with (c/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'status':'sampling_production_payload_staged_not_dispatched','manifest_sha256':hashlib.sha256((c/'files.json').read_bytes()).hexdigest(),'files':len(r['entries'])}))
"""
        result = ssh(args.ssh_control, code, {**request, "entries": entries})
        save(audit / "stage-verified.json", result)
    else:
        require_dispatch(state, args.mode)
        save(
            audit / (args.mode + "-dispatch-intent.json"),
            dict(request=request, policy=args.mode, state=state, automatic_retry=False),
        )
        # Atomic local coordinator claim prevents another arm from racing the live checks.
        code = r"""
import json,os,subprocess,sys
from pathlib import Path
r=json.load(sys.stdin);c=Path(r['control']);role=r['policy'];lock=c/'dispatch.lock';lock.mkdir(mode=0o700)
try:
 assert not (c/(role+'-dispatch-intent.json')).exists() and not Path(r['manifest']['runs'][role]).exists()
 # Re-run the exact read-only acceptance immediately before scheduler mutation.
 original=sys.stdin;sys.stdin=__import__('io').StringIO(json.dumps(r));capture=__import__('io').StringIO()
 with __import__('contextlib').redirect_stdout(capture):exec(compile(r['probe'],'live_sampling_acceptance','exec'),{'__name__':'__main__'})
 live=json.loads(capture.getvalue());assert live['status']=='accepted_inputs_and_funded_pair' and len(live['actual_gpu_jobs'])<2 and live['staged']
 def save(name,v):
  with (c/name).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 save(role+'-dispatch-intent.json',{'policy':role,'manifest_sha256':r['manifest_sha256'],'live_acceptance':live,'automatic_retry':False})
 response=subprocess.check_output(['sbatch','--parsable',str(c/(role+'.sbatch'))],text=True).strip();job=response.split(';')[0];assert job.isdecimal()
 result={'status':'single_full_sampling_arm_submitted','policy':role,'job_id':job,'run':r['manifest']['runs'][role],'manifest_sha256':r['manifest_sha256'],'CPU':2,'GPU':1}
 save(role+'-dispatch-result.json',result);print(json.dumps(result))
finally:lock.rmdir()
"""
        result = ssh(args.ssh_control, code, {**request, "policy": args.mode, "probe": PROBE})
        save(audit / (args.mode + "-dispatch-result.json"), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
