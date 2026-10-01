"""Inspect or dispatch each native384 prerequisite/production phase once.

No credential or dataset transfer is involved. Initial GPU dispatch requires a
terminal CPU byte preflight and a fresh CPU-only check on the actual cache node.
Uncertain scheduler mutations leave an intent and require read-only review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "initiative/audits/native384-production-preparation-v1"
LOCAL = ROOT / "artifacts/native384-production-v1"
HOST = "sa305415@newton.ist.ucf.edu"
CONTROL = "/home/sa305415/rsna-knee/production-launch-native384-a32-c9-v1"
ASSEMBLY = "/home/sa305415/rsna-knee/native384-assembly-c9-v1"
RUN = "/home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-native384-fold0-c9fcc7e-production-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    import os

    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def ssh(socket, args, *, payload=None, timeout=60):
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
            shlex.join(args),
        ],
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"native384_remote_phase_exit_{result.returncode}")
    return result.stdout.strip()


PROBE = r"""
import hashlib,json,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);control=Path(p['control']);assembly=Path(p['assembly'])
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
if not (control/'production-files.json').is_file():
 print(json.dumps({'status':'not_staged','gpu_dispatch_permitted':False}));raise SystemExit(0)
assert sha(control/'production-files.json')==p['manifest_sha256']
for name,row in p['files'].items():
 q=control/name;assert q.is_file() and not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
queue=subprocess.check_output(['squeue','-h','-u','sa305415','-o','%i|%T|%j|%b|%N'],text=True).splitlines()
gpu=[row for row in queue if 'gpu' in row.split('|')[3]]
state=subprocess.check_output(['sacct','-X','-j','866458','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).strip()
quota=subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'],text=True).splitlines()[-1].split()
used,soft,files,file_soft=[int(quota[i]) for i in (1,2,5,6)]
batch=Path('/home/sa305415/rsna-knee/native-full-cache-two-lanes-v1')
complete=sum((batch/f'shard-{n:03d}/result.json').is_file() for n in range(1,75))
remaining_files=(74-complete)*20+190
funded_bytes=5246914048+12*1024**2
gib=((soft-used)*1024-funded_bytes)/1024**3
remaining=file_soft-files-remaining_files
cpu=control/'cpu-preflight.json'
proof=json.loads(cpu.read_text()) if cpu.is_file() else None
cpu_state=subprocess.check_output(['sacct','-X','-j',proof['job_id'],'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).strip() if proof else None
result={'status':'ready_for_cpu_preflight','assembly_terminal':state,'cpu_terminal':cpu_state,'actual_gpu_jobs':gpu,'projected_remaining_gib':gib,'projected_remaining_files':remaining,'gpu_dispatch_permitted':False,'existing_cpu_intent':(control/'cpu-dispatch-intent.json').exists(),'existing_gpu_intent':(control/'gpu-dispatch-intent.json').exists(),'existing_run':Path(p['run']).exists(),'manifest_sha256':p['manifest_sha256']}
if state!='866458|COMPLETED|0:0' or not (assembly/'accepted.json').is_file():result['status']='waiting_for_exact_cpu_assembly'
elif gib<20 or remaining<5000:result['status']='waiting_for_funded_shared_reserve'
elif proof and cpu_state==proof['job_id']+'|COMPLETED|0:0':
 assert proof['status']=='full_native384_cpu_training_preflight_accepted' and proof['all_member_bytes_verified'] is True and proof['members']==37812 and proof['node']=='evc31'
 result.update(status='ready_for_live_cpu_recheck',cpu_preflight_sha256=sha(cpu))
 if len(gpu)>=2:result['status']='waiting_for_gpu_node_limit'
elif proof:result['status']='cpu_preflight_not_terminal_success'
print(json.dumps(result))
"""


def require_phase(state, mode):
    expected = "ready_for_cpu_preflight" if mode == "cpu" else "ready_for_live_cpu_recheck"
    if state.get("status") != expected:
        raise ValueError("native384_prerequisites_missing")
    if (
        state.get("assembly_terminal") != "866458|COMPLETED|0:0"
        or state.get("projected_remaining_gib", 0) < 20
        or state.get("projected_remaining_files", 0) < 5000
        or (mode == "gpu" and len(state.get("actual_gpu_jobs", [])) >= 2)
    ):
        raise ValueError("native384_actual_terminal_and_resource_prerequisites_required")
    if state.get("existing_run") or state.get("existing_" + mode + "_intent"):
        raise FileExistsError("native384_phase_already_attempted_reconcile_readonly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "cpu", "gpu"))
    parser.add_argument("--ssh-control", type=Path, required=True)
    args = parser.parse_args()
    if not args.ssh_control.is_socket():
        raise ValueError("live_ssh_control_socket_required")
    manifest_file = AUDIT / "production-files.json"
    manifest = json.loads(manifest_file.read_text())
    request = {
        "control": CONTROL,
        "assembly": ASSEMBLY,
        "run": RUN,
        "manifest_sha256": sha(manifest_file),
        "files": manifest["files"],
    }
    state = json.loads(
        ssh(args.ssh_control, ["/usr/bin/python3.12", "-B", "-c", PROBE], payload=request)
    )
    if args.mode == "check":
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    require_phase(state, args.mode)
    LOCAL.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = LOCAL / (args.mode + "-dispatch-intent.json")
    result = LOCAL / (args.mode + "-dispatch-result.json")
    if intent.exists() or result.exists():
        raise FileExistsError("native384_dispatch_already_attempted_reconcile_readonly")
    save_new(
        intent,
        {
            "status": "single_native384_phase_dispatch_intent",
            "mode": args.mode,
            "request": request,
            "prerequisites": state,
            "automatic_retry": False,
        },
    )
    if args.mode == "gpu":
        # Allocates only one CPU; no GPU queue entry exists during this check.
        command = [
            "srun",
            "--job-name=rsna-native384-live-cpu-recheck",
            "--account=skattel",
            "--partition=normal",
            "--nodes=1",
            "--nodelist=evc31",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=4G",
            "--time=00:30:00",
            "/usr/bin/python3.12",
            "-B",
            CONTROL + "/native384_admission.py",
            "check",
            "--assembly",
            ASSEMBLY,
            "--config",
            CONTROL + "/config.json",
            "--cpu-proof",
            CONTROL + "/cpu-preflight.json",
            "--deterministic-wrapper",
            CONTROL + "/enforce_deterministic_runtime.py",
        ]
        recheck = json.loads(ssh(args.ssh_control, command, timeout=3600))
        if (
            recheck.get("status") != "full_native384_cpu_prerequisites_still_valid"
            or recheck.get("cpu_preflight_sha256") != state["cpu_preflight_sha256"]
        ):
            raise ValueError("actual_native_cpu_recheck_required")
        state = json.loads(
            ssh(args.ssh_control, ["/usr/bin/python3.12", "-B", "-c", PROBE], payload=request)
        )
        require_phase(state, args.mode)
        acceptance = {
            "status": "native384_production_prerequisites_verified",
            "production_manifest_sha256": sha(manifest_file),
            "cpu_preflight_sha256": state["cpu_preflight_sha256"],
            "projected_remaining_gib": state["projected_remaining_gib"],
            "projected_remaining_files": state["projected_remaining_files"],
            "maximum_gpu_nodes_after_dispatch": len(state["actual_gpu_jobs"]) + 1,
            "actual_cpu_recheck": recheck,
        }
    else:
        acceptance = None
    mutation = r"""
import json,os,subprocess,sys
from pathlib import Path
r=json.load(sys.stdin);c=Path(r['control']);m=r['mode'];intent=c/(m+'-dispatch-intent.json')
if intent.exists():raise FileExistsError('existing_dispatch_requires_readonly_review')
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
save(intent,{'mode':m,'manifest_sha256':r['manifest_sha256'],'automatic_retry':False})
if r['acceptance'] is not None:save(c/'dispatch-acceptance.json',r['acceptance'])
output=subprocess.check_output(['sbatch','--parsable',str(c/('cpu-preflight.sbatch' if m=='cpu' else 'production.sbatch'))],text=True).strip()
if not output.split(';')[0].isdecimal():raise ValueError('uncertain_scheduler_result')
row={'status':'single_native384_phase_dispatched','mode':m,'job_id':output.split(';')[0],'manifest_sha256':r['manifest_sha256']}
save(c/(m+'-dispatch-result.json'),row);print(json.dumps(row))
"""
    outcome = json.loads(
        ssh(
            args.ssh_control,
            ["/usr/bin/python3.12", "-B", "-c", mutation],
            payload={
                "control": CONTROL,
                "mode": args.mode,
                "manifest_sha256": sha(manifest_file),
                "acceptance": acceptance,
            },
        )
    )
    if not re.fullmatch("[1-9][0-9]*", outcome.get("job_id", "")):
        raise ValueError("uncertain_native384_scheduler_result")
    save_new(result, outcome)
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
