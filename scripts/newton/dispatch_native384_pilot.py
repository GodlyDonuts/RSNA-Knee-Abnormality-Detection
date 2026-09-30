"""Check or dispatch the staged native384 pilot once, without network credentials."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "artifacts/native384-gpu-pilot-preparation-v1"
HOST = "sa305415@newton.ist.ucf.edu"
CONTROL = "/home/sa305415/rsna-knee/native384-recovery-c9-pilot-v1"
CPU_PYTHON = "/usr/bin/python3.12"


def now():
    return datetime.now(timezone.utc).isoformat()


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def ssh(socket, args, request=None, timeout=45):
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
        input=json.dumps(request) if request is not None else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"Remote phase failed ({result.returncode}): {result.stderr[-1800:]}")
    return json.loads(result.stdout)


CHECK = r"""
import hashlib,json,subprocess,sys,time
from pathlib import Path
r=json.load(sys.stdin);p=r['plan'];c=Path(p['control']);assert str(c)=='/home/sa305415/rsna-knee/native384-recovery-c9-pilot-v1'
ready=json.loads((c/'cpu-stage-verified.json').read_text());assert ready['status']=='private_native_pilot_payloads_cpu_staged_fully_verified'
assert ready['files']==p['files'] and ready['private_publication']==p['pinned_private_cache_publication']
account=subprocess.check_output(['sacct','-X','--noheader','-j',ready['job_id'],'--format=JobIDRaw,State,ExitCode','--parsable2'],text=True)
assert account.strip()==ready['job_id']+'|COMPLETED|0:0'
for n,x in p['files'].items():
 q=c/n;assert not q.is_symlink() and q.stat().st_size==x['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256']
for n,x in p['pinned_private_cache_publication']['files'].items():
 q=c/'payloads'/n;assert not q.is_symlink() and q.stat().st_size==x['bytes']
rows=subprocess.check_output(['squeue','--noheader','-u','sa305415','-o','%i|%T|%j|%b|%N'],text=True).splitlines()
gpu=[line for line in rows if 'gpu' in line.split('|')[3]]
attempted=(c/'gpu-dispatch-intent.json').exists() or (c/'gpu-dispatch-result.json').exists()
same=[line for line in rows if line.split('|')[2]=='rsna-native384-recovery-pilot']
status='already_attempted_reconcile_read_only' if attempted or same else ('waiting_for_gpu_role' if len(gpu)>1 else 'ready_for_cpu_prerequisites')
print(json.dumps({'status':status,'checked_at_unix':time.time(),'actual_gpu_jobs':gpu,'matching_pilot_jobs':same,'private_cpu_stage_terminal_verified':True,'prepared_small_file_hashes_verified':len(p['files']),'gpu_reserved':False}))
"""

CPU_ACCEPT = r"""
import hashlib,io,json,os,subprocess,sys,time
from pathlib import Path
r=json.load(sys.stdin);p=r['plan'];c=Path(p['control']);assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_JOB_GPUS')
assert not (c/'gpu-prerequisites-intent.json').exists()
def save(n,v):
 with (c/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
save('gpu-prerequisites-intent.json',{'job_id':os.environ['SLURM_JOB_ID'],'created_unix':time.time(),'gpu_reserved':False,'one_cpu_phase_only':True})
ns={'__name__':'native_gpu_cpu_admission'};exec(compile(r['admission_source'],'native_gpu_cpu_admission','exec'),ns)
sys.stdin=io.StringIO(json.dumps({**p,'phase':'staged_payloads'}));old=sys.stdout;sys.stdout=io.StringIO()
try:ns['main']();admission=json.loads(sys.stdout.getvalue())
finally:sys.stdout=old
assert admission['gpu_dispatch_admitted'] is True
code=Path('/home/sa305415/rsna-knee/code-delta-recovery-c9fcc7e')
for n,h in r['scientific_files'].items():assert ns['digest'](code/n)==h
selection=json.loads((c/'selection.json').read_text())
for n,x in selection['input_bindings'].items():
 q=Path('/home/sa305415/rsna-knee/code-stopfix-1f486765')/n;assert q.stat().st_size==x['bytes'] and ns['digest'](q)==x['sha256']
assets={str(c/'payloads'/n):x for n,x in p['pinned_private_cache_publication']['files'].items()}
encoder=json.loads((c/'pilot-config.json').read_text())['encoder']
assets[encoder['weights_path']]={'bytes':1212559808,'sha256':encoder['weights_sha256']}
assets[encoder['config_path']]={'bytes':Path(encoder['config_path']).stat().st_size,'sha256':encoder['config_sha256']}
assets['/home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar']={'sha256':'89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338'}
def signature(q):
 s=q.stat();assert not q.is_symlink() and q.is_file()
 return {'dev':s.st_dev,'ino':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
verified={}
for n,x in assets.items():
 q=Path(n);before=signature(q);assert x.get('bytes',before['bytes'])==before['bytes']
 assert ns['digest'](q)==x['sha256'] and signature(q)==before
 verified[n]={'sha256':x['sha256'],'stat':before}
save('gpu-prerequisites-verified.json',{'status':'native_gpu_prerequisites_cpu_hash_verified','job_id':os.environ['SLURM_JOB_ID'],'verified_unix':time.time(),'assets':verified,'scientific_files':r['scientific_files'],'accepted_input_bindings':selection['input_bindings'],'plan_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'admission_source_sha256':hashlib.sha256(r['admission_source'].encode()).hexdigest(),'admission':admission,'gpu_reserved':False})
print(json.dumps({'status':'native_gpu_prerequisites_cpu_hash_verified','job_id':os.environ['SLURM_JOB_ID'],'assets_verified':len(verified),'gpu_reserved':False}))
"""

DISPATCH = r"""
import hashlib,io,json,os,subprocess,sys,time
from pathlib import Path
r=json.load(sys.stdin);p=r['plan'];c=Path(p['control'])
assert not (c/'gpu-dispatch-intent.json').exists() and not (c/'gpu-dispatch-result.json').exists()
proof=json.loads((c/'gpu-prerequisites-verified.json').read_text());assert proof['status']=='native_gpu_prerequisites_cpu_hash_verified'
assert proof['plan_sha256']==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert proof['admission_source_sha256']==hashlib.sha256(r['admission_source'].encode()).hexdigest()
assert proof['scientific_files']==r['scientific_files']
selection=json.loads((c/'selection.json').read_text());assert proof['accepted_input_bindings']==selection['input_bindings']
expected={str(c/'payloads'/n):x['sha256'] for n,x in p['pinned_private_cache_publication']['files'].items()}
encoder=json.loads((c/'pilot-config.json').read_text())['encoder']
expected.update({encoder['weights_path']:encoder['weights_sha256'],encoder['config_path']:encoder['config_sha256'],'/home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar':'89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338'})
assert set(proof['assets'])==set(expected)
for n,h in expected.items():assert proof['assets'][n]['sha256']==h
account=subprocess.check_output(['sacct','-X','--noheader','-j',proof['job_id'],'--format=JobIDRaw,State,ExitCode','--parsable2'],text=True)
assert account.strip()==proof['job_id']+'|COMPLETED|0:0'
def signature(q):
 s=q.stat();assert not q.is_symlink() and q.is_file()
 return {'dev':s.st_dev,'ino':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
for n,x in proof['assets'].items():assert signature(Path(n))==x['stat']
for n,x in p['files'].items():
 q=c/n;assert not q.is_symlink() and q.is_file() and q.stat().st_size==x['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256']
ns={'__name__':'fresh_native_gpu_admission'};exec(compile(r['admission_source'],'fresh_native_gpu_admission','exec'),ns)
original=ns['digest']
def verified_digest(path):
 if str(path) in proof['assets']:
  x=proof['assets'][str(path)];assert signature(path)==x['stat'];return x['sha256']
 return original(path)
ns['digest']=verified_digest
sys.stdin=io.StringIO(json.dumps({**p,'phase':'staged_payloads'}));old=sys.stdout;sys.stdout=io.StringIO()
try:ns['main']();admission=json.loads(sys.stdout.getvalue())
finally:sys.stdout=old
assert admission['gpu_dispatch_admitted'] is True
code=Path('/home/sa305415/rsna-knee/code-delta-recovery-c9fcc7e')
for n,h in proof['scientific_files'].items():assert original(code/n)==h
for n,x in proof['accepted_input_bindings'].items():
 q=Path('/home/sa305415/rsna-knee/code-stopfix-1f486765')/n;assert q.stat().st_size==x['bytes'] and original(q)==x['sha256']
subprocess.run(['bash','-n',str(c/'pilot.sbatch')],check=True)
def save(n,v):
 with (c/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
save('gpu-dispatch-intent.json',{'created_unix':time.time(),'script_sha256':p['files']['pilot.sbatch']['sha256'],'cpu_prerequisite_job':proof['job_id'],'admission':admission,'assets_unchanged_since_full_cpu_hash_proof':True,'gpus':1,'cpus':2,'automatic_retry':False})
out=subprocess.run(['sbatch','--parsable',str(c/'pilot.sbatch')],text=True,capture_output=True,timeout=40)
record={'returncode':out.returncode,'stdout':out.stdout,'stderr':out.stderr,'dispatched_unix':time.time()}
if out.returncode==0:
 job=out.stdout.strip().split(';')[0];assert job.isdecimal();record['job_id']=int(job)
save('gpu-dispatch-result.json',record);print(json.dumps(record));assert out.returncode==0
"""


def request_data():
    plan = json.loads((FOLDER / "prepared.json").read_text())
    assert plan["control"] == CONTROL
    for name, row in plan["files"].items():
        data = (FOLDER / name).read_bytes()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
    scientific = {}
    names = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "rsna").rglob("*.py"))]
    names += [
        "scripts/newton/" + n
        for n in (
            "runtime_archive.py",
            "runtime_environment.py",
            "preflight.py",
            "supervise.py",
            "verify_adapter_pilot.py",
        )
    ]
    for name in names:
        data = subprocess.check_output(
            ["git", "show", plan["scientific_runtime_git"] + ":" + name], cwd=ROOT
        )
        scientific[name] = hashlib.sha256(data).hexdigest()
    return {
        "plan": plan,
        "admission_source": (FOLDER / "admission.py").read_text(),
        "scientific_files": scientific,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-control", type=Path, required=True)
    parser.add_argument(
        "--dispatch", action="store_true", help="Submit only after all actual gates pass"
    )
    args = parser.parse_args()
    assert args.ssh_control.is_socket()
    request = request_data()
    checked = ssh(args.ssh_control, [CPU_PYTHON, "-c", CHECK], request)
    print(json.dumps(checked), flush=True)
    if not args.dispatch or checked["status"] != "ready_for_cpu_prerequisites":
        return
    cpu_intent = FOLDER / "gpu-prerequisites-local-intent.json"
    if not cpu_intent.exists():
        save_new(
            cpu_intent,
            {
                "created_at": now(),
                "requested_cpus": 1,
                "requested_gpus": 0,
                "one_attempt_only": True,
            },
        )
        remote = [
            "srun",
            "--account=skattel",
            "--partition=normal",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=8G",
            "--time=00:15:00",
            CPU_PYTHON,
            "-c",
            CPU_ACCEPT,
        ]
        cpu = ssh(args.ssh_control, remote, request, timeout=1200)
        save_new(FOLDER / "gpu-prerequisites-local-result.json", cpu)
        print(json.dumps(cpu), flush=True)
    else:
        # Never redo an attempted CPU allocation. DISPATCH checks its exact
        # remote verified receipt, actual terminal state and every signature.
        print(json.dumps({"status": "using_existing_cpu_phase_read_only_no_repeat"}), flush=True)
    checked = ssh(args.ssh_control, [CPU_PYTHON, "-c", CHECK], request)
    if checked["status"] != "ready_for_cpu_prerequisites":
        print(json.dumps(checked), flush=True)
        return
    intent = FOLDER / "gpu-dispatch-local-intent.json"
    assert not intent.exists(), "GPU dispatch already attempted; reconcile read-only"
    save_new(
        intent,
        {
            "created_at": now(),
            "control": CONTROL,
            "one_attempt_only": True,
            "script_sha256": request["plan"]["files"]["pilot.sbatch"]["sha256"],
        },
    )
    result = ssh(args.ssh_control, [CPU_PYTHON, "-c", DISPATCH], request, timeout=70)
    save_new(FOLDER / "gpu-dispatch-local-result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
