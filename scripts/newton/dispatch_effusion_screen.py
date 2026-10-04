"""Single coordinator: exact CPU admission, two isolated GPU jobs, private epochs."""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "artifacts/effusion-screen-evc32-v1"
CONTROL = "/tmp/rsna-effusion-screen-evc32-20261004-v1"
SOCKET = "/tmp/rsna-research-20261001-stable.sock"
BASE = "/home/sa305415/rsna-knee"
ROUTE = "/tmp/rsna-effusion-native-route-evc32-20261004-v1"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")


def launch_files():
    original = (
        ROOT
        / "initiative/audits/effusion-supervision-engineering-v1/recovery-v2/recovery.sbatch.txt"
    ).read_text()
    driver = original[
        original.index("probe_allocated_driver() {") : original.index(
            "if ! probe_allocated_driver;"
        )
    ]
    driver = driver.replace("--kill-after=5s 25s", "--kill-after=10s 180s")
    driver = driver.replace(
        "errors = []",
        "print(json.dumps(dict(record, stage='allocated_context')), flush=True)\nerrors = []",
    )
    driver = driver.replace(
        "result = driver.cuInit(0)",
        "print(json.dumps({'stage':'before_cuInit'}), flush=True)\n    result = driver.cuInit(0)\n    print(json.dumps({'stage':'after_cuInit','code':result}), flush=True)",
    )
    arm = "#!/bin/bash\nset -euo pipefail\numask 077\n" + driver
    arm += (
        'probe_allocated_driver\nexec "$1" -B '
        + CONTROL
        + '/scripts/newton/run_effusion_screen.py arm --role "$2"\n'
    )
    headers = original[original.index("CPATH=$(") : original.index("export CPATH")]
    headers = headers.replace(
        BASE + "/effusion-supervision-engineering-20261003-v1/recovery-v2", CONTROL
    )
    headers = "#!/bin/bash\nset -euo pipefail\numask 077\n" + headers + 'printf "%s\\n" "$CPATH"\n'
    files = {"arm.sh": arm.encode(), "headers.sh": headers.encode()}
    for name in (
        "scripts/newton/run_effusion_screen.py",
        "scripts/hf/publish_effusion_screen_epoch.py",
    ):
        files[name] = (ROOT / name).read_bytes()
    for role in ("control", "candidate"):
        files[role + "-screen.json"] = (
            ROOT
            / f"initiative/experiments/competitive-v1/effusion-review-contrast-v1/{role}-screen.json"
        ).read_bytes()
        script = f'''#!/bin/bash
#SBATCH --job-name=rsna-effusion-evc32-{role}
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output={CONTROL}/{role}.out
#SBATCH --error={CONTROL}/{role}.err
set -euo pipefail
umask 077
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH={CONTROL}/code
CPATH=$(bash {CONTROL}/headers.sh)
export CPATH
RSNA_SCREEN_PYTHON=$(/usr/bin/python3.12 -B -c 'import json; print(json.load(open("{CONTROL}/runtime.json"))["python"])')
export TRITON_CACHE_DIR=$(/usr/bin/python3.12 -B -c 'import json; print(json.load(open("{CONTROL}/runtime.json"))["runtime_root"] + "/triton-cache")')
exec bash {CONTROL}/arm.sh "$RSNA_SCREEN_PYTHON" {role}
'''
        files[role + ".sbatch"] = script.encode()
    return files


def remote(program, payload, *, cpu_name):
    command = [
        "salloc",
        "--quiet",
        "--account=skattel",
        "--partition=normal",
        "--nodelist=evc32",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=2G",
        "--time=00:15:00",
        "--job-name=" + cpu_name,
        "srun",
        "--cpu-bind=cores",
        "--unbuffered",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "/usr/bin/python3.12",
        "-B",
        "-c",
        program,
    ]
    return subprocess.run(
        ["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "newton", shlex.join(command)],
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=900,
    )


def dispatch():
    if (LOCAL / "dispatch-intent.json").exists():
        raise ValueError("Original dispatch exists; reconcile read-only, never retry")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Verified committed launch source required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = launch_files()
    proof = json.loads(
        (
            ROOT
            / "initiative/audits/effusion-supervision-engineering-v1/recovery-v2/screen-preparation-accepted.json"
        ).read_text()
    )
    files["source-provenance.json"] = (
        json.dumps(
            {
                **proof,
                "launch_git": head,
                "actual_complete_copy_job": "869745",
                "actual_assigned_driver_job": "869722",
                "independent_single_GPU_jobs": True,
                "competition_eligible": False,
                "failed_original_jobs_preserved": ["869702", "869714", "869718"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    program = """
import base64,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
os.umask(0o077);b=Path('/home/sa305415/rsna-knee');route=Path('/tmp/rsna-effusion-native-route-evc32-20261004-v1');c=Path('/tmp/rsna-effusion-screen-evc32-20261004-v1')
assert os.uname().nodename.split('.')[0]=='evc32' and os.environ['SLURM_CPUS_PER_TASK']=='1'
assert not os.environ.get('SLURM_JOB_GPUS') and not c.exists()
assert not any('gpu' in x.lower() for x in subprocess.check_output(['squeue','-u','sa305415','-h','-o','%b'],text=True).splitlines())
rows=subprocess.check_output(['sacct','-X','-j','869300,869745,869722,869718','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).splitlines()
assert set(rows)=={'869300|COMPLETED|0:0','869745|COMPLETED|0:0','869722|COMPLETED|0:0','869718|FAILED|1:0'}
copy_path=route/'copy-accepted.json';copy=json.loads(copy_path.read_text());assert copy['status']=='private_original_native384_copy_fully_byte_verified' and copy['job_id']=='869745' and copy['destination_node']=='evc32'
assert copy['members']==37812 and copy['bytes']==140626483351 and Path(copy['cache_root']).is_dir()
row=[x.split() for x in subprocess.check_output(['lfs','quota','-u','sa305415','/lustre/fs1'],text=True).splitlines() if x.lstrip().startswith('/lustre/fs1')][0]
used,soft,count,limit=[int(row[i]) for i in [1,2,5,6]];future=1404996457;nf=768
assert (soft-used)*1024-future>=20*2**30 and limit-count-nf>=5000
assert shutil.disk_usage('/tmp').free>=(20+7+3)*2**30+32*2**20
payload=json.load(sys.stdin);raw={k:base64.b64decode(v,validate=True) for k,v in payload['files'].items()}
assert set(raw)=={'arm.sh','headers.sh','scripts/newton/run_effusion_screen.py','scripts/hf/publish_effusion_screen_epoch.py','control-screen.json','candidate-screen.json','control.sbatch','candidate.sbatch','source-provenance.json'}
a=json.loads(raw['control-screen.json']);d=json.loads(raw['candidate-screen.json']);policy=d['fitting_supervision'];d['fitting_supervision']=a['fitting_supervision']
assert a==d and a['fitting_supervision']['enabled'] is False and policy['enabled'] is True and a['epochs']==3 and a['side']==384
c.mkdir(mode=0o700)
def save(n,v):
 with (c/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\\n');f.flush();os.fsync(f.fileno())
for n,data in raw.items():
 p=c/n;p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open('xb') as f:f.write(data)
m={'launch_git':payload['source_git'],'scientific_git':'665c0d515410df24f1f37ed5d9f8cb7a0f0fc480','copy_receipt_sha256':hashlib.sha256(copy_path.read_bytes()).hexdigest(),'files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in raw.items()}}
save('files.json',m)
f={'shared_used_kB':used,'shared_soft_kB':soft,'shared_used_files':count,'shared_file_soft':limit,'future_shared_bytes':future,'future_shared_files':nf,'after_liabilities_GiB':((soft-used)*1024-future)/2**30,'after_liabilities_files':limit-count-nf,'new_shared_bytes':0,'new_shared_files':0,'node_local_controls_MiB':32,'node_local_runtime_GiB':7,'node_local_checkpoints_GiB':3,'node_local_reserve_GiB':20,'independent_GPU_jobs':2,'GPU_nodes':1,'memory_GiB_per_arm':96,'original_assets_retained':True,'deletion_credit':0}
save('funding.json',f);save('prepare-intent.json',{'automatic_retry':False,'new_GPU_jobs':0})
subprocess.run(['/usr/bin/python3.12','-B',str(c/'scripts/newton/run_effusion_screen.py'),'prepare'],check=True)
save('dispatch-intent.json',{'automatic_retry':False,'launch_git':payload['source_git'],'independent_GPU_jobs':2});jobs={}
for role in ['control','candidate']:
 save(role+'-dispatch-intent.json',{'automatic_retry':False,'role':role})
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('SLURM_STEP_') or key.startswith('SLURM_CPU_BIND'):env.pop(key)
 job=subprocess.check_output(['sbatch','--parsable',str(c/(role+'.sbatch'))],env=env,text=True).strip();assert job.isdecimal();jobs[role]=job
 save(role+'-dispatch-result.json',{'job_id':job,'role':role,'automatic_retry':False})
r={'job_ids':jobs,'node':'evc32','control':str(c),'launch_git':payload['source_git'],'scientific_git':m['scientific_git'],'GPU_nodes':1,'GPUs_per_job':1,'CPUs_per_job':2,'RAM_GiB_per_job':96,'epochs_per_arm':3,'updates_per_arm':843,'training_studies':2247,'heldout_studies':1156,'automatic_retry':False,'staging_CPU_job':os.environ['SLURM_JOB_ID'],'clinical_result_observed':False}
save('dispatch-result.json',r);print(json.dumps({'dispatch':r,'funding':f,'source_manifest':m,'CPU_prepared':json.loads((c/'CPU-prepared.json').read_text())}),flush=True)
"""
    save(LOCAL / "dispatch-intent.json", {"source_git": head, "automatic_retry": False})
    result = remote(
        program,
        {"source_git": head, "files": {k: base64.b64encode(v).decode() for k, v in files.items()}},
        cpu_name="rsna-effusion-evc32-prepare",
    )
    (LOCAL / "dispatch.stdout").write_bytes(result.stdout)
    (LOCAL / "dispatch.stderr").write_bytes(result.stderr)
    if result.returncode:
        raise ValueError("Original state uncertain or failed; reconcile read-only")
    record = json.loads(result.stdout)
    save(LOCAL / "dispatch-result.json", record)
    audit = ROOT / "initiative/audits/effusion-supervision-screen-v1/evc32-independent-jobs"
    for name in ("dispatch", "funding", "source_manifest", "CPU_prepared"):
        save(audit / (name.replace("_", "-") + ".json"), record[name])
    for name in ("control.sbatch", "candidate.sbatch", "arm.sh", "headers.sh"):
        (audit / (name + ".txt")).write_bytes(files[name])
    print(json.dumps(record["dispatch"], sort_keys=True))


def publish(role, epoch):
    from scripts.hf.dispatch_native384_epoch import credential

    intent = LOCAL / "publications" / role / f"epoch-{epoch:03d}-intent.json"
    if intent.exists():
        raise ValueError("Original publication attempt exists; reconcile read-only")
    program = f"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
c=Path({CONTROL!r});r=json.loads((c/'runtime.json').read_text());m=json.loads((c/'files.json').read_text())
for n in ['scripts/newton/run_effusion_screen.py','scripts/hf/publish_effusion_screen_epoch.py']:
 assert hashlib.sha256((c/n).read_bytes()).hexdigest()==m['files'][n]['sha256']
assert (c/{role!r}/'epoch-checkpoints'/"epoch-{epoch:03d}.json").is_file()
assert not (c/'epoch-publication'/{role!r}/"epoch-{epoch:03d}"/'intent.json').exists()
client=c/'publication-client';client.mkdir(mode=0o700,exist_ok=True)
os.environ['HF_HOME']=str(client/'huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['PYTHONPATH']=str(c);os.environ['PYTHONNOUSERSITE']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
subprocess.run([r['python'],'-B',str(c/'scripts/hf/publish_effusion_screen_epoch.py'),'publish',{role!r},str({epoch})],input=sys.stdin.buffer.read(),check=True)
"""
    token = credential()
    save(intent, {"role": role, "epoch": epoch, "automatic_retry": False})
    result = remote(program, {"token": token}, cpu_name=f"rsna-effusion-{role}-e{epoch}-publish")
    output = intent.with_name(intent.name.replace("intent", "result"))
    if result.returncode:
        save(output, {"status": "uncertain_or_failed", "automatic_retry": False})
        raise ValueError("Private publication uncertain; inspect original journals")
    receipt = json.loads(result.stdout)
    save(output, receipt)
    save(
        ROOT
        / f"initiative/audits/effusion-supervision-screen-v1/{role}-epoch-{epoch:03d}-verified.json",
        receipt,
    )
    print(json.dumps(receipt, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dispatch", "publish"))
    parser.add_argument("--role", choices=("control", "candidate"))
    parser.add_argument("--epoch", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.phase == "dispatch":
        dispatch()
    elif args.role and args.epoch:
        publish(args.role, args.epoch)
    else:
        raise ValueError("Exactly one completed role/epoch required")


if __name__ == "__main__":
    main()
