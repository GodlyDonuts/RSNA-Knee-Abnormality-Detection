"""Single coordinator for the admitted parallel screen and its private CPU publications."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "artifacts/effusion-screen-v2"
CONTROL = "/tmp/rsna-effusion-screen-20261004-v2"
SOCKET = "/tmp/rsna-research-20261001-stable.sock"
BASE = "/home/sa305415/rsna-knee"
ROUTE = BASE + "/effusion-native-cache-route-20261004-v1/recovery-v2"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


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
        "count = len([line",
        "print(json.dumps(dict(record, stage='NVML_returned')), flush=True)\n    count = len([line",
    )
    driver = driver.replace(
        "result = driver.cuInit(0)",
        "print(json.dumps({'stage':'before_cuInit'}), flush=True)\n    result = driver.cuInit(0)\n    print(json.dumps({'stage':'after_cuInit', 'code':result}), flush=True)",
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
    script = f"""#!/bin/bash
#SBATCH --job-name=rsna-effusion-screen-v2
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc34
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output={CONTROL}/allocation-%j.out
#SBATCH --error={CONTROL}/allocation-%j.err
set -euo pipefail
umask 077
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH
/usr/bin/python3.12 -B {CONTROL}/scripts/newton/run_effusion_screen.py boot
"""
    files = {
        "screen.sbatch": script.encode(),
        "arm.sh": arm.encode(),
        "headers.sh": headers.encode(),
    }
    for name in (
        "scripts/newton/run_effusion_screen.py",
        "scripts/hf/publish_effusion_screen_epoch.py",
    ):
        files[name] = (ROOT / name).read_bytes()
    for role in ("control", "candidate"):
        path = (
            ROOT
            / f"initiative/experiments/competitive-v1/effusion-review-contrast-v1/{role}-screen.json"
        )
        files[role + "-screen.json"] = path.read_bytes()
    return files


def remote(program, payload, *, cpu_name):
    # One CPU staging/publication step on the owning node, no shared output files.
    command = [
        "salloc",
        "--quiet",
        "--account=skattel",
        "--partition=normal",
        "--nodelist=evc34",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=2G",
        "--time=00:15:00",
        "--job-name=" + cpu_name,
        "srun",
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
        raise ValueError("Verified committed source required")
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
                "actual_complete_copy_job": "869665",
                "independent_single_GPU_arms": True,
                "epochs_preserved_privately": 3,
                "competition_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    envelope = {
        "source_git": head,
        "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
    }
    program = """
import base64,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
os.umask(0o077)
b=Path("/home/sa305415/rsna-knee");route=b/"effusion-native-cache-route-20261004-v1/recovery-v2"
c=Path("/tmp/rsna-effusion-screen-20261004-v2")
assert os.uname().nodename.split('.')[0]=='evc34' and os.environ['SLURM_CPUS_PER_TASK']=='1'
assert not os.environ.get('SLURM_JOB_GPUS') and not c.exists()
prior=Path('/tmp/rsna-effusion-screen-20261004-v1')
assert subprocess.check_output(['sacct','-X','-j','869702','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).strip()=='869702|FAILED|1:0'
assert not (prior/'control-trainer-intent.json').exists() and not (prior/'candidate-trainer-intent.json').exists()
assert json.loads((prior/'terminal.json').read_text())['role_exit_codes']=={'control':124,'candidate':124}
assert sum(p.stat().st_size for p in prior.rglob('*') if p.is_file())<16*2**20
active=subprocess.check_output(['squeue','-u','sa305415','-h','-o','%b'],text=True).splitlines()
assert not any('gpu' in x.lower() for x in active),'Reconcile existing GPU jobs before screen dispatch'
assert subprocess.check_output(['sacct','-X','-j','869300,869665','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).splitlines()==['869300|COMPLETED|0:0','869665|COMPLETED|0:0']
copy_path=route/'copy-accepted.json';copy=json.loads(copy_path.read_text())
assert copy['status']=='private_original_native384_copy_fully_byte_verified' and copy['members']==37812 and copy['bytes']==140626483351
assert copy['destination_node']=='evc34' and copy['job_id']=='869665'
assert Path(copy['cache_root']).is_dir()
f=json.loads((route/'funding.json').read_text());paid={}
for p in [route]+list(route.iterdir())+[b/'logs/rsna-effusion-native-route-v2-869665.out',b/'logs/rsna-effusion-native-route-v2-869665.err']:
 assert not p.is_symlink();s=p.stat();paid[s.st_dev,s.st_ino]=s.st_size
assert len(paid)<=10 and sum(paid.values())<3*2**20
row=[x.split() for x in subprocess.check_output(['lfs','quota','-u','sa305415','/lustre/fs1'],text=True).splitlines() if x.lstrip().startswith('/lustre/fs1')][0]
used,soft,count,limit=[int(row[i]) for i in [1,2,5,6]]
future=f['future_shared_bytes_before_placement']-sum(paid.values());nf=f['future_shared_files_before_placement']-len(paid)
remaining=(soft-used)*1024-future;files_left=limit-count-nf
assert remaining>=20*2**30 and files_left>=5000
assert shutil.disk_usage('/tmp').free>=(20+7+3)*2**30+32*2**20
e=json.load(sys.stdin);raw={k:base64.b64decode(v,validate=True) for k,v in e['files'].items()}
names={'screen.sbatch','arm.sh','headers.sh','scripts/newton/run_effusion_screen.py','scripts/hf/publish_effusion_screen_epoch.py','control-screen.json','candidate-screen.json','source-provenance.json'}
assert set(raw)==names and sum(map(len,raw.values()))<2**20
a=json.loads(raw['control-screen.json']);d=json.loads(raw['candidate-screen.json']);policy=d['fitting_supervision'];d['fitting_supervision']=a['fitting_supervision']
assert a==d and a['fitting_supervision']['enabled'] is False and policy['enabled'] is True
assert a['epochs']==3 and a['diagnostic_study_limit'] is None and a['side']==384
assert hashlib.sha256((b/'effusion-supervision-engineering-20261003-v1/recovery-v2/recovery-accepted.json').read_bytes()).hexdigest()=='be201fc89f16ede13f379d854018f4d7dc5ca9a55a26a528081418eea87632ef'
c.mkdir(mode=0o700)
def save(name,value):
 with (c/name).open('x') as out:json.dump(value,out,indent=2,sort_keys=True);out.write('\\n');out.flush();os.fsync(out.fileno())
for name,data in raw.items():
 p=c/name;p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open('xb') as out:out.write(data)
manifest={'launch_git':e['source_git'],'scientific_git':'665c0d515410df24f1f37ed5d9f8cb7a0f0fc480','copy_receipt_sha256':hashlib.sha256(copy_path.read_bytes()).hexdigest(),'reused_code_root':str(prior/'code'),'reused_runtime_receipt':str(prior/'runtime.json'),'reused_runtime_receipt_sha256':hashlib.sha256((prior/'runtime.json').read_bytes()).hexdigest(),'failed_original_GPU_job':'869702','allocated_driver_timeout_seconds':180,'files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in raw.items()}}
save('files.json',manifest)
funding={'shared_used_kB':used,'shared_soft_kB':soft,'shared_used_files':count,'shared_file_soft':limit,'future_shared_bytes':future,'future_shared_files':nf,'after_liabilities_GiB':remaining/2**30,'after_liabilities_files':files_left,'new_shared_bytes':0,'new_shared_files':0,'new_node_local_controls_MiB':16,'original_control_MiB_cap_retained':16,'node_local_runtime_GiB':7,'node_local_checkpoints_GiB':3,'node_local_reserve_GiB':20,'epoch_count_per_arm':3,'private_epoch_objects_per_epoch':5,'original_assets_retained':True,'deletion_credit':0,'memory_GiB_per_arm':96,'requested_CPUs_per_arm':2,'independent_H100s':2,'GPU_nodes':1}
save('funding.json',funding);save('dispatch-intent.json',{'automatic_retry':False,'launch_git':e['source_git'],'two_independent_arms':True})
job=subprocess.check_output(['sbatch','--parsable',str(c/'screen.sbatch')],text=True).strip();assert job.isdecimal()
result={'job_id':job,'node':'evc34','control':str(c),'launch_git':e['source_git'],'scientific_git':manifest['scientific_git'],'GPU_nodes':1,'GPUs':2,'CPUs':4,'RAM_GiB':192,'epochs_per_arm':3,'updates_per_arm':843,'training_studies':2247,'heldout_studies':1156,'automatic_retry':False,'staging_CPU_job':os.environ['SLURM_JOB_ID'],'clinical_result_observed':False}
save('dispatch-result.json',result);print(json.dumps({'dispatch':result,'funding':funding,'source_manifest':manifest}),flush=True)
"""
    save(LOCAL / "dispatch-intent.json", {"source_git": head, "automatic_retry": False})
    result = remote(program, envelope, cpu_name="rsna-effusion-screen-stage")
    (LOCAL / "dispatch.stdout").write_bytes(result.stdout)
    (LOCAL / "dispatch.stderr").write_bytes(result.stderr)
    if result.returncode:
        raise ValueError("Dispatch uncertain or failed; inspect original before any retry")
    record = json.loads(result.stdout)
    save(LOCAL / "dispatch-result.json", record)
    audit = ROOT / "initiative/audits/effusion-supervision-screen-v1/recovery-v2"
    for name in ("dispatch", "funding", "source_manifest"):
        save(audit / (name.replace("_", "-") + ".json"), record[name])
    for name in ("screen.sbatch", "arm.sh", "headers.sh"):
        (audit / (name + ".txt")).write_bytes(files[name])
    print(json.dumps(record["dispatch"], sort_keys=True))


def publish(role, epoch):
    from scripts.hf.dispatch_native384_epoch import credential

    intent = LOCAL / "publications" / role / f"epoch-{epoch:03d}-intent.json"
    if intent.exists():
        raise ValueError("Original publication attempt exists; reconcile read-only")
    program = f"""
import json,os,subprocess,sys
from pathlib import Path
c=Path({CONTROL!r});r=json.loads((c/'runtime.json').read_text());m=json.loads((c/'files.json').read_text())
import hashlib
for n in ['scripts/newton/run_effusion_screen.py','scripts/hf/publish_effusion_screen_epoch.py']:
 assert hashlib.sha256((c/n).read_bytes()).hexdigest()==m['files'][n]['sha256']
assert (c/{role!r}/'epoch-checkpoints'/"epoch-{epoch:03d}.json").is_file()
assert not (c/'epoch-publication'/{role!r}/"epoch-{epoch:03d}"/'intent.json').exists()
os.environ['PYTHONPATH']=str(c);os.environ['PYTHONNOUSERSITE']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
subprocess.run([r['python'],'-B',str(c/'scripts/hf/publish_effusion_screen_epoch.py'),'publish',{role!r},str({epoch})],input=sys.stdin.buffer.read(),check=True)
"""
    token = credential()
    save(intent, {"role": role, "epoch": epoch, "automatic_retry": False})
    result = remote(program, {"token": token}, cpu_name=f"rsna-effusion-{role}-e{epoch}-publish")
    output = intent.with_name(intent.name.replace("intent", "result"))
    if result.returncode:
        save(output, {"status": "uncertain_or_failed", "automatic_retry": False})
        raise ValueError("Private publication uncertain; inspect original journals read-only")
    receipt = json.loads(result.stdout)
    save(output, receipt)
    save(
        ROOT
        / f"initiative/audits/effusion-supervision-screen-v1/recovery-v2/{role}-epoch-{epoch:03d}-verified.json",
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
