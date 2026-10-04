"""One second-seed contrast, reusing verified original assets without copying them."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess

from scripts.newton import dispatch_effusion_screen as original

ROOT = original.ROOT
LOCAL = ROOT / "artifacts/effusion-seed2-v1"
AUDIT = ROOT / "initiative/audits/effusion-supervision-seed2-v1"
CONTROL = "/tmp/rsna-effusion-seed20261004-evc32-v1"
SEED = 20261004
PREFIX = "experiments/dinov3-v1/training-epochs/effusion-reviewed31-fold0-665c0d5-seed20261004-screen3-v1"
OLD_MANIFEST_SHA = "6b8e5c48a9da90b4dbc371ecd529cf371b31cb159e9f0e94670ccaf04b60f669"
OLD_PREPARED_SHA = "c69250b428c4dc92a7a4db2acd67b7e75e9fda02f80fb2702c4a9688c1c00c78"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_configs(configs):
    """Reject all recipe changes beyond the one predeclared seed and arm policy."""
    for role in ("control", "candidate"):
        expected = json.loads(
            (
                ROOT
                / f"initiative/experiments/competitive-v1/effusion-review-contrast-v1/{role}-screen.json"
            ).read_text()
        )
        expected["seed"] = SEED
        if configs[role] != expected:
            raise ValueError("Second seed must preserve the exact original arm and recipe")


def launch_files():
    files = original.launch_files()
    for name in ("arm.sh", "headers.sh", "control.sbatch", "candidate.sbatch"):
        files[name] = (
            files[name]
            .decode()
            .replace(original.CONTROL, CONTROL)
            .replace("rsna-effusion-evc32-", "rsna-effusion-seed2-")
            .encode()
        )
    run = files["scripts/newton/run_effusion_screen.py"].decode().replace(original.CONTROL, CONTROL)
    # Extraction is deliberately unavailable; only the retained original is used.
    start, stop = run.index("def prepare():"), run.index("def arm(role):")
    run = run[:start] + run[stop:]
    run = run.replace('choices=("prepare", "arm")', 'choices=("arm",)')
    run = run.replace(
        'if args.phase == "prepare":\n        prepare()\n    elif args.role:', "if args.role:"
    )
    check = f"""    prior = Path({original.CONTROL!r})
    for name in ("code", "runtime.json", "cache-stats-private.json"):
        path = CONTROL / name
        if not path.is_symlink() or path.resolve() != (prior / name).resolve():
            raise ValueError("Only exact retained original node-local assets may be reused")
    if sha(prior / "files.json") != {OLD_MANIFEST_SHA!r} or sha(prior / "CPU-prepared.json") != {OLD_PREPARED_SHA!r}:
        raise ValueError("Original completed source/runtime/input admission changed")
    for name, expected in manifest["reused_original_metadata_sha256"].items():
        if sha(prior / name) != expected:
            raise ValueError("Retained runtime or cache inventory changed")
"""
    run = run.replace(
        '    manifest = json.loads((CONTROL / "files.json").read_text())\n',
        '    manifest = json.loads((CONTROL / "files.json").read_text())\n' + check,
        1,
    )
    files["scripts/newton/run_effusion_screen.py"] = run.encode()
    pub = (
        files["scripts/hf/publish_effusion_screen_epoch.py"]
        .decode()
        .replace("effusion-reviewed31-fold0-665c0d5-screen3-v1", PREFIX.split("/")[-1])
    )
    pub = pub.replace(
        'expected["epochs"] = 3', 'expected["epochs"] = 3\n        expected["seed"] = 20261004'
    )
    pub = pub.replace(
        'or meta.get("epoch") != args.epoch',
        'or meta.get("epoch") != args.epoch\n            or meta.get("seed") != 20261004',
    )
    files["scripts/hf/publish_effusion_screen_epoch.py"] = pub.encode()
    configs = {}
    for role in ("control", "candidate"):
        configs[role] = json.loads(
            (
                ROOT
                / f"initiative/audits/effusion-supervision-screen-final-v1/second-seed-replication/{role}-screen.json"
            ).read_text()
        )
        files[role + "-screen.json"] = (
            json.dumps(configs[role], indent=2, sort_keys=True) + "\n"
        ).encode()
    validate_configs(configs)
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    return files


STAGE = """
import base64,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
os.umask(0o077);old=Path('/tmp/rsna-effusion-screen-evc32-20261004-v1');c=Path('/tmp/rsna-effusion-seed20261004-evc32-v1')
assert os.uname().nodename.split('.')[0]=='evc32' and os.environ['SLURM_CPUS_PER_TASK']=='1' and not os.environ.get('SLURM_JOB_GPUS')
assert not c.exists(), 'Existing original intent/control must be reconciled read-only'
p=json.load(sys.stdin);raw={k:base64.b64decode(v,validate=True) for k,v in p['files'].items()}
assert set(raw)=={'arm.sh','headers.sh','control.sbatch','candidate.sbatch','control-screen.json','candidate-screen.json','scripts/newton/run_effusion_screen.py','scripts/hf/publish_effusion_screen_epoch.py'}
assert hashlib.sha256((old/'files.json').read_bytes()).hexdigest()==p['old_manifest_sha256']
assert hashlib.sha256((old/'CPU-prepared.json').read_bytes()).hexdigest()==p['old_prepared_sha256']
source=json.loads((old/'source-provenance.json').read_text());assert source['scientific_source_git']=='665c0d515410df24f1f37ed5d9f8cb7a0f0fc480'
terminal=subprocess.check_output(['sacct','-X','-j','869815,869816,869300,867367','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).splitlines()
assert set(terminal)=={j+'|COMPLETED|0:0' for j in ['869815','869816','869300','867367']}
q=subprocess.check_output(['squeue','-u','sa305415','-h','-o','%i|%T|%N|%b'],text=True).splitlines();assert not any('gpu' in r.lower() for r in q)
row=[r.split() for r in subprocess.check_output(['lfs','quota','-u','sa305415','/home'],text=True).splitlines() if r.lstrip().startswith('/home ')][0];used,soft,count,limit=[int(row[i]) for i in [1,2,5,6]]
future=1404996457;nf=760;gib=((soft-used)*1024-future)/2**30;slots=limit-count-nf;assert gib>=20 and slots>=5000,(gib,slots)
assert shutil.disk_usage('/tmp').free>=(20+7+3)*2**30+32*2**20
# Count retained checkpoints and control bytes without rehashing original models/pixels.
old_checkpoint_bytes=sum(f.stat().st_size for role in ['control','candidate'] for f in (old/role).rglob('*.pt'))
future_checkpoint_bytes=2*4*128*2**20
assert old_checkpoint_bytes+future_checkpoint_bytes<3*2**30
old_control_bytes=sum(f.stat().st_size for f in old.rglob('*') if f.is_file() and not f.is_symlink() and f.suffix!='.pt')
assert old_control_bytes+sum(map(len,raw.values()))+8*2**20<32*2**20
runtime=json.loads((old/'runtime.json').read_text());assert runtime['archive_sha256']=='89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338'
assert runtime['file_members']==40403 and runtime['unpacked_file_bytes']==6322691335 and Path(runtime['python']).is_file()
for role in ['control','candidate']:
 r=json.loads((old/role/'run.json').read_text());assert r['status']=='completed' and r['binding']['config']['seed']==20260922
 for name,want in r['binding']['code_sha256'].items():assert hashlib.sha256((old/'code/rsna'/name).read_bytes()).hexdigest()==want
c.mkdir(mode=0o700)
def save(n,v):
 with (c/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\\n');f.flush();os.fsync(f.fileno())
for name,data in raw.items():
 path=c/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with path.open('xb') as f:f.write(data)
for name in ['code','runtime.json','cache-stats-private.json']:(c/name).symlink_to(old/name)
provenance=dict(source,launch_git=p['source_git'],replication_seed=20261004,parent_seed=20260922,parent_original_jobs=['869815','869816'],original_code_runtime_and_cache_reused=True,new_bulk_extraction_or_copy=False,competition_eligible=False)
save('source-provenance.json',provenance)
meta={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in ['runtime.json','cache-stats-private.json']}
copy=json.loads((old/'files.json').read_text())['copy_receipt_sha256']
m={'launch_git':p['source_git'],'scientific_git':'665c0d515410df24f1f37ed5d9f8cb7a0f0fc480','copy_receipt_sha256':copy,'reused_original_metadata_sha256':meta,'files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in raw.items()}}
data=(c/'source-provenance.json').read_bytes();m['files']['source-provenance.json']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};save('files.json',m)
fund={'used_kB':used,'soft_kB':soft,'used_files':count,'file_soft':limit,'future_shared_bytes':future,'future_shared_files':nf,'after_liabilities_GiB':gib,'after_liabilities_files':slots,'new_shared_files':0,'new_shared_bytes':0,'deletion_credit':0,'all_originals_retained':True,'old_checkpoint_bytes':old_checkpoint_bytes,'new_checkpoint_cap_bytes':future_checkpoint_bytes,'total_checkpoint_cap_bytes':3*2**30,'old_control_bytes':old_control_bytes,'total_control_cap_bytes':32*2**20,'new_control_and_publication_metadata_cap_bytes':8*2**20,'runtime_extraction_or_copy':False,'node_local_reserve_bytes':20*2**30,'retired_original_native_closeout_future_slots':8};save('funding.json',fund)
save('CPU-prepared.json',{'status':'retained_original_assets_actual_reuse_admitted','CPU_job':os.environ['SLURM_JOB_ID'],'source_git':p['source_git'],'seed':20261004,'old_metadata_sha256':meta,'no_model_or_optimizer_run':True})
save('dispatch-intent.json',{'automatic_retry':False,'seed':20261004});jobs={}
for role in ['control','candidate']:
 save(role+'-dispatch-intent.json',{'automatic_retry':False,'seed':20261004,'role':role})
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('SLURM_STEP_') or key.startswith('SLURM_CPU_BIND'):env.pop(key)
 jid=subprocess.check_output(['sbatch','--parsable',str(c/(role+'.sbatch'))],env=env,text=True).strip();assert jid.isdecimal();jobs[role]=jid
 save(role+'-dispatch-result.json',{'role':role,'job_id':jid,'automatic_retry':False})
r={'job_ids':jobs,'CPU_preparation_job':os.environ['SLURM_JOB_ID'],'control':str(c),'source_git':p['source_git'],'scientific_source_git':m['scientific_git'],'seed':20261004,'epochs_per_arm':3,'updates_per_arm':843,'training_studies':2247,'outer_studies':1156,'GPU_nodes':1,'GPUs_per_job':1,'CPUs_per_job':2,'memory_GiB_per_job':96,'automatic_retry':False,'new_bulk_copy_or_extraction':False};save('dispatch-result.json',r)
print(json.dumps({'dispatch':r,'funding':fund,'source_manifest':m,'CPU_prepared':json.loads((c/'CPU-prepared.json').read_text())}),flush=True)
"""


def dispatch():
    if (LOCAL / "dispatch-intent.json").exists():
        raise ValueError("Original attempt exists; read-only reconciliation required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Verified committed source required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    proof = json.loads((AUDIT / "funding-reconciliation.json").read_text())
    if proof["future_shared_files"] != 760 or proof["after_liabilities_files"] < 5000:
        raise ValueError("Explicit funded reconciliation required")
    files = launch_files()
    original.save(LOCAL / "dispatch-intent.json", {"source_git": head, "automatic_retry": False})
    r = original.remote(
        STAGE,
        {
            "source_git": head,
            "old_manifest_sha256": OLD_MANIFEST_SHA,
            "old_prepared_sha256": OLD_PREPARED_SHA,
            "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
        },
        cpu_name="rsna-effusion-seed2-prepare",
    )
    (LOCAL / "dispatch.stdout").write_bytes(r.stdout)
    (LOCAL / "dispatch.stderr").write_bytes(r.stderr)
    if r.returncode:
        raise ValueError("Original stage uncertain/failed; inspect read-only, never repeat")
    record = json.loads(r.stdout)
    original.save(LOCAL / "dispatch-result.json", record)
    for name, value in record.items():
        original.save(AUDIT / (name.replace("_", "-") + ".json"), value)
    for name, data in files.items():
        path = AUDIT / "original-launch-files" / (name + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(record["dispatch"], sort_keys=True))


def publish(role, epoch):
    from scripts.hf.dispatch_native384_epoch import credential

    intent = LOCAL / "publications" / role / f"epoch-{epoch:03d}-intent.json"
    if intent.exists():
        raise ValueError("Existing original publication attempt; reconcile read-only")
    program = f"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
c=Path({CONTROL!r});m=json.loads((c/'files.json').read_text());r=json.loads((c/'runtime.json').read_text())
for n in ['scripts/newton/run_effusion_screen.py','scripts/hf/publish_effusion_screen_epoch.py']:
 assert hashlib.sha256((c/n).read_bytes()).hexdigest()==m['files'][n]['sha256']
assert (c/{role!r}/'epoch-checkpoints/epoch-{epoch:03d}.json').is_file()
assert not (c/'epoch-publication'/{role!r}/'epoch-{epoch:03d}/intent.json').exists()
client=c/'publication-client';client.mkdir(mode=0o700,exist_ok=True);os.environ['HF_HOME']=str(client/'huggingface');os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['PYTHONPATH']=str(c);os.environ['PYTHONNOUSERSITE']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
subprocess.run([r['python'],'-B',str(c/'scripts/hf/publish_effusion_screen_epoch.py'),'publish',{role!r},str({epoch})],input=sys.stdin.buffer.read(),check=True)
"""
    token = credential()
    original.save(intent, {"role": role, "epoch": epoch, "automatic_retry": False})
    r = original.remote(program, {"token": token}, cpu_name=f"rsna-effusion-seed2-{role}-e{epoch}")
    if r.returncode:
        original.save(
            intent.with_name(intent.name.replace("intent", "result")),
            {"status": "uncertain_or_failed", "automatic_retry": False},
        )
        raise ValueError("Original publication uncertain; reconcile read-only")
    proof = json.loads(r.stdout)
    original.save(intent.with_name(intent.name.replace("intent", "result")), proof)
    original.save(AUDIT / f"{role}-epoch-{epoch:03d}-verified.json", proof)
    print(json.dumps(proof, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("dispatch", "publish"))
    p.add_argument("--role", choices=("control", "candidate"))
    p.add_argument("--epoch", type=int, choices=(1, 2, 3))
    a = p.parse_args()
    if a.phase == "dispatch":
        dispatch()
    elif a.role and a.epoch:
        publish(a.role, a.epoch)
    else:
        raise ValueError("One completed immutable role/epoch required")


if __name__ == "__main__":
    main()
