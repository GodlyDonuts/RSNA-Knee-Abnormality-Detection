"""Admit one isolated private Colab checkpoint namespace in the canonical ledger.

No Dataset creation, GPU allocation or submission occurs here. The sole Colab
writer can create seven named private v1 Datasets; primary entries stay with
their existing coordinator. Actual receipts are reconciled back to this ledger.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def payload(pin):
    spec = importlib.util.spec_from_file_location("Colab_lease_sources", HERE / "derive.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    spec = importlib.util.spec_from_file_location("Colab_lease_training", HERE / "training.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    for name in ("lease.py", "colab_storage.py", "training.py", "derive.py", "kaggle_io.py"):
        assert (
            d.frozen((HERE / name).relative_to(REPO).as_posix(), pin) == (HERE / name).read_text()
        )
    files = d.sources()
    files["colab_storage.py"] = (HERE / "colab_storage.py").read_text()
    files["actual_colab_train.py"] = t.build()
    return files


def main(pin, socket):
    os.umask(0o077)
    files = payload(pin)
    code_sha = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in files.items()}
    journal = REPO / "artifacts" / HERE.name / "checkpoint-lease"
    journal.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (journal / "intent.json").open("x") as f:
        json.dump({"source_git": pin, "code_sha256": code_sha}, f)
    env = REPO / ".env"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    token = next(
        v.strip().strip("\"'")
        for line in env.read_text().splitlines()
        for k, sep, v in [line.partition("=")]
        if sep and k.strip() == "KAGGLE_API_TOKEN"
    )
    program = """import importlib.util,json,os,sqlite3,sys
from pathlib import Path
v=json.load(sys.stdin)
root=Path('/tmp/rsna-colab-private-kaggle-control-20261021-v2')
def load(name):
 s=importlib.util.spec_from_file_location('lease_'+name,root/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
w,p=load('workflow'),load('publication')
import requests
r=w.Remote.__new__(w.Remote);r.session=requests.Session();r.session.headers.update({'Authorization':'Bearer '+v.pop('token')})
# Existing owner identity was authenticated by the original transport; this
# fresh call again binds the current credential, without printing the token.
i=r.call('security.OAuthService','IntrospectToken',{'token':r.session.headers['Authorization'][7:]})
assert i.get('active') is True and i.get('username')=='godlydonuts';r.username=i['username']
ledger=Path('/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3')
claim=ledger.parent/'coordinator-claim';claim.mkdir(mode=0o700)
owner={'kind':'isolated_private_Colab_checkpoint_namespace','CPU_job':os.environ['SLURM_JOB_ID'],'source_git':v['source_git']}
w.new_json(claim/'owner.json',owner)
try:
 with w.ledger(ledger,'godlydonuts') as db:
  w.verify_account(db,r)
  slug='godlydonuts/rsna-colab-c100-contexts-20261021-v1'
  row=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('private-colab-input-handoff:'+slug,)).fetchone()[0])
  assert row['state']=='private_create_returned_download_pending'
  meta=p.metadata(r,slug)
  status=r.call('datasets.DatasetApiService','GetDatasetStatus',{'ownerSlug':'godlydonuts','datasetSlug':slug.split('/')[1]})
  assert status.get('status') in ('READY',5)
  assert p.list_files(r,slug)=={n:x['bytes'] for n,x in row['files'].items()}
  manifest_sha=row['files']['input-manifest.bin']['sha256']
  keys=[f'epoch-{e:03d}' for e in range(1,7)]+['terminal']
  slugs={k:'godlydonuts/rsna-colab-c100-'+k+'-'+manifest_sha[:12] for k in keys}
  for s in slugs.values():p.require_absent(r,s)
  key='private-colab-checkpoint-lease:'+manifest_sha
  assert db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone() is None
  record={'status':'canonical_isolated_Colab_checkpoint_namespace_admitted',**owner,
    'input_slug':slug,'input_dataset':meta,'input_manifest_sha256':manifest_sha,'seed':20261021,
    'storage_control_git':v['source_git'],'code_sha256':v['code_sha256'],'slugs':slugs,
    'maximum_private_dataset_creations':7,'maximum_bytes_per_dataset':2*2**30,
    'reserved_account_bytes':80*2**30,'canonical_ledger_key':key,
    'only_one_Colab_writer':True,'automatic_mutation_retry':False,
    'GPU_allocations_or_competition_entries_authorized_by_this_lease':False,
    'original_B_C_C0_coordinator_unchanged':True}
  w.new_json(root/'checkpoint-lease.json',record)
  db.execute('INSERT INTO meta VALUES (?,?)',(key,json.dumps(record)));db.commit()
  print(json.dumps(record),flush=True)
finally:
 assert json.loads((claim/'owner.json').read_text())==owner
 (claim/'owner.json').unlink();claim.rmdir()
"""
    cmd = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc27",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=1G",
        "--time=00:05:00",
        "--immediate=60",
        "--job-name=rsna-colab-checkpoint-lease",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        program,
    ]
    result = subprocess.run(
        ["ssh", "-S", socket, "-o", "BatchMode=yes", "newton", shlex.join(cmd)],
        input=json.dumps({"token": token, "source_git": pin, "code_sha256": code_sha}),
        text=True,
        capture_output=True,
        timeout=360,
    )
    with (journal / "transport.json").open("x") as f:
        json.dump(
            {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, f
        )
    assert result.returncode == 0, (
        "Reconcile original canonical lease; never repeat uncertain mutation"
    )
    value = json.loads(result.stdout)
    with (journal / "accepted.json").open("x") as f:
        json.dump(value, f, sort_keys=True)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-git", required=True)
    p.add_argument("--socket", required=True)
    args = p.parse_args()
    main(args.source_git, args.socket)
