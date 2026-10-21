"""One finite CPU input bridge; all private payload and credentials use stdin."""
import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JOURNAL = REPO / 'artifacts' / HERE.name / 'bridge-v3'
SCIENCE = '94a3921aefdf333624caf41aff77aa10b4a0c57e'


def main(pin, socket):
    os.umask(0o077)
    files = {}
    for name in ('bridge.py', 'colab_download.py', 'colab_train.py', 'dispatch.py'):
        path = HERE / name
        raw = subprocess.check_output(['git', 'show', pin + ':' + path.relative_to(REPO).as_posix()], cwd=REPO)
        assert raw == path.read_bytes()
        files[name] = raw.decode()
        if name in ('colab_download.py', 'colab_train.py'):
            assert raw == subprocess.check_output(['git','show',SCIENCE+':'+path.relative_to(REPO).as_posix()],cwd=REPO)
    env = REPO / '.env'
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    subprocess.run(['git', 'check-ignore', '-q', '.env'], cwd=REPO, check=True)
    token = next(line.partition('=')[2].strip().strip('\"\'') for line in env.read_text().splitlines()
                 if line.partition('=')[0].strip() == 'HF_TOKEN')
    assert token.startswith('hf_')
    JOURNAL.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (JOURNAL / 'dispatch-intent.json').open('x') as f:
        json.dump({'source_git':pin, 'created_UTC':datetime.now(timezone.utc).isoformat(),
                   'program_sha256':hashlib.sha256(files['bridge.py'].encode()).hexdigest(),
                   'node':'evc27', 'CPUs':2, 'GPUs':0, 'new_node_scope_bytes':80*2**30,
                   'new_shared_files':0, 'automatic_retry':False}, f)
    command = "import io,json,sys; e=json.load(sys.stdin); sys.stdin=io.StringIO(json.dumps(e['payload'])); exec(compile(e['program'],'original_colab_bridge','exec'),{'__name__':'__main__'})"
    remote = ['srun','--account=skattel','--partition=normal','--qos=normal','--nodelist=evc27',
              '--ntasks=1','--cpus-per-task=2','--gres=none','--mem=8G','--time=02:00:00',
              '--immediate=60','--job-name=rsna-colab-input-bridge','env','CUDA_VISIBLE_DEVICES=',
              'PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2',
              '/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python','-B','-c',command]
    payload = {'source_git':SCIENCE, 'bridge_control_git':pin, 'token':token,
               'own_files':{n:files[n] for n in ('colab_train.py','colab_download.py')}}
    with (JOURNAL/'stdout.jsonl').open('x') as out, (JOURNAL/'stderr.txt').open('x') as err:
        result = subprocess.run(['ssh','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=10',
                                 'newton',shlex.join(remote)],
                                input=json.dumps({'program':files['bridge.py'],'payload':payload}),
                                text=True, stdout=out, stderr=err, timeout=7500, check=False)
    with (JOURNAL/'exit.json').open('x') as f:
        json.dump({'returncode':result.returncode, 'ended_UTC':datetime.now(timezone.utc).isoformat()},f)
    if result.returncode:
        raise SystemExit('Bridge failed or uncertain; reconcile original, never repeat dispatch')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-git',required=True);p.add_argument('--socket',required=True)
    args=p.parse_args();main(args.source_git,args.socket)
