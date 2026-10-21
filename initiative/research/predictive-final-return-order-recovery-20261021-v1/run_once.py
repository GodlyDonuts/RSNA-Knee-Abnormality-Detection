"""Root-only one-shot recovery of an actually failed, reconciled return."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'initiative/research/predictive-evc27-deadline-placement-20261020-v1'
JOURNAL = ROOT / 'artifacts/predictive-evc27-deadline-placement-20261020-v1/finite-owner'
COMMAND = "import sys,json,hashlib;e=json.load(sys.stdin);assert hashlib.sha256(e['source'].encode()).hexdigest()==e['payload']['recovery_source_sha256'];n={'__name__':'frozen_return_recovery'};exec(compile(e['source'],'frozen_return_recovery.py','exec'),n);print(json.dumps(n['main'](e['payload'])),flush=True)"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def once(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def payload(arm, pin):
    source = (HERE / 'transport.py').read_bytes()
    for name in ['run_once.py', 'transport.py']:
        path = HERE / name
        assert subprocess.check_output(['git', 'show', pin + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT) == path.read_bytes()
    original = read(OLD / ('actual-fixed-final-' + arm + '.json'))
    assert original['actual_Slurm_terminal']['state'] == 'COMPLETED'
    assert original['actual_Slurm_terminal']['exit_code'] == '0:0' and original['stderr_bytes'] == 0
    terminal = {k: v for k, v in original.items() if k not in ['actual_Slurm_terminal', 'stderr_bytes', 'placement_source_git']}
    pubs = [read(OLD / f'actual-owner-publish-{arm}-{e:03d}.json') for e in range(1, 7)]
    job = read(OLD / 'actual-known-dispatches-all27.json')['jobs'][arm]
    for e, p in enumerate(pubs, 1):
        assert p['arm'] == arm and p['epoch'] == e and p['GPU_job'] == job['GPU_job']
        assert p['status'] == 'actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified'
        assert p['objects'] == 5 and p['canonical_shared_publication_lock_used']
        assert p == read(JOURNAL / f'publish-{arm}-{e:03d}-result.json')
    assert read(OLD / ('actual-return-unresolved-' + arm + '.json'))['automatic_retry'] is False
    assert not (OLD / ('actual-owner-return-' + arm + '.json')).exists()
    inspected = read(HERE / ('actual-' + arm + '-failed-return-readonly.json'))
    assert inspected['mutations'] == 0 and inspected['root_exists']
    assert set(inspected['files']) == {'return-intent.json', 'publication/epoch-006/manifest.json'}
    old = inspected['files']['return-intent.json']
    receiver = old['intent']['CPU_job']
    transport = read(JOURNAL / ('return-' + arm + '-sender-transport.json'))
    assert transport['returncode'] != 0
    import re
    senders = re.findall(r'srun: job (\d+) has been allocated resources', transport['stderr'])
    assert len(senders) == 1
    sender = senders[0]
    for identity in [receiver, sender]:
        assert any(row.startswith(identity + '|FAILED|1:0|') for row in inspected['sacct'])
    return source.decode(), {'arm': arm, 'job': job, 'terminal': terminal, 'publications': pubs,
        'destination': '/tmp/rsna-predictive-' + arm + '-evc27-return-20261020-v1',
        'failed_partial': inspected['files']['publication/epoch-006/manifest.json'],
        'failed_intent_sha256': old['sha256'], 'failed_receiver': receiver, 'failed_sender': sender,
        'handoff_sha256': old['intent']['handoff_sha256'],
        'minimum_free_bytes': inspected['free_tmp_bytes'] - 512 * 2**20,
        'recovery_source_sha256': sha(source)}


def command(node, mode, socket):
    remote = ['srun', '--account=skattel', '--partition=normal', '--qos=normal',
        '--nodelist=' + node, '--ntasks=1', '--cpus-per-task=1', '--gres=none', '--mem=1G',
        '--time=00:15:00', '--job-name=rsna-final-return-order-' + mode,
        'env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
        '/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python', '-B', '-c', COMMAND]
    return ['ssh', '-S', socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'newton', shlex.join(remote)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['inspect', 'repair'])
    parser.add_argument('--arm', choices=['C', 'B', 'C0'], required=True)
    parser.add_argument('--source-git', required=True)
    parser.add_argument('--socket', default='/tmp/rsna-research-20261001-stable.sock')
    args = parser.parse_args()
    os.umask(0o077)
    source, v = payload(args.arm, args.source_git)
    def envelope(mode, **extra):
        return json.dumps({'source': source, 'payload': {**v, 'mode': mode, **extra}})
    phase = args.arm + '-' + args.phase
    journal = ROOT / 'artifacts' / HERE.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.phase == 'inspect':
        once(journal / (phase + '-intent.json'), {'source_git': args.source_git, 'automatic_retry': False})
        r = subprocess.run(command('evc27', 'inspect', args.socket), input=envelope('inspect'),
                           capture_output=True, text=True, timeout=900)
        once(journal / (phase + '-transport.json'), {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
        assert r.returncode == 0, r.stderr
        result = json.loads(r.stdout)
        assert result['status'] == 'actual_original_source_and_failed_prefix_reconciled' and result['mutations'] == 0
        once(HERE / ('actual-' + args.arm + '-source-reconciled.json'), result)
    else:
        proof = read(HERE / ('actual-' + args.arm + '-source-reconciled.json'))
        assert proof['arm'] == args.arm and proof['header']['terminal'] == v['terminal']
        assert proof['header']['publications'] == v['publications']
        # Original owner has durably abandoned this arm's return, but still owns training/publication.
        assert read(JOURNAL / ('return-unresolved-' + args.arm + '.json'))['automatic_retry'] is False
        once(journal / (phase + '-intent.json'), {'source_git': args.source_git, 'arm': args.arm,
            'source_sha256': v['recovery_source_sha256'], 'original_failed_receiver': v['failed_receiver'],
            'original_failed_sender': v['failed_sender'], 'automatic_retry': False})
        with (journal / (phase + '-receiver.stderr')).open('x') as err:
            receiver = subprocess.Popen(command('evc33', 'receive', args.socket), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=err, text=True)
            receiver.stdin.write(envelope('receive'))
            receiver.stdin.close()
            receiver.stdin = None
            endpoint = json.loads(receiver.stdout.readline())
            assert endpoint['host'] == 'evc33' and len(endpoint['token']) == 64
            sent = subprocess.run(command('evc27', 'send', args.socket), input=envelope('send', endpoint=endpoint),
                                  capture_output=True, text=True, timeout=900)
            once(journal / (phase + '-sender-transport.json'), {'returncode': sent.returncode, 'stdout': sent.stdout, 'stderr': sent.stderr})
            output, _ = receiver.communicate(timeout=900)
            once(journal / (phase + '-receiver-transport.json'), {'returncode': receiver.returncode, 'stdout': output})
        assert sent.returncode == receiver.returncode == 0, 'Uncertain corrected return; reconcile, never repeat'
        result = json.loads(output)
        assert result == json.loads(sent.stdout)
        assert result['status'] == 'actual_predictive_final_export_returned_to33_all_bytes_SHA_verified'
        assert result['original_training_GPU_job'] == v['job']['GPU_job'] and result['arm'] == args.arm
        once(journal / (phase + '-result.json'), result)
        once(HERE / ('actual-' + args.arm + '-recovered-return.json'), result)
        # Same proven original bytes and expected destination; preserve old failure receipts.
        once(OLD / ('actual-owner-return-' + args.arm + '.json'), result)
    print(json.dumps({'status': result['status'], 'arm': args.arm, 'CPU_job': result['CPU_job']}))


if __name__ == '__main__':
    main()
