"""One admitted failed final-return repair; no model or publication mutation."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import stat
import subprocess
import sys

MIB = 2**20
CAP = 512 * MIB
EOF = b'RSNA_FINAL_RETURN_RECOVERY_EOF_V1\n'
NAMES = {'terminal.json', 'publication/epoch-006/verified.json'} | {
    'publication/epoch-006/' + n for n in
    ['model.pt', 'provenance.json', 'own-source.json', 'manifest.json', 'source-permission.txt']}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def once(path, value):
    with Path(path).open('xb') as out:
        out.write(canonical(value) + b'\n')
        out.flush()
        os.fsync(out.fileno())


def exact(stream, count):
    parts = []
    while count:
        part = stream.read(count)
        require(bool(part), 'Interrupted original-object stream')
        parts.append(part)
        count -= len(part)
    return b''.join(parts)


def write_frame(stream, value):
    raw = canonical(value)
    require(len(raw) < 2 * MIB, 'Bounded control frame')
    stream.write(len(raw).to_bytes(8, 'big') + raw)
    stream.flush()


def read_frame(stream):
    size = int.from_bytes(exact(stream, 8), 'big')
    require(0 < size < 2 * MIB, 'Bounded control frame')
    return json.loads(exact(stream, size))


def signature(path):
    path = Path(path)
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_uid == os.geteuid()
            and not s.st_mode & 0o077, 'Private owned regular object')
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def check_header(header, v):
    require(header['terminal'] == v['terminal'], 'Original complete terminal changed')
    require(header['publications'] == v['publications'], 'Original six publications changed')
    require(header['binding_sha256'] == v['job']['binding_sha256'], 'Original binding changed')
    require(set(header['files']) == NAMES and header['ordered_names'] == sorted(NAMES),
            'Exact explicitly ordered seven original objects')
    require(all(type(x['bytes']) is int and x['bytes'] > 0 and len(x['sha256']) == 64
                for x in header['files'].values())
            and sum(x['bytes'] for x in header['files'].values()) < CAP, 'Bounded object lengths')
    for name, expected in v['publications'][-1]['files'].items():
        require({k: header['files']['publication/epoch-006/' + name][k]
                 for k in ['bytes', 'sha256']} == expected, 'Original private publication object changed')


def send_objects(out, paths, files):
    # The metadata serializer sorts mapping keys. The body MUST use the same order.
    for name in sorted(NAMES):
        path = paths[name]
        require(signature(path) == files[name]['stat'], 'Original source stat changed')
        with path.open('rb') as source:
            shutil.copyfileobj(source, out, MIB)
        require(signature(path) == files[name]['stat'], 'Original source changed in transit')
    out.write(EOF)
    out.flush()


def receive_objects(inc, root, files):
    for name in sorted(NAMES):
        path = root / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        h, left = hashlib.sha256(), files[name]['bytes']
        with path.open('xb') as target:
            while left:
                block = exact(inc, min(left, MIB))
                target.write(block)
                h.update(block)
                left -= len(block)
            target.flush()
            os.fsync(target.fileno())
        require(h.hexdigest() == files[name]['sha256'], 'Original object SHA mismatch')
    require(exact(inc, len(EOF)) == EOF and inc.read(1) == b'', 'Complete stream EOF')


def source(v):
    require(os.uname().nodename == 'evc27', 'Actual original owning node')
    job = v['job']
    state = subprocess.check_output(['sacct', '-j', job['GPU_job'],
        '--format=JobID,State,ExitCode,NodeList', '-n', '-P', '-X'], text=True).strip()
    require(state == job['GPU_job'] + '|COMPLETED|0:0|evc27', 'Actual original terminal GPU job')
    binding_path = Path('/tmp/rsna-predictive-evc27-deadline-placement-20261020-v1/full-admission') / ('binding-' + v['arm'] + '.private.json')
    signature(binding_path)
    require(sha(binding_path.read_bytes()) == job['binding_sha256'], 'Exact original binding')
    binding = json.loads(binding_path.read_bytes())
    run = Path(job['run_root'])
    require(binding['run_root'] == str(run) and binding['arm'] == v['arm'], 'Original run binding')
    terminal_raw = (run / 'terminal.json').read_bytes()
    require(json.loads(terminal_raw) == v['terminal'], 'Original terminal bytes decode equally')
    publications = [json.loads((run / f'publication/epoch-{e:03d}/verified.json').read_bytes())
                    for e in range(1, 7)]
    require(publications == v['publications'], 'Six exact original private proofs')
    # Establish the observed failure: receiver consumed terminal bytes as manifest.
    require(sha(terminal_raw[:v['failed_partial']['bytes']]) == v['failed_partial']['sha256'],
            'Observed partial is not the original terminal prefix; reconcile separately')
    paths = {name: run / name for name in NAMES}
    paths['publication/epoch-006/source-permission.txt'] = Path(binding['license']['path'])
    files = {}
    for name in sorted(NAMES):
        s = signature(paths[name])
        original = publications[-1]['files'].get(Path(name).name)
        expected = original or {'bytes': s[1], 'sha256': sha(paths[name].read_bytes())}
        require(s[1] == expected['bytes'], 'Original export length changed')
        files[name] = {**expected, 'stat': s}
    header = {'terminal': v['terminal'], 'publications': publications, 'files': files,
              'ordered_names': sorted(NAMES), 'binding_sha256': job['binding_sha256']}
    check_header(header, v)
    return paths, header


def sender(v):
    paths, header = source(v)
    endpoint = v['endpoint']
    require(endpoint['host'] == 'evc33' and len(endpoint['token']) == 64, 'Original destination only')
    with socket.create_connection(('evc33', endpoint['port']), timeout=60) as connection:
        connection.settimeout(600)
        out, inc = connection.makefile('wb'), connection.makefile('rb')
        out.write(endpoint['token'].encode() + b'\n')
        write_frame(out, header)
        send_objects(out, paths, header['files'])
        connection.shutdown(socket.SHUT_WR)
        return read_frame(inc)


def receiver(v):
    require(os.uname().nodename == 'evc33' and not os.environ.get('SLURM_JOB_GPUS'),
            'Original destination CPU only')
    root = Path(v['destination'])
    require(root == Path('/tmp/rsna-predictive-' + v['arm'] + '-evc27-return-20261020-v1')
            and root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.geteuid(),
            'Original failed return directory')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(all(not p.is_symlink() and p.stat().st_uid == os.geteuid()
                for p in root.rglob('*')), 'Original private tree ownership')
    require(actual == {'return-intent.json', 'publication/epoch-006/manifest.json'},
            'Only exact reconciled original partial may recover; never repeat a recovery')
    old = root / 'return-intent.json'
    partial = root / 'publication/epoch-006/manifest.json'
    signature(old)
    signature(partial)
    require(sha(old.read_bytes()) == v['failed_intent_sha256'], 'Original failure intent changed')
    require(partial.stat().st_size == v['failed_partial']['bytes']
            and sha(partial.read_bytes()) == v['failed_partial']['sha256'], 'Original partial changed')
    require(shutil.disk_usage('/tmp').free >= v['minimum_free_bytes'], 'Previously admitted capacity decreased')
    # Existing <=512MiB funded return scope is reused; old 681-byte partial is retained.
    once(root / 'order-recovery-intent.json', {
        'source_sha256': v['recovery_source_sha256'], 'CPU_job': os.environ['SLURM_JOB_ID'],
        'failed_receiver': v['failed_receiver'], 'failed_sender': v['failed_sender'],
        'original_partial': v['failed_partial'], 'automatic_retry': False})
    listener = socket.socket()
    listener.bind(('0.0.0.0', 0))
    listener.listen(1)
    listener.settimeout(600)
    token = secrets.token_hex(32)
    print(json.dumps({'host': 'evc33', 'port': listener.getsockname()[1], 'token': token}), flush=True)
    connection, _ = listener.accept()
    listener.close()
    with connection:
        connection.settimeout(600)
        inc, out = connection.makefile('rb'), connection.makefile('wb')
        require(inc.readline(80) == token.encode() + b'\n', 'Private endpoint token')
        header = read_frame(inc)
        check_header(header, v)
        preserved = root / ('failed-original-' + v['failed_receiver'] + '-manifest.partial')
        require(not preserved.exists(), 'Original failure preservation target exists')
        os.link(partial, preserved)
        require(signature(partial) == signature(preserved), 'Retain exact original partial inode')
        partial.unlink()
        receive_objects(inc, root, header['files'])
        require(json.loads((root / 'terminal.json').read_bytes()) == header['terminal'], 'Terminal body changed')
        require(json.loads((root / 'publication/epoch-006/verified.json').read_bytes()) == header['publications'][-1],
                'Publication body changed')
        result = {'status': 'actual_predictive_final_export_returned_to33_all_bytes_SHA_verified',
            'CPU_job': os.environ['SLURM_JOB_ID'], 'arm': v['arm'], 'source_node': 'evc27',
            'destination_node': 'evc33', 'original_training_GPU_job': v['job']['GPU_job'],
            'science_git': v['job']['source_git'], 'handoff_sha256': v['handoff_sha256'],
            'binding_sha256': v['job']['binding_sha256'],
            'objects': {name: {k: row[k] for k in ['bytes', 'sha256']} for name, row in header['files'].items()},
            'original_six_private_revisions': [p['revision'] for p in v['publications']],
            'mirror_run_root': str(root), 'originals_deleted': False, 'new_publication': False,
            'order_recovery_source_sha256': v['recovery_source_sha256'],
            'original_failed_partial_preserved': str(preserved)}
        once(root / 'return-verified.json', result)
        write_frame(out, result)
        return result


def main(v):
    os.umask(0o077)
    require(v['arm'] in ['B', 'C', 'C0'] and v['job']['arm'] == v['arm'], 'Exact original arm')
    require(v['terminal']['arm'] == v['arm'] and v['terminal']['updates'] == 2916
            and v['terminal']['epochs'] == 6 and v['terminal']['study_exposures'] == 23322,
            'Full-data fixed final only')
    require(v['terminal']['source_git'] == v['job']['source_git']
            == '44cc559e947560a635c34fee7b1f2a3b7be3a241', 'Original numerical source only')
    if v['mode'] == 'inspect':
        _, header = source(v)
        return {'status': 'actual_original_source_and_failed_prefix_reconciled',
                'CPU_job': os.environ['SLURM_JOB_ID'], 'arm': v['arm'],
                'header': header, 'model_bytes_rehashed': 0, 'mutations': 0}
    return {'send': sender, 'receive': receiver}[v['mode']](v)


if __name__ == '__main__':
    try:
        print(json.dumps(main(json.load(sys.stdin))), flush=True)
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__, 'reason': str(error)}), file=sys.stderr)
        raise SystemExit(1)
