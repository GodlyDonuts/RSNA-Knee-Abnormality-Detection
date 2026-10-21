"""One explicit continuation of CPU995557 after its actual HF storage rejection.

Never creates a repository, repeats a completed upload, or rewrites an archive.
Requires the owner/storage handoff to be resolved before dispatch. Original
numerical source and all original failure/intent receipts remain unchanged.
"""

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path('/tmp/rsna-colab-independent-C100-20261021-v3')
REPO = 'GodlyDonuts/RSNA-Colab-Deadline-20261021'
BINDING = Path('/tmp/rsna-predictive-density100-full-20261020-v1/binding.private.json')
BINDING_SHA = 'f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd'
LOCK_SOURCE = Path('/home/sa305415/rsna-knee/native-full-cache-two-lanes-v1/common/runner.py')
LOCK_SHA = '3575a18ecd7a13a2becc9348c48a9ebda95f8934069e67dd1912737e6194118c'
LOCK = LOCK_SOURCE.parent.parent / 'publication.lock'
GIB = 2**30


class Archive:
    """Replay existing tar headers to reconstruct the lost in-memory manifest."""

    def __init__(self, path):
        self.existing = Path(path).exists()
        self.tar = tarfile.open(path, 'r:' if self.existing else 'x',
                                format=tarfile.USTAR_FORMAT)
        self.members = iter(self.tar) if self.existing else None

    def __enter__(self):
        return self

    def addfile(self, info, src):
        if self.existing:
            old = next(self.members)
            assert old.isfile() and not old.issym() and not old.islnk()
            assert (old.name, old.size, old.mode) == (info.name, info.size, 0o600)
        else:
            self.tar.addfile(info, src)

    def __exit__(self, kind, value, trace):
        try:
            if kind is None and self.existing:
                assert next(self.members, None) is None
        finally:
            self.tar.close()


def shards(rows, cap=4 * GIB):
    group, size = [], 10240
    for index, row in enumerate(rows):
        cost = 512 + ((row['proof']['bytes'] + 511) // 512) * 512
        assert cost + 10240 <= cap
        if group and size + cost > cap:
            yield group
            group, size = [], 10240
        group.append((index, row)); size += cost
    if group:
        yield group


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def signature(path):
    p = Path(path)
    s = p.lstat()
    assert p.is_file() and not p.is_symlink() and s.st_uid == os.getuid()
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def original(ref):
    p = Path(ref['path'])
    assert digest(p) == ref['sha256']
    return p


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == 'evc27' and not os.environ.get('SLURM_JOB_GPUS')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert v.pop('storage_handoff_resolved') is True
    assert digest(BINDING) == BINDING_SHA and ROOT.is_dir()
    old_intent = json.loads((ROOT / 'intent.json').read_text())
    assert old_intent['CPU_job'] == '995557'
    assert old_intent['source_git'] == v['source_git']
    assert not (ROOT / 'complete.json').exists()
    assert not (ROOT / 'manifest.private.json').exists()
    b = json.loads(BINDING.read_text())
    rows = json.loads(original(b['context_index']).read_text())
    assert len(rows) == 3887 and sum(r['proof']['bytes'] for r in rows) == 67005012716
    for r in rows:
        assert signature(r['path']) == r['proof']['stat']
    parent = Path(b['parent_weight']['path'])
    assert signature(parent) == b['parent_weight']['stat']
    # Existing admission retains ~331 GB including all other scopes. Add a
    # separate 80 GiB bridge cap without using any original's reservation.
    assert shutil.disk_usage('/tmp').free > 440 * GIB
    assert digest(LOCK_SOURCE) == LOCK_SHA
    spec = importlib.util.spec_from_file_location('original_publication_lock', LOCK_SOURCE)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    os.environ.update(HF_HOME=str(ROOT / 'hf'), HF_HUB_DISABLE_XET='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_TELEMETRY='1')
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=v.pop('token'))
    assert api.whoami()['name'].lower() == 'godlydonuts'
    assert api.repo_info(REPO, repo_type='dataset').private
    completed = sorted(ROOT.glob('*.uploaded.json'))
    assert len(completed) == 7
    expected_paths = {'.gitattributes'}
    for path in completed:
        proof = json.loads(path.read_text())
        info = api.get_paths_info(REPO, paths=[proof['path']], repo_type='dataset')
        assert len(info) == 1 and info[0].size == proof['bytes']
        assert info[0].lfs.sha256 == proof['sha256']
        expected_paths.add(proof['path'])
    assert set(api.list_repo_files(REPO, repo_type='dataset')) == expected_paths
    failed_intent = ROOT / 'contexts-006.tar.upload-intent.json'
    failed = json.loads(failed_intent.read_text())
    assert failed['path'] == 'inputs/contexts-006.tar'
    assert not (ROOT / 'contexts-006.tar.uploaded.json').exists()
    assert digest(ROOT / 'contexts-006.tar') == failed['sha256']
    assert (ROOT / 'contexts-006.tar').stat().st_size == failed['bytes']
    with common.publication_lock(LOCK):
        assert not list(api.get_paths_info(REPO, paths=[failed['path']], repo_type='dataset'))
        save(ROOT / 'storage-continuation-intent.json', {
            'source_git': v['source_git'], 'control_git': v['bridge_control_git'],
            'original_failed_CPU_job': '995557', 'CPU_job': os.environ['SLURM_JOB_ID'],
            'original_completed_archives': 7, 'original_completed_bytes': 26024355840,
            'known_storage_403_reconciled': True, 'automatic_retry': False})

    manifest = {'schema': 'private_colab_predictive_contexts_v1', 'repo': REPO,
                'source_git': v['source_git'], 'base_science_git': b['base_science_git'],
                'bridge_control_git': v['bridge_control_git'],
                'original_binding_sha256': BINDING_SHA, 'context_bytes': 67005012716,
                'cohort': 3887, 'seed': 20261021, 'archives': [], 'members': {}, 'rows': [],
                'recipe': dict(arm='C', epochs=6, effective_batch=8, encoder_lr=3e-5,
                               head_lr=3e-5, weight_decay=.02, warmup_fraction=.05,
                               end_lr_fraction=.1, density_weight=100., head_only_updates=3,
                               clip=1., seed=20261021, schedule_updates=2916),
                'lineage': {k: b[k] for k in ('source_revision', 'renderer_sha256')},
                'original_fitting_order_sha256': '0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42',
                'new_seed_and_A100_runtime_not_strict_causal_comparison': True,
                'independent_clinical_validation_claim': False}

    def upload(path):
        h, size = digest(path), path.stat().st_size
        assert size < 8 * GIB
        name = 'inputs/' + path.name
        expected = {'path': name, 'bytes': size, 'sha256': h}
        proof_path = ROOT / (path.name + '.uploaded.json')
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            assert all(proof[k] == value for k, value in expected.items())
            info = api.get_paths_info(REPO, paths=[name], repo_type='dataset')[0]
            assert info.size == size and info.lfs.sha256 == h
            manifest['archives'].append(proof)
            print(json.dumps({'phase': 'reused_original_upload', 'archive': path.name,
                              'bytes': size, 'revision': proof['revision']}), flush=True)
            return
        intent = ROOT / (path.name + '.upload-intent.json')
        if intent.exists():
            assert path.name == 'contexts-006.tar'
            assert json.loads(intent.read_text()) == expected
            save(ROOT / (path.name + '.storage-continuation-upload-intent.json'), expected)
        else:
            save(intent, expected)
        with common.publication_lock(LOCK):
            assert api.repo_info(REPO, repo_type='dataset').private
            assert not list(api.get_paths_info(REPO, paths=[name], repo_type='dataset'))
            commit = api.create_commit(REPO, repo_type='dataset',
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))],
                commit_message='Private original Colab input ' + path.name)
            info = api.get_paths_info(REPO, paths=[name], repo_type='dataset', revision=commit.oid)[0]
            assert info.size == size and info.lfs and info.lfs.sha256 == h
        proof = {'path': name, 'bytes': size, 'sha256': h, 'revision': commit.oid,
                 'private': True, 'independent_Colab_download_verification_required': True}
        save(ROOT / (path.name + '.uploaded.json'), proof)
        manifest['archives'].append(proof)
        print(json.dumps({'phase': 'private_input_archive_uploaded', 'archive': path.name,
                          'bytes': size, 'revision': commit.oid}), flush=True)

    def add(tar, name, path=None, data=None, expected=None):
        if path is not None:
            before = signature(path)
            assert expected is not None
            info = tarfile.TarInfo(name); info.size = before[1]; info.mode = 0o600
            with Path(path).open('rb') as f:
                tar.addfile(info, f)
            assert signature(path) == before
            h, size = expected, before[1]
        else:
            h, size = hashlib.sha256(data).hexdigest(), len(data)
            info = tarfile.TarInfo(name); info.size = size; info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))
        assert name not in manifest['members']
        manifest['members'][name] = {'sha256': h, 'bytes': size}

    asset = ROOT / 'assets.tar'
    with Archive(asset) as tar:
        add(tar, 'parent.pt', path=parent, expected=b['parent_weight']['sha256'])
        for key, name in (('parent_source', 'parent_source.private.py'), ('targets', 'targets.private.npz'),
                          ('license', 'source-permission.txt')):
            add(tar, name, path=original(b[key]), expected=b[key]['sha256'])
        for name, h in b['code_sha256'].items():
            if name.startswith('rsna/'):
                p = Path(b['code_root']) / name
                assert digest(p) == h
                add(tar, 'code/' + name, path=p, expected=h)
        for name, body in v['own_files'].items():
            assert name in ('colab_train.py', 'colab_download.py')
            add(tar, name, data=body.encode())
    upload(asset)
    for shard, group in enumerate(shards(rows)):
        archive = ROOT / f'contexts-{shard:03d}.tar'
        with Archive(archive) as tar:
            for index, row in group:
                assert signature(row['path']) == row['proof']['stat']
                name = f'contexts/{index:06d}.npz'
                add(tar, name, path=row['path'], expected=row['proof']['sha256'])
                manifest['rows'].append({'uid': row['uid'], 'file': name,
                    'original_proof': row['proof'], 'provenance': row['provenance']})
        upload(archive)
        assert sum(p.stat().st_size for p in ROOT.glob('*.tar')) < 80 * GIB
    save(ROOT / 'manifest.private.json', manifest)
    raw = (ROOT / 'manifest.private.json').read_bytes()
    with common.publication_lock(LOCK):
        assert not list(api.get_paths_info(REPO, paths=['inputs/manifest.private.json'], repo_type='dataset'))
        commit = api.create_commit(REPO, repo_type='dataset', operations=[
            CommitOperationAdd(path_in_repo='inputs/manifest.private.json', path_or_fileobj=raw)],
            commit_message='Seal complete original 3887-study Colab inputs')
    result = {'status': 'complete_private_Colab_input_handoff_uploaded', 'CPU_job': os.environ['SLURM_JOB_ID'],
              'source_git': v['source_git'], 'repo': REPO, 'private': True, 'revision': commit.oid,
              'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'studies': 3887,
              'archives': len(manifest['archives']), 'new_shared_files': 0, 'new_GPU_jobs': 0,
              'clinical_or_Kaggle_result': False, 'Colab_independent_download_pending': True}
    save(ROOT / 'complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main(json.load(sys.stdin))
    except BaseException as error:
        tb = error.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(json.dumps({'status': 'bridge_stopped_reconcile_before_retry',
                          'error_type': type(error).__name__, 'source_line': tb.tb_lineno,
                          'http_status': getattr(getattr(error, 'response', None), 'status_code', None),
                          'server_message': getattr(error, 'server_message', None)}), flush=True)
        raise SystemExit(1) from None
