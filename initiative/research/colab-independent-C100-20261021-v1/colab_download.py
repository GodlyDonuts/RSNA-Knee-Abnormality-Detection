"""Private immutable Colab input download; no saved token, unsafe extraction or retries of writes."""

import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

REPO = 'GodlyDonuts/RSNA-Colab-Deadline-20261021'
ROOT = Path('/content/rsna-independent-C100-20261021-v1')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(token, expected_source_git):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    os.umask(0o077)
    os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_TELEMETRY='1')
    assert not ROOT.exists(), 'Existing original Colab inputs require reconciliation'
    assert shutil.disk_usage('/content').free > 170 * 2**30
    api = HfApi(token=token)
    deadline = time.monotonic() + 7200
    while True:
        try:
            info = api.repo_info(REPO, repo_type='dataset')
            assert info.private
            manifest_path = hf_hub_download(REPO, 'inputs/manifest.private.json',
                repo_type='dataset', revision=info.sha, token=token)
            break
        except (EntryNotFoundError, RepositoryNotFoundError):
            if time.monotonic() >= deadline:
                raise TimeoutError('Original private handoff incomplete; no training started') from None
            time.sleep(30)
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest['source_git'] == expected_source_git and manifest['cohort'] == 3887
    assert manifest['repo'] == REPO and manifest['schema'] == 'private_colab_predictive_contexts_v1'
    ROOT.mkdir(mode=0o700)
    seen = set()
    for archive in manifest['archives']:
        path = hf_hub_download(REPO, archive['path'], repo_type='dataset',
                               revision=info.sha, token=token)
        assert Path(path).stat().st_size == archive['bytes'] and sha(path) == archive['sha256']
        with tarfile.open(path, 'r:') as tar:
            for member in tar:
                name = member.name
                parts = PurePosixPath(name).parts
                assert member.isfile() and not member.issym() and not member.islnk()
                assert parts and '..' not in parts and not PurePosixPath(name).is_absolute()
                assert name in manifest['members'] and name not in seen
                expected = manifest['members'][name]
                assert member.size == expected['bytes']
                dest = ROOT / name
                dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                digest = hashlib.sha256(); total = 0
                with tar.extractfile(member) as src, dest.open('xb') as out:
                    while chunk := src.read(8 * 2**20):
                        total += len(chunk); digest.update(chunk); out.write(chunk)
                assert total == member.size and digest.hexdigest() == expected['sha256']
                seen.add(name)
        print(json.dumps({'phase': 'original_input_archive_independently_verified',
                          'archive': archive['path'], 'members_completed': len(seen)}), flush=True)
    assert seen == set(manifest['members'])
    assert len(manifest['rows']) == 3887
    manifest['actual_private_revision'] = info.sha
    manifest['all_original_member_bytes_SHA_verified_on_Colab'] = True
    manifest['downloaded_manifest_sha256'] = sha(manifest_path)
    with (ROOT / 'input-verified.private.json').open('x') as f:
        json.dump(manifest, f, sort_keys=True)
    print(json.dumps({'phase': 'complete_original_inputs_verified', 'studies': 3887,
                      'source_git': expected_source_git, 'revision': info.sha}), flush=True)
    return ROOT
