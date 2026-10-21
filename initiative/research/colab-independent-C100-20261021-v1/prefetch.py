"""Overlap private archive download with the original single-writer upload.

Only warms the ordinary HF content-addressed cache. Original downloader still
requires the sealed manifest and independently checks every archive/member SHA
before any extraction is accepted or any model training starts.
"""
import json
import time


def prefetch(token, stop):
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'GodlyDonuts/RSNA-Colab-Deadline-20261021'
    api = HfApi(token=token)
    downloaded = set()
    deadline = time.monotonic() + 7200
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            info = api.repo_info(repo, repo_type='dataset')
            assert info.private
            files = api.list_repo_files(repo, repo_type='dataset', revision=info.sha)
            for name in sorted(files):
                if name.startswith('inputs/') and name.endswith('.tar') and name not in downloaded:
                    hf_hub_download(repo, name, repo_type='dataset', revision=info.sha, token=token)
                    downloaded.add(name)
                    print(json.dumps({'phase':'private_archive_prefetched_verification_pending',
                                      'archive':name}), flush=True)
                if stop.is_set():
                    return
        except Exception as error:
            print(json.dumps({'phase':'prefetch_unavailable_main_verifier_remains_authoritative',
                              'error_type':type(error).__name__}), flush=True)
            return
        stop.wait(30)
