#!/usr/bin/env python3
"""Resume uploading the remaining RSNA files to Hugging Face via git LFS push.

This script uses a local shallow clone of the HF dataset, computes the set of
files still missing from the remote, and streams them from the Kaggle ZIP in
chunks, committing and pushing each chunk with the git CLI.  create_commit had
started returning 504 Gateway Time-out for even tiny commits once the repo grew
large; git push + LFS is stable for the same payloads.
"""
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import fsspec
import requests

# Use a newer git-lfs binary that supports lfs.transfer.batchSize
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")

REPO = "/home/ubuntu/rsna-hf-repo"
ZIP_FILES_PATH = "/home/ubuntu/zip_files.txt"
CP_PATH = "/home/ubuntu/rsna_git_upload_checkpoint.json"
LOG_PATH = "/home/ubuntu/upload_missing_git.log"
CHUNK_SIZE = int(os.environ.get("RSNA_CHUNK_SIZE", "999"))

# Pace git/LFS pushes to stay under Hugging Face's free-account API rate limit.
# One ~999-file chunk every ~330s keeps the rolling 5-minute window under the
# 1000-request free-plan limit (one objects/batch + one objects/verify per file).
TARGET_SECS_PER_FILE = float(os.environ.get("RSNA_TARGET_SECS_PER_FILE", "0.33"))


def log(msg):
    line = f"{datetime.utcnow().isoformat()}Z {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_signed_url():
    token = os.environ["KAGGLE_API_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}
    endpoint = "https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-knee-abnormality-detection"
    log("Refreshing signed URL from Kaggle...")
    r = requests.get(endpoint, headers=headers, allow_redirects=False, timeout=120)
    r.raise_for_status()
    url = r.headers["Location"]
    log(f"Got signed URL (length {len(url)}).")
    return url


def save_signed_url(url):
    cp = {}
    if os.path.exists(CP_PATH):
        cp = json.load(open(CP_PATH))
    cp["signed_url"] = url
    json.dump(cp, open(CP_PATH, "w"))


def load_signed_url():
    cp = json.load(open(CP_PATH)) if os.path.exists(CP_PATH) else {}
    return cp.get("signed_url")


def make_fs(url):
    return fsspec.filesystem(
        "zip",
        fo=url,
        target_options={
            "block_size": 100 * 1024 * 1024,
            "headers": {"Host": "storage.googleapis.com:443"},
        },
        cache_type="readahead",
    )


def get_git_files():
    proc = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True, check=True)
    return set(proc.stdout.splitlines())


def compute_missing(zip_paths, head_files):
    return [p for p in zip_paths if p.lstrip("/") not in head_files]


def write_chunk(fs, repo, chunk):
    written = []
    total = 0
    t0 = time.time()
    for p in chunk:
        data = fs.open(p, "rb").read()
        out = os.path.join(repo, p.lstrip("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        written.append(p.lstrip("/"))
        total += len(data)
    log(f"Read {len(chunk)} files ({total / 1024 / 1024:.1f} MB) in {time.time() - t0:.1f}s")
    return written


def _setup_lfs_config(repo):
    """Tune LFS to reduce API calls and fast-fail on 429s so our script can back off."""
    subprocess.run(["git", "-C", repo, "config", "lfs.locksverify", "false"], check=False)
    subprocess.run(["git", "-C", repo, "config", "lfs.transfer.batchSize", "1000"], check=False)
    subprocess.run(["git", "-C", repo, "config", "lfs.transfer.maxretries", "1"], check=False)
    subprocess.run(["git", "-C", repo, "config", "lfs.transfer.maxverifies", "1"], check=False)


def _git_env():
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Devin"
    env["GIT_AUTHOR_EMAIL"] = "devin@devin.ai"
    env["GIT_COMMITTER_NAME"] = "Devin"
    env["GIT_COMMITTER_EMAIL"] = "devin@devin.ai"
    return env


def stage_commit(repo, paths, chunk_num, max_attempts=10):
    """Stage and commit the current chunk; leave pushing to push_chunk."""
    env = _git_env()

    for attempt in range(1, max_attempts + 1):
        log(f"Staging/committing chunk {chunk_num} (attempt {attempt}/{max_attempts})...")

        # Make sure the local remote ref is as fresh as possible before committing.
        subprocess.run(
            ["git", "-C", repo, "fetch", "origin", "main"],
            check=False,
            env=env,
            timeout=60,
        )

        # Stage any files that are not yet staged (new chunk or retry after a failed push)
        subprocess.run(["git", "-C", repo, "add"] + paths, check=True, env=env, timeout=120)

        # If there are staged changes, commit them; otherwise a local commit may already exist.
        diff = subprocess.run(["git", "-C", repo, "diff", "--cached", "--quiet"], env=env, timeout=60)
        if diff.returncode == 1:
            commit_msg = f"Upload missing RSNA files chunk {chunk_num} ({len(paths)} files)"
            subprocess.run(["git", "-C", repo, "commit", "-m", commit_msg], check=True, env=env, timeout=120)
            log(f"Committed chunk {chunk_num}")
        else:
            log(f"No staged changes for chunk {chunk_num}; using existing commit")

        return

    raise RuntimeError(f"Could not stage/commit chunk {chunk_num}")


def push_commit(repo, rev, chunk_num, max_attempts=10):
    """Push <rev> to origin/main and return (push_start, push_end)."""
    env = _git_env()
    _setup_lfs_config(repo)

    # Resolve symbolic refs (e.g. HEAD) to a SHA for both push and verify.
    sha = subprocess.run(
        ["git", "-C", repo, "rev-parse", rev],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=30,
    ).stdout.strip()

    for attempt in range(1, max_attempts + 1):
        log(f"Pushing commit {sha[:12]} (chunk {chunk_num}, attempt {attempt}/{max_attempts})...")

        # Refresh the remote ref before each attempt.
        subprocess.run(
            ["git", "-C", repo, "fetch", "origin", "main"],
            check=False,
            env=env,
            timeout=60,
        )

        push_start = time.time()
        try:
            result = subprocess.run(
                ["git", "-C", repo, "push", "origin", f"{sha}:main"],
                capture_output=True,
                text=True,
                check=True,
                env=env,
                timeout=900,
            )
            push_end = time.time()
            log(result.stdout[-2000:] if result.stdout else "")
            if result.stderr:
                log(result.stderr[-2000:])
            log(f"Pushed commit {sha[:12]} in {push_end - push_start:.1f}s")
        except subprocess.CalledProcessError as e:
            log(f"Push attempt {attempt} failed: {e}")
            if e.stdout:
                log(f"push stdout: {e.stdout[-2000:]}")
            if e.stderr:
                log(f"push stderr: {e.stderr[-2000:]}")
            if attempt == max_attempts:
                raise
            err_text = (e.stdout or "") + (e.stderr or "")
            if "rate limit" in err_text.lower():
                # Back off for the HF 5-minute rate-limit window.
                log("Rate limit hit; backing off for 300s before retrying push...")
                time.sleep(300)
            else:
                time.sleep(30)
            continue

        # Verify the remote moved to the expected ref
        remote = subprocess.run(
            ["git", "-C", repo, "rev-parse", "origin/main"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=30,
        ).stdout.strip()
        if remote == sha:
            return push_start, push_end
        log(f"origin/main ({remote[:12]}) does not match {sha[:12]}; retrying push")
        if attempt == max_attempts:
            raise RuntimeError(f"Remote did not advance to {sha[:12]} after {max_attempts} attempts")
        time.sleep(30)

    raise RuntimeError(f"Could not push commit {sha[:12]}")


def push_chunk(repo, chunk_num, max_attempts=10):
    """Push HEAD to origin/main and prune local LFS objects to save disk."""
    result = push_commit(repo, "HEAD", chunk_num, max_attempts)
    prune_lfs_objects(repo)
    return result


def prune_lfs_objects(repo):
    """Remove uploaded LFS objects from the local cache after a successful push."""
    lfs_objects = os.path.join(repo, ".git", "lfs", "objects")
    if os.path.isdir(lfs_objects):
        log("Pruning local LFS object cache...")
        shutil.rmtree(lfs_objects)
    os.makedirs(lfs_objects, exist_ok=True)


def remove_working_files(repo, paths):
    for p in paths:
        try:
            os.remove(os.path.join(repo, p))
        except FileNotFoundError:
            pass
    # Prune empty directories under the repo to keep status manageable
    subprocess.run(["find", repo, "-type", "d", "-empty", "-delete"], check=False)


def main():
    log("=== Starting upload_missing_git ===")

    # Ensure we have the dataset repo
    if not os.path.isdir(REPO):
        raise RuntimeError(f"Repo {REPO} not found")

    _setup_lfs_config(REPO)

    signed_url = load_signed_url()
    if not signed_url:
        signed_url = get_signed_url()
    save_signed_url(signed_url)

    # Build fsspec view of the archive
    log("Building fsspec ZIP filesystem...")
    fs = make_fs(signed_url)

    # Load or build the full ZIP file list
    if os.path.exists(ZIP_FILES_PATH):
        log(f"Loading ZIP file list from {ZIP_FILES_PATH}")
        with open(ZIP_FILES_PATH) as f:
            zip_paths = [line.strip() for line in f if line.strip()]
    else:
        log("Listing ZIP contents (one-time)...")
        zip_paths = fs.find("/")
        with open(ZIP_FILES_PATH, "w") as f:
            for p in zip_paths:
                f.write(p + "\n")
        log(f"Saved {len(zip_paths)} ZIP paths")

    # Determine missing files from the current HEAD
    log("Loading HEAD file list...")
    head_files = get_git_files()
    missing = compute_missing(zip_paths, head_files)
    log(f"ZIP total: {len(zip_paths)}, HEAD total: {len(head_files)}, missing: {len(missing)}")

    if not missing:
        log("All files already uploaded.")
        return

    chunk_num = 0
    total_done = 0
    start_time = time.time()
    last_push_start = None

    # If a previous run left unpushed commits, flush them one at a time so each
    # push stays within the LFS free-plan rate limit.
    pending = subprocess.run(
        ["git", "-C", REPO, "rev-list", "--reverse", "origin/main..HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip().splitlines()
    if pending:
        log(f"Found {len(pending)} unpushed local commit(s); pushing first")
        prev_parent = "origin/main"
        for sha in pending:
            file_count = int(
                subprocess.run(
                    ["git", "-C", REPO, "diff", "--name-only", f"{prev_parent}..{sha}"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                ).stdout.count("\n") or 0
            ) or CHUNK_SIZE
            if last_push_start is not None:
                sleep_secs = TARGET_SECS_PER_FILE * file_count - (time.time() - last_push_start)
                if sleep_secs > 0:
                    log(f"Pacing: sleeping {sleep_secs:.1f}s before pushing pending commit")
                    time.sleep(sleep_secs)
            try:
                push_start, _ = push_commit(REPO, sha, 0)
                last_push_start = push_start
            except Exception as e:
                log(f"Pushing pending commit {sha[:12]} failed: {e}")
                raise
            prev_parent = sha

    while total_done < len(missing):
        chunk = missing[total_done : total_done + CHUNK_SIZE]
        chunk_num += 1
        log(f"Chunk {chunk_num}: {len(chunk)} files (missing index {total_done}-{total_done + len(chunk) - 1})")

        try:
            paths = write_chunk(fs, REPO, chunk)
        except Exception as e:
            log(f"Chunk {chunk_num} read failed: {e}")
            log("Sleeping 60s and refreshing signed URL...")
            time.sleep(60)
            try:
                signed_url = get_signed_url()
                fs = make_fs(signed_url)
                save_signed_url(signed_url)
            except Exception as e2:
                log(f"Could not refresh signed URL: {e2}")
            continue

        # Commit locally.  If this fails we can retry without re-reading the chunk.
        try:
            stage_commit(REPO, paths, chunk_num)
        except Exception as e:
            log(f"Chunk {chunk_num} staging/commit failed: {e}")
            log("Sleeping 60s and retrying commit...")
            time.sleep(60)
            continue

        # Pace pushes so consecutive git-lfs batch calls stay inside HF's
        # 1000 requests / 5 minutes free-plan limit.  We sleep *before* the
        # push based on the previous push start time.
        if last_push_start is not None:
            sleep_secs = TARGET_SECS_PER_FILE * len(chunk) - (time.time() - last_push_start)
            if sleep_secs > 0:
                log(f"Pacing: sleeping {sleep_secs:.1f}s before next push")
                time.sleep(sleep_secs)

        # Push.  On rate-limit errors push_chunk backs off internally.
        try:
            push_start, _ = push_chunk(REPO, chunk_num)
            last_push_start = push_start
        except Exception as e:
            log(f"Chunk {chunk_num} push failed: {e}")
            log("Sleeping 60s and retrying push...")
            time.sleep(60)
            continue

        remove_working_files(REPO, paths)
        total_done += len(chunk)

        elapsed = time.time() - start_time
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = (len(missing) - total_done) / rate if rate > 0 else 0
        log(
            f"Progress: {total_done}/{len(missing)} files ({100 * total_done / len(missing):.2f}%) "
            f"at {rate:.2f} files/s, ETA {remaining / 3600:.1f}h"
        )

        # Update checkpoint
        cp = {"signed_url": signed_url, "total_done": total_done, "missing_count": len(missing)}
        json.dump(cp, open(CP_PATH, "w"))

    log(f"All {len(missing)} missing files uploaded.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Fatal error: {e}")
        raise
