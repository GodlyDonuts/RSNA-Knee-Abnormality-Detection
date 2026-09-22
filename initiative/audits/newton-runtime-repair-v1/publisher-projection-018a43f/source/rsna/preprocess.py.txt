from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import requests
import torch
from fsspec.spec import AbstractFileSystem
from requests.adapters import HTTPAdapter
from urllib3.response import BaseHTTPResponse
from urllib3.util.retry import Retry

from rsna.dicom import decode_slice, make_volume
from rsna.schema import CACHE_VERSION

COMPETITION = "rsna-knee-abnormality-detection"


class HubRetry(Retry):
    def get_retry_after(self, response: BaseHTTPResponse) -> float | None:
        retry_after = super().get_retry_after(response)
        if retry_after is not None or response.status != 429:
            return retry_after
        reset = re.search(r"(?:^|;)\s*t=(\d+)", response.headers.get("RateLimit", ""))
        return float(reset[1]) + 1 if reset else 300


class HttpSessions(threading.local):
    def __init__(self) -> None:
        self.session = requests.Session()
        retry = HubRetry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))


HTTP = HttpSessions()


class Source:
    def __init__(
        self,
        root: Path | None,
        lfs_objects: dict | None = None,
        repo_id: str = "GodlyDonuts/RSNA-Data",
        revision: str = "main",
    ):
        self.root = root
        self.fs: AbstractFileSystem | None = None
        self.lfs_objects = lfs_objects
        self.repo_id = repo_id
        self.revision = revision
        token = os.environ.get("HF_TOKEN")
        self.hf_headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.downloads = ThreadPoolExecutor(max_workers=32)

    def read_series(self, paths: list[str]) -> list[bytes]:
        if self.lfs_objects is None:
            return [self.read(path) for path in paths]
        objects = [self.lfs_objects[path] for path in paths]
        lfs_batch = [
            {"oid": obj["sha256"], "size": obj["bytes"]}
            for obj in objects
            if obj["storage"] == "lfs"
        ]
        actions = {}
        if lfs_batch:
            response = HTTP.session.post(
                f"https://huggingface.co/datasets/{self.repo_id}.git/info/lfs/objects/batch",
                headers={
                    "Accept": "application/vnd.git-lfs+json",
                    "Content-Type": "application/vnd.git-lfs+json",
                    **self.hf_headers,
                },
                json={"operation": "download", "transfers": ["basic"], "objects": lfs_batch},
                timeout=120,
            )
            response.raise_for_status()
            actions = {obj["oid"]: obj for obj in response.json()["objects"]}

        def download(item: tuple[str, dict]) -> bytes:
            path, obj = item
            if obj["storage"] == "lfs":
                action = actions[obj["sha256"]]["actions"]["download"]
            else:
                action = {
                    "href": f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/{path}",
                    "header": self.hf_headers,
                }
            for attempt in range(3):
                try:
                    result = HTTP.session.get(
                        action["href"], headers=action.get("header", {}), timeout=120
                    )
                    result.raise_for_status()
                    if (
                        len(result.content) != obj["bytes"]
                        or hashlib.sha256(result.content).hexdigest() != obj["sha256"]
                    ):
                        raise ValueError("Downloaded bytes fail LFS checksum/size verification")
                    return result.content
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise
            raise RuntimeError("Unreachable")

        return list(self.downloads.map(download, zip(paths, objects)))

    def connect(self) -> None:
        token = os.environ["KAGGLE_API_TOKEN"]
        response = requests.get(
            f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}",
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=False,
            timeout=120,
        )
        response.raise_for_status()
        self.fs = fsspec.filesystem(
            "zip",
            fo=response.headers["Location"],
            target_options={"block_size": 32 * 1024 * 1024},
            cache_type="readahead",
        )

    def read(self, path: str) -> bytes:
        if self.root is not None:
            return (self.root / path).read_bytes()
        for attempt in range(3):
            try:
                if self.fs is None:
                    self.connect()
                if self.fs is None:
                    raise RuntimeError("Source connection did not initialize a filesystem")
                with self.fs.open(path, "rb") as handle:
                    return handle.read()
            except Exception:
                if attempt == 2:
                    raise
                self.fs = None
        raise RuntimeError("Unreachable")


def cache_series(
    source: Source,
    paths: list[str],
    output: Path,
    side: int,
    source_revision: str,
    min_free_gib: float = 3.0,
) -> dict:
    if not np.isfinite(min_free_gib) or min_free_gib < 3:
        raise ValueError("Disk reserve must be at least 3 GiB")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("A cache series requires nonempty unique source paths")
    first = Path(paths[0]).parts
    if len(first) != 4 or first[0] not in {"train_series", "test_series"}:
        raise ValueError("Invalid DICOM source path")
    split, study_uid, series_uid = first[:3]
    if any(
        Path(path).parts[:3] != first[:3]
        or len(Path(path).parts) != 4
        or Path(path).suffix != ".dcm"
        for path in paths
    ):
        raise ValueError("Source paths do not identify a single DICOM series")
    folder = output / split / study_uid
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{series_uid}.npz"
    metadata_path = destination.with_suffix(".json")
    fingerprint = hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()
    if destination.exists() != metadata_path.exists():
        raise ValueError(f"Incomplete cache pair: {destination}; inspect before restarting")
    if destination.exists() and metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if (
            meta["cache_version"] == CACHE_VERSION
            and meta["side"] == side
            and meta["path_fingerprint"] == fingerprint
            and meta["source_revision"] == source_revision
            and (meta["split"], meta["study_uid"], meta["series_uid"])
            == (split, study_uid, series_uid)
            and meta["n_slices"] == len(paths)
            and set(meta["sop_uids"]) == {Path(path).stem for path in paths}
            and meta["cache_bytes"] == destination.stat().st_size
            and hashlib.sha256(destination.read_bytes()).hexdigest() == meta["cache_sha256"]
        ):
            return meta
        raise ValueError(f"Stale/corrupted cache: {destination}; use a fresh output directory")
    if shutil.disk_usage(output).free < min_free_gib * 1024**3:
        raise OSError(f"Less than {min_free_gib:g} GiB free: pause rather than filling the disk")
    source_data = source.read_series(paths)
    if len(source_data) != len(paths):
        raise ValueError("Downloaded series has incomplete source coverage")
    slices = [
        decode_slice(data, study_uid, series_uid, Path(path).stem)
        for path, data in zip(paths, source_data)
    ]
    del source_data
    volume, info = make_volume(slices, side)
    temporary = destination.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, volume=volume)
    temporary.replace(destination)
    info.update(
        {
            "cache_version": CACHE_VERSION,
            "source_revision": source_revision,
            "path_fingerprint": fingerprint,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "split": split,
            "side": side,
            "n_slices": len(slices),
            "cache_path": str(destination.relative_to(output)),
            "cache_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "cache_bytes": destination.stat().st_size,
        }
    )
    temporary_meta = metadata_path.with_suffix(".partial.json")
    temporary_meta.write_text(json.dumps(info, sort_keys=True))
    temporary_meta.replace(metadata_path)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--limit-series", type=int)
    parser.add_argument("--study-csv", type=Path)
    parser.add_argument("--source", choices=["kaggle", "hf"], default="hf")
    parser.add_argument("--series-workers", type=int, default=4)
    parser.add_argument("--min-free-gib", type=float, default=3.0)
    args = parser.parse_args()
    if args.series_workers < 1:
        raise ValueError("--series-workers must be positive")
    if args.limit_series is not None and args.limit_series < 1:
        raise ValueError("--limit-series must be positive")
    if not np.isfinite(args.min_free_gib) or args.min_free_gib < 3:
        raise ValueError("--min-free-gib must be at least 3")
    torch.set_num_threads(2)
    manifest = pd.read_parquet(args.manifest)
    if args.study_csv:
        ids = set(pd.read_csv(args.study_csv)["study_uid"])
        manifest = manifest[manifest.study_uid.isin(ids)]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source == "kaggle" and args.series_workers != 1:
        raise ValueError("Kaggle ZIP streaming requires --series-workers 1")
    objects = None
    if args.source == "hf" and args.root is None:
        objects = manifest.set_index("path")[["sha256", "bytes", "storage"]].to_dict("index")
    source = Source(args.root, objects, revision=args.source_revision)
    failures = 0
    consecutive_failures = 0
    start = time.monotonic()
    total_slices = 0
    log_path = args.output / "progress.jsonl"
    groups = list(manifest.groupby(["split", "study_uid", "series_uid"], sort=True))
    if args.limit_series is not None:
        groups = groups[: args.limit_series]
    with log_path.open("a", buffering=1) as log, ThreadPoolExecutor(args.series_workers) as pool:
        pending = {}
        for index, (keys, frame) in enumerate(groups):
            pending[index] = pool.submit(
                cache_series,
                source,
                frame.path.tolist(),
                args.output,
                args.side,
                args.source_revision,
                args.min_free_gib,
            )
            if len(pending) < args.series_workers * 2 and index < len(groups) - 1:
                continue
            while pending and (len(pending) >= args.series_workers * 2 or index == len(groups) - 1):
                done_index = min(pending)
                future = pending.pop(done_index)
                done_keys = groups[done_index][0]
                try:
                    result = future.result()
                except Exception as exc:
                    failures += 1
                    consecutive_failures += 1
                    event = {
                        "status": "error",
                        "study_uid": done_keys[1],
                        "series_uid": done_keys[2],
                        "error_type": type(exc).__name__,
                    }
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        event["http_status"] = exc.response.status_code
                    if isinstance(exc, ValueError):
                        event["detail"] = str(exc)
                    log.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                    if (
                        isinstance(exc, OSError) and not isinstance(exc, requests.RequestException)
                    ) or consecutive_failures >= 5:
                        for remaining in pending.values():
                            remaining.cancel()
                        raise RuntimeError(json.dumps(event)) from None
                    continue
                consecutive_failures = 0
                total_slices += result["n_slices"]
                event = {
                    "status": "ok",
                    "study_uid": done_keys[1],
                    "series_uid": done_keys[2],
                    "n_slices": result["n_slices"],
                    "cache_bytes": result["cache_bytes"],
                    "warnings": result["warnings"],
                }
                log.write(json.dumps(event) + "\n")
                if done_index % 25 == 0:
                    print(
                        f"series={done_index + 1}/{len(groups)} slices={total_slices} "
                        f"seconds={time.monotonic() - start:.1f}",
                        flush=True,
                    )
    print(f"Finished: {total_slices} decoded slices; {failures} failed series", flush=True)
    source.downloads.shutdown()
    if failures:
        raise RuntimeError("Incomplete cache: inspect progress.jsonl")


if __name__ == "__main__":
    main()
