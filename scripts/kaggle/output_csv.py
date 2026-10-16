"""Verify a versioned public-run CSV in RAM without exporting study records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from urllib.parse import urljoin, urlparse

import requests

API = "https://api.kaggle.com/v1/"


def verify_csv(content: bytes, expected_columns: list[str], roster_sha256: str) -> dict:
    if (
        not content
        or len(content) > 1024**2
        or len(expected_columns) != 13
        or len(set(expected_columns)) != 13
        or expected_columns[0] != "StudyInstanceUID"
    ):
        raise ValueError("Bounded twelve-target public CSV required")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    if reader.fieldnames != expected_columns:
        raise ValueError("Exact target/header order required")
    identifiers = []
    for row in reader:
        if set(row) != set(expected_columns) or any(row[k] is None for k in expected_columns):
            raise ValueError("Complete CSV row required")
        uid = row[expected_columns[0]]
        if not uid or uid in identifiers:
            raise ValueError("Distinct complete study identities required")
        values = [float(row[k]) for k in expected_columns[1:]]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("Finite probability values required")
        identifiers.append(uid)
    digest = hashlib.sha256(json.dumps(identifiers, separators=(",", ":")).encode()).hexdigest()
    if not identifiers or digest != roster_sha256:
        raise ValueError("Exact public test.csv roster required")
    return {
        "status": "versioned_public_CSV_verified_in_RAM",
        "studies": len(identifiers),
        "columns": len(expected_columns),
        "bytes": len(content),
        "CSV_sha256": hashlib.sha256(content).hexdigest(),
        "test_roster_sha256": digest,
        "individual_records_exported": False,
        "clinical_or_hidden_score_inferred": False,
    }


def download_csv(remote, slug: str, version: int) -> bytes:
    if type(version) is not int or version < 1 or len(slug.split("/")) != 2:
        raise ValueError("Exact notebook owner/slug and version required")
    owner, kernel = slug.split("/")
    try:
        remote.session.cookies.clear()
        response = remote.session.post(
            API + "kernels.KernelsApiService/DownloadKernelOutput",
            json={
                "ownerSlug": owner,
                "kernelSlug": kernel,
                "versionNumber": version,
                "filePath": "submission.csv",
            },
            timeout=(10, 60),
            allow_redirects=False,
            stream=True,
        )
        for _ in range(5):
            with response:
                if response.status_code == 200:
                    content = bytearray()
                    for block in response.iter_content(65536):
                        content.extend(block)
                        if len(content) > 1024**2:
                            raise ValueError("Public CSV exceeds bounded size")
                    return bytes(content)
                if response.status_code not in {301, 302, 303, 307, 308}:
                    raise ValueError("Versioned public CSV not available")
                url = urljoin(response.url, response.headers.get("Location", ""))
            host = urlparse(url).hostname or ""
            if urlparse(url).scheme != "https" or not (
                host
                in {
                    "storage.googleapis.com",
                    "api.kaggle.com",
                    "www.kaggle.com",
                    "www.kaggleusercontent.com",
                }
                or host.endswith(".googleusercontent.com")
            ):
                raise ValueError("Unrecognized output redirect host")
            transport = remote.session if host == "api.kaggle.com" else requests
            response = transport.get(url, timeout=(10, 60), allow_redirects=False, stream=True)
        response.close()
        raise ValueError("Too many output redirects")
    except (requests.RequestException, UnicodeError, ValueError):
        raise ValueError("Versioned public CSV verification failed") from None
