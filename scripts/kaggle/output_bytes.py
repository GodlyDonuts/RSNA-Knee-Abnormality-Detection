"""Bounded exact-version private output verification, without saving payloads."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urljoin, urlparse

import requests

API = "https://api.kaggle.com/v1/"


def verify(remote, slug, version, name, expected):
    try:
        return _verify(remote, slug, version, name, expected)
    except requests.RequestException:
        raise ValueError("Original output transport failed") from None


def _verify(remote, slug, version, name, expected):
    size, sha = expected["bytes"], expected["sha256"]
    if (
        not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9-]+", slug)
        or type(version) is not int
        or version < 1
        or not isinstance(name, str)
        or name.startswith("/")
        or any(p in ("", ".", "..") for p in name.split("/"))
        or type(size) is not int
        or not 0 <= size <= 2 * 1024**2
        or not re.fullmatch(r"[a-f0-9]{64}", sha)
    ):
        raise ValueError("Bounded exact-version original output contract required")
    owner, kernel = slug.split("/")
    remote.session.cookies.clear()
    response = remote.session.post(
        API + "kernels.KernelsApiService/DownloadKernelOutput",
        json={"ownerSlug": owner, "kernelSlug": kernel, "versionNumber": version, "filePath": name},
        timeout=(10, 60),
        allow_redirects=False,
        stream=True,
    )
    for _ in range(5):
        with response:
            if response.status_code == 200:
                h = hashlib.sha256()
                count = 0
                for block in response.iter_content(65536):
                    count += len(block)
                    if count > size:
                        raise ValueError("Original output exceeds its bound")
                    h.update(block)
                if count != size or h.hexdigest() != sha:
                    raise ValueError("Original output bytes differ")
                return {"bytes": count, "sha256": h.hexdigest(), "full_stream_verified": True}
            if response.status_code not in (301, 302, 303, 307, 308):
                raise ValueError("Original output is unavailable")
            url = urljoin(response.url, response.headers.get("Location", ""))
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (
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
    raise ValueError("Too many original output redirects")
