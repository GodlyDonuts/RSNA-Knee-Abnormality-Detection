"""Build the private paired two-T4 production segment notebook; never pushes anything.

The package is the exact concurrent engineering package (science commit ``971d7dac``
numerical files plus ``window_bag_t4.py``/``window_bag_plan.py`` bytes) extended only by
``window_bag_paired.py``. The engineering manifest's hash is embedded in the paired
manifest and re-verified inside the session against the segment contract, so a paired
segment cannot run a package other than the one accepted on the actual two T4s.
"""

from __future__ import annotations

import base64
import gzip
import json

from scripts.kaggle import window_bag_notebook as nb
from scripts.kaggle import window_bag_paired as wp
from scripts.kaggle import window_bag_plan as plan

MODE = "paired_segment"
_HEAD, _TAIL = nb.BRANCH.split('        if mode == "engineering":\n', 1)
BRANCH = (
    _HEAD.replace("WINDOW_BAG_T4_PROBE", "WINDOW_BAG_PAIRED_PROBE").replace(
        'str(code / "scripts/kaggle/window_bag_t4.py"), "--mode", mode, ',
        'str(code / "scripts/kaggle/window_bag_paired.py"), ',
    )
    + '        if mode != "paired_segment" or WINDOW_BAG_PARALLEL is not True:\n'
    '            raise ValueError("Paired production segment only")\n'
    '        (private / "segment.json").write_text(json.dumps(WINDOW_BAG_SEGMENT))\n'
    '        command += ["--segment", str(private / "segment.json")]\n'
    + _TAIL[_TAIL.index("        bounded_process(") :]
)


def paired_package(root=nb.ROOT) -> tuple[dict, dict, str]:
    """(files, manifest, engineering manifest SHA): the engineering package plus one file."""
    files, engineering = nb.immutable_package(root)
    text = json.dumps(engineering, sort_keys=True, indent=2) + "\n"
    data = (root / wp.PAIRED_SOURCE).read_bytes()
    files[wp.PAIRED_SOURCE] = data.decode()
    manifest = {
        **engineering,
        "bound_current": {**engineering["bound_current"], wp.PAIRED_SOURCE: nb.sha_bytes(data)},
        "engineering_package_manifest_sha256": nb.sha_bytes(text.encode()),
    }
    if wp.engineering_manifest_text(manifest) != text:
        raise ValueError("Paired package does not extend the engineering package exactly")
    return files, manifest, manifest["engineering_package_manifest_sha256"]


def compose(entry: str, values: dict) -> str:
    if entry.count(nb.ANCHOR) != 1:
        raise ValueError("Pinned entry anchor must occur exactly once")
    notebook = "".join(f"{k} = {v!r}\n" for k, v in values.items()) + entry.replace(
        nb.ANCHOR, BRANCH + nb.ANCHOR
    )
    compile(notebook, "window-bag-paired-notebook.py", "exec")
    if len(notebook.encode()) >= nb.SOURCE_CAP_BYTES:
        raise ValueError("Private notebook source exceeds its explicit cap")
    return notebook


def build(
    *,
    bundle_manifest_sha256: str,
    capsule: dict,
    fixture: bytes,
    bag_cpu: bytes,
    overlay: bytes,
    segment: dict,
    timeout_seconds: int,
) -> tuple[str, dict]:
    """Return (notebook source, binding record). Writes nothing and calls no API."""
    wp.paired_contract(segment)
    if not 0 < timeout_seconds <= plan.SESSION_LIMIT_SECONDS * plan.SESSION_MARGIN:
        raise ValueError("Paired session timeout exceeds the nine-hour margin")
    if set(capsule) != {"archive_name", "archive_sha256", "arguments"} or set(
        capsule["arguments"]
    ) != {"studies", "artifact-root", "folds", "series"}:
        raise ValueError("Exact render-capsule layout required")
    files, manifest, engineering = paired_package()
    if segment["engineering_package_manifest_sha256"] != engineering:
        raise ValueError("Segment names a different engineering package")
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    package = gzip.compress(
        json.dumps({"files": files, "manifest": manifest_text}).encode(), mtime=0
    )
    inputs = nb.private_inputs(fixture, bag_cpu, overlay)
    values = {
        "EXPECTED_MANIFEST_SHA256": bundle_manifest_sha256,
        "WINDOW_BAG_PAIRED_PROBE": True,
        "WINDOW_BAG_MODE": MODE,
        "WINDOW_BAG_PARALLEL": True,
        "WINDOW_BAG_SEGMENT": segment,
        "WINDOW_BAG_TIMEOUT_SECONDS": timeout_seconds,
        "WINDOW_BAG_CAPSULE": capsule,
        "WINDOW_BAG_MANIFEST_SHA256": nb.sha_bytes(manifest_text.encode()),
        "WINDOW_BAG_PACKAGE": base64.b64encode(package).decode(),
        "WINDOW_BAG_PRIVATE": {
            k: base64.b64encode(gzip.compress(v, mtime=0)).decode() for k, v in inputs.items()
        },
        "WINDOW_BAG_PRIVATE_SHA256": {k: nb.sha_bytes(v) for k, v in inputs.items()},
    }
    notebook = compose(nb.ENTRY.read_text(), values)
    return notebook, {
        "source_sha256": nb.sha_bytes(notebook.encode()),
        "source_bytes": len(notebook.encode()),
        "science_git": manifest["science_git"],
        "package_manifest_sha256": values["WINDOW_BAG_MANIFEST_SHA256"],
        "engineering_package_manifest_sha256": engineering,
        "bound_current": manifest["bound_current"],
        "entry_sha256": nb.sha_bytes(nb.ENTRY.read_bytes()),
        "mode": MODE,
        "parallel": True,
        "segment": segment,
    }
