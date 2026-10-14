"""Assemble the private offline single-T4 whole-window notebook source (prepare only).

The numerical package is read from Git objects of the immutable science commit and every
file must match the original GPU-recovery stage manifest; current-tree changes to those
files are never used. The only current sources are the explicitly bound wrapper/plan files.
Private inputs (original CPU receipts, fitting-only overlay) are supplied by root from their
authorized node locations, SHA-checked and embedded only in the private notebook text.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.kaggle import window_bag_t4 as t4

ROOT = Path(__file__).resolve().parents[2]
STAGE_MANIFEST = (
    ROOT / "initiative/audits/source-window-bag-GPU-recovery-dispatch-20261013-v1/stage-intent.json"
)
PIXEL_GROUP = (
    ROOT / "initiative/audits/public-dinov2-real-inputs-20261013-v2/actual-input-comparison.json"
)
BOUND_CURRENT = ("scripts/kaggle/window_bag_t4.py", "scripts/kaggle/window_bag_plan.py")
ENTRY = ROOT / "scripts/kaggle/notebook_entry.py"
ANCHOR = '    if globals().get("WITNESS_T4_RECOVERY_PROBE") is True:\n'
SOURCE_CAP_BYTES = 1024**2
BRANCH = """    if globals().get("WINDOW_BAG_T4_PROBE") is True:
        import base64, gzip
        package = json.loads(gzip.decompress(base64.b64decode(WINDOW_BAG_PACKAGE, validate=True)))
        science_manifest = json.loads(package["manifest"])
        if hashlib.sha256(package["manifest"].encode()).hexdigest() != WINDOW_BAG_MANIFEST_SHA256:
            raise ValueError("Numerical package manifest differs")
        # Code, private inputs, capsule and MRI cache stay in session scratch; only the
        # accepted receipt or the write-once segment publication reaches /kaggle/working.
        stage = Path("/tmp/rsna-window-bag-stage")
        stage.mkdir(mode=0o700)
        code = stage / "window-bag-source"
        code.mkdir(mode=0o700)
        for name, text in package["files"].items():
            relative = Path(name)
            expected = {**science_manifest["files"], **science_manifest["bound_current"]}[name]
            if relative.is_absolute() or ".." in relative.parts or hashlib.sha256(text.encode()).hexdigest() != expected:
                raise ValueError("Numerical package path/hash differs")
            path = code / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                handle.write(text)
        private = stage / "window-bag-private"
        private.mkdir(mode=0o700)
        for name, blob in WINDOW_BAG_PRIVATE.items():
            data = gzip.decompress(base64.b64decode(blob, validate=True))
            if hashlib.sha256(data).hexdigest() != WINDOW_BAG_PRIVATE_SHA256[name]:
                raise ValueError("Private bound input differs")
            (private / name).write_bytes(data)
        (private / "source-manifest.json").write_text(package["manifest"])
        layout = WINDOW_BAG_CAPSULE
        archive = [p for p in attached_dataset_files(Path("/kaggle/input"), layout["archive_name"]) if sha256(p) == layout["archive_sha256"]]
        if len(archive) != 1:
            raise ValueError("Exactly the original private render capsule archive required")
        capsule = stage / "window-bag-capsule"
        capsule.mkdir(mode=0o700)
        with tarfile.open(archive[0]) as tar:
            tar.extractall(capsule, filter="data")
        paths = {k: capsule / v for k, v in layout["arguments"].items()}
        env.update(PYTHONPATH=str(code), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0,1" if WINDOW_BAG_PARALLEL is True else "0")
        remaining = WINDOW_BAG_TIMEOUT_SECONDS - 300 - (time.monotonic() - notebook_started)
        mode = WINDOW_BAG_MODE
        command = [str(python), "-B", str(code / "scripts/kaggle/window_bag_t4.py"), "--mode", mode, "--source", str(code), "--source-manifest", str(private / "source-manifest.json"), "--fixture", str(private / "fixture.json"), "--bag-cpu", str(private / "bag-cpu.json"), "--overlay", str(private / "overlay.json"), "--pixel-group", str(private / "pixel-group.json"), *encoder_arguments(bundle, stage, manifest), *[x for k, v in paths.items() for x in ("--" + k, str(v))], "--output", "/tmp/rsna-window-bag", "--publish-root", str(working), "--remaining-seconds", str(remaining), "--bootstrap-seconds", str(time.monotonic() - notebook_started)]
        if mode == "engineering":
            command += ["--receipt", str(working / "window-bag-engineering-accepted.json")]
            if WINDOW_BAG_PARALLEL is True:
                command += ["--parallel"]
        else:
            (private / "segment.json").write_text(json.dumps(WINDOW_BAG_SEGMENT))
            command += ["--segment", str(private / "segment.json")]
        bounded_process(command, env=env, timeout=remaining)
        return
"""


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def immutable_package(root: Path = ROOT) -> tuple[dict, dict]:
    """(files, manifest): science-commit bytes verified against the original stage manifest."""
    stage = json.loads(STAGE_MANIFEST.read_text())["source"]
    if stage["source_git"] != t4.SCIENCE_GIT:
        raise ValueError("Stage manifest names a different science commit")
    files = {}
    for name, expected in stage["files"].items():
        data = subprocess.run(
            ["git", "-C", str(root), "show", f"{t4.SCIENCE_GIT}:{name}"],
            check=True,
            capture_output=True,
        ).stdout
        if sha_bytes(data) != expected:
            raise ValueError("Science commit bytes differ from the stage manifest")
        files[name] = data.decode()
    bound = {}
    for name in BOUND_CURRENT:
        if name in stage["files"]:
            raise ValueError("Bound current sources must not replace numerical files")
        data = (root / name).read_bytes()
        bound[name] = sha_bytes(data)
        files[name] = data.decode()
    manifest = {"science_git": t4.SCIENCE_GIT, "files": stage["files"], "bound_current": bound}
    return files, manifest


def private_inputs(fixture: bytes, bag_cpu: bytes, overlay: bytes) -> dict[str, bytes]:
    if (sha_bytes(fixture), sha_bytes(bag_cpu), sha_bytes(overlay)) != (
        t4.FIXTURE_SHA256,
        t4.BAG_CPU_SHA256,
        t4.OVERLAY_SHA256,
    ):
        raise ValueError("Root-supplied private inputs differ from the original receipts")
    group = json.loads(PIXEL_GROUP.read_text())["provenance"]["member_pixel_group"]
    return {
        "fixture.json": fixture,
        "bag-cpu.json": bag_cpu,
        "overlay.json": overlay,
        "pixel-group.json": json.dumps(group).encode(),
    }


def compose(entry: str, values: dict) -> str:
    if entry.count(ANCHOR) != 1:
        raise ValueError("Pinned entry anchor must occur exactly once")
    text = entry.replace(ANCHOR, BRANCH + ANCHOR)
    header = "".join(f"{k} = {v!r}\n" for k, v in values.items())
    notebook = header + text
    compile(notebook, "window-bag-notebook.py", "exec")
    if len(notebook.encode()) >= SOURCE_CAP_BYTES:
        raise ValueError("Private notebook source exceeds its explicit cap")
    return notebook


def build(
    *,
    bundle_manifest_sha256: str,
    capsule: dict,
    fixture: bytes,
    bag_cpu: bytes,
    overlay: bytes,
    mode: str = "engineering",
    segment: dict | None = None,
    timeout_seconds: int = 7200,
    parallel: bool = False,
) -> tuple[str, dict]:
    """Return (notebook source, binding record). Writes nothing and calls no API."""
    if mode not in {"engineering", "segment"} or (mode == "segment") != (segment is not None):
        raise ValueError("Engineering, or segment with an exact contract")
    if parallel is not False and (parallel is not True or mode != "engineering"):
        raise ValueError("Concurrent two-T4 arms are an engineering-only option")
    if set(capsule) != {"archive_name", "archive_sha256", "arguments"} or set(
        capsule["arguments"]
    ) != {"studies", "artifact-root", "folds", "series"}:
        raise ValueError("Exact render-capsule layout required")
    files, manifest = immutable_package()
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    package = gzip.compress(
        json.dumps({"files": files, "manifest": manifest_text}).encode(), mtime=0
    )
    inputs = private_inputs(fixture, bag_cpu, overlay)
    values = {
        "EXPECTED_MANIFEST_SHA256": bundle_manifest_sha256,
        "WINDOW_BAG_T4_PROBE": True,
        "WINDOW_BAG_MODE": mode,
        "WINDOW_BAG_PARALLEL": parallel,
        "WINDOW_BAG_SEGMENT": segment,
        "WINDOW_BAG_TIMEOUT_SECONDS": timeout_seconds,
        "WINDOW_BAG_CAPSULE": capsule,
        "WINDOW_BAG_MANIFEST_SHA256": sha_bytes(manifest_text.encode()),
        "WINDOW_BAG_PACKAGE": base64.b64encode(package).decode(),
        "WINDOW_BAG_PRIVATE": {
            k: base64.b64encode(gzip.compress(v, mtime=0)).decode() for k, v in inputs.items()
        },
        "WINDOW_BAG_PRIVATE_SHA256": {k: sha_bytes(v) for k, v in inputs.items()},
    }
    notebook = compose(ENTRY.read_text(), values)
    record = {
        "source_sha256": sha_bytes(notebook.encode()),
        "source_bytes": len(notebook.encode()),
        "science_git": t4.SCIENCE_GIT,
        "package_manifest_sha256": values["WINDOW_BAG_MANIFEST_SHA256"],
        "bound_current": manifest["bound_current"],
        "entry_sha256": sha_bytes(ENTRY.read_bytes()),
        "mode": mode,
        "parallel": parallel,
        "segment": segment,
    }
    return notebook, record
