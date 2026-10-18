"""Exercise original stream code with synthetic data and mocked node/quota IO.

Only pinned fixture hashes/paths and cluster infrastructure are replaced. The
actual sender/receiver validation/write logic runs; no original MRI bytes load.
"""

import copy
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

D = Path(__file__).resolve().parent
sender = (D / "fixture_sender.py.txt").read_text()
receiver = (D / "fixture_receiver.py.txt").read_text()


def sha(data):
    return hashlib.sha256(data).hexdigest()


membership = {
    "fitting": [f"fit-{j:04}" for j in range(2247)],
    "outer": [f"outer-{j:04}" for j in range(1156)],
}
contents = {f"synthetic-{j:02}.npz": f"synthetic, non-MRI bytes{j}".encode() for j in range(32)}
rows = [
    {
        "file": name,
        "study_uid": membership["fitting" if j < 16 else "outer"][j % 16],
        "role": "fitting" if j < 16 else "outer",
        "windows": 74 if j < 9 else 73,
        "bytes": len(data),
        "sha256": sha(data),
    }
    for j, (name, data) in enumerate(contents.items())
]
raw_index = json.dumps({"private_records": rows}).encode()
raw_membership = json.dumps(membership).encode()
raw_parent = json.dumps(
    {
        "binding": {
            "release": {
                "manifest_sha256": "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7"
            }
        }
    }
).encode()
metadata = [
    ("pilot-recovery-v2-accepted.json", raw_index),
    ("membership.private.json", raw_membership),
    ("parent-run-binding.json", raw_parent),
]
features = [("features/" + name, data) for name, data in contents.items()]
original_index_sha = "f058c0969c56d754874ceb3fbb1fb9c3fff2d2a5c9bb1554d94cf68d9a3bd18f"
roster_sha = {role: sha(json.dumps(ids).encode()) for role, ids in membership.items()}
old_rosters = {
    "fitting": "850f170598e5ddd09a80317950acba29c03799f1a620eb2d4d9793d22a2f4e74",
    "outer": "2983434d353efee3b3f13af5bb8cb7520e43f8fc23c329cfd0351def93da62cd",
}


def packet(members, symlink=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|") as archive:
        for j, (name, data) in enumerate(members):
            info = tarfile.TarInfo(name)
            if symlink and j == 3:
                info.type, info.linkname = tarfile.SYMTYPE, "/tmp/outside"
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


with tempfile.TemporaryDirectory(prefix="rsna-synthetic-private-stream-") as tmp:
    tmp = Path(tmp)
    source_root = tmp / "source"
    (source_root / "features").mkdir(parents=True)
    for name, data in metadata[:2] + features:
        (source_root / name).write_bytes(data)
    parent = tmp / "parent.json"
    parent.write_bytes(raw_parent)
    source_code = (
        sender.replace("/tmp/rsna-criterion-N3-features-20261017-v1", str(source_root))
        .replace(
            "/tmp/rsna-compact-native-n3-production-evc33-20261016-v1/production/run.json",
            str(parent),
        )
        .replace(original_index_sha, sha(raw_index))
    )
    wire = io.BytesIO()
    with (
        patch("os.uname", return_value=SimpleNamespace(nodename="evc33")),
        patch("shutil.disk_usage", return_value=SimpleNamespace(free=10**15)),
        patch("sys.stdout", SimpleNamespace(buffer=wire)),
        patch.dict(os.environ, {"SLURM_JOB_GPUS": ""}),
    ):
        exec(compile(source_code, "actual_sender_with_synthetic_infrastructure", "exec"), {})
    valid_wire = wire.getvalue()

    def receive(data, name, *, existing=False, tiny_cap=False):
        root = tmp / name
        if existing:
            root.mkdir()
            (root / "sentinel").write_text("keep")
        code = (
            receiver.replace(
                "import torch,numpy,pandas\nassert str(torch.__version__)=='2.14.0+cu130' and numpy.__version__=='2.2.1' and pandas.__version__=='2.2.3'\n",
                "",
            )
            .replace("/tmp/rsna-path-recovery-evc48-20261018-v3", str(root))
            .replace(original_index_sha, sha(raw_index))
        )
        for role, old in old_rosters.items():
            code = code.replace(old, roster_sha[role])
        if tiny_cap:
            code = code.replace("cap = 3 * 2**30", "cap = 128 * 2**20")
        stdout = io.StringIO()
        before_umask = os.umask(0o077)
        try:
            with (
                patch("os.uname", return_value=SimpleNamespace(nodename="evc48")),
                patch("shutil.disk_usage", return_value=SimpleNamespace(free=10**15)),
                patch(
                    "subprocess.check_output",
                    return_value="/home/sa305415 1 1000000000 1000000000 - 1 10000000 10000000 -\n",
                ),
                patch("sys.stdin", SimpleNamespace(buffer=io.BytesIO(data))),
                patch("sys.stdout", stdout),
                patch("sys.argv", ["-c", "synthetic-source", sha(code.encode())]),
                patch(
                    "sys.orig_argv",
                    ["python", "-B", "-c", code, "synthetic-source", sha(code.encode())],
                ),
                patch.dict(os.environ, {"SLURM_JOB_GPUS": "", "SLURM_JOB_ID": "SYNTHETIC"}),
            ):
                exec(compile(code, "actual_receiver_with_synthetic_infrastructure", "exec"), {})
        finally:
            os.umask(before_umask)
        return root, json.loads(stdout.getvalue())

    root, proof = receive(valid_wire, "valid")
    assert proof["feature_files_fully_stream_download_size_SHA_verified"] == 32
    for name, data in features:
        assert (root / name).read_bytes() == data and (root / name).stat().st_mode & 0o077 == 0
    assert root.stat().st_mode & 0o077 == 0
    corrupt = copy.deepcopy(features)
    corrupt[0] = (corrupt[0][0], b"X" * len(corrupt[0][1]))
    wrong_index = [(metadata[0][0], raw_index + b" "), *metadata[1:]]
    escaped = [("../outside", features[0][1]), *features[1:]]
    oversized = copy.deepcopy(features)
    oversized[0] = (oversized[0][0], oversized[0][1] + b"X")
    cases = [
        (packet(metadata + features + features[:1]), {}),
        (packet(metadata + corrupt), {}),
        (packet(wrong_index + features), {}),
        (packet(metadata + escaped), {}),
        (packet(metadata + oversized), {}),
        (packet(metadata + features, symlink=True), {}),
        (packet(metadata + features[:-1]), {}),
        (packet(metadata + features + [("unexpected", b"X")]), {}),
        (valid_wire, {"existing": True}),
        (valid_wire, {"tiny_cap": True}),
    ]
    refused = 0
    for j, (payload, options) in enumerate(cases):
        try:
            receive(payload, f"refusal-{j}", **options)
        except (AssertionError, ValueError, tarfile.TarError):
            refused += 1
        else:
            raise AssertionError("Malformed private packet accepted")
    assert refused == 10 and not (tmp / "outside").exists()
    assert (tmp / "refusal-8/sentinel").read_text() == "keep"

result = {
    "status": "actual_private_fixture_stream_logic_synthetic_positive_and10negative_checks_passed",
    "sender_sha256": sha(sender.encode()),
    "receiver_sha256": sha(receiver.encode()),
    "synthetic_members": 35,
    "synthetic_case_roster_counts": [2247, 1156],
    "all32_synthetic_file_bytes_equal": True,
    "owner_only_permissions_and_existing_root_preservation": True,
    "refused": 10,
    "scope": "Actual sender/receiver code with synthetic private-index hashes, paths and mocked cluster node/quota/stdin. No original MRI data, cluster dispatch or GPU acceptance.",
}
(D / "fixture-transport-checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
