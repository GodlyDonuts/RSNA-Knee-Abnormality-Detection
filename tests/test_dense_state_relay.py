"""Once-only private dense state/runtime relay for the PR76 allocation transition (#80)."""

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dense_suffix_production import write
from test_dense_suffix_transition import admit, fund, record, stop_original
from test_dense_suffix_transition import env as env  # noqa: F401  (fixture)
from test_dense_suffix_transition import t as t  # noqa: F401  (fixture)

from scripts.newton import dense_state_relay as relay
from scripts.newton import transition_dense_suffix_allocation as tr
from scripts.newton import triplet_cache_relay as triplet

REPO = Path(__file__).resolve().parents[1]
TRIPLET_AUDIT = REPO / "initiative/audits/dense-triplet-cache-relay-20261014-v1"
UIDS = ["1.2.826.0.1.1", "1.2.826.0.1.2"]
GIB = 2**30
HEADER = len(relay.MAGIC) + 32 + 16


def npz(uid):
    return hashlib.sha256(uid.encode()).digest() * 2


@pytest.fixture
def s(t, tmp_path, monkeypatch):  # noqa: F811
    cache = t.cache
    (cache / triplet.MEMBERS_DIR).mkdir(parents=True)
    members = {}
    for uid in UIDS:
        (cache / triplet.MEMBERS_DIR / f"{uid}.npz").write_bytes(npz(uid))
        members[uid] = {
            "path": f"members/{uid}.npz",
            "bytes": 64,
            "sha256": hashlib.sha256(npz(uid)).hexdigest(),
        }
    (cache / triplet.DIRECTORY / "finalized.json").write_text("{}")  # producer bookkeeping
    index = {
        "schema_version": 1,
        "kind": "k",
        "renderer": "r",
        "source_sha256": "a" * 64,
        "node_text_sha256": "b" * 64,
        "pixel_group": {},
        "source_revision": "rev",
        "fitting_uids_sha256": triplet.roster_sha256(UIDS[:1]),
        "evaluation_uids_sha256": triplet.roster_sha256(UIDS[1:]),
        "cohort_sha256": triplet.roster_sha256(UIDS),
        "members": members,
    }
    raw = json.dumps(index, sort_keys=True).encode()
    (cache / triplet.INDEX).write_bytes(raw)
    for name, value in {
        "STUDIES": len(UIDS),
        "MEMBER_BYTES": 64,
        "NPZ_BYTES": 64 * len(UIDS),
        "INDEX_SHA256": hashlib.sha256(raw).hexdigest(),
        "COHORT_SHA256": triplet.roster_sha256(UIDS),
        "FITTING_UIDS_SHA256": triplet.roster_sha256(UIDS[:1]),
        "EVALUATION_UIDS_SHA256": triplet.roster_sha256(UIDS[1:]),
        "CACHE": cache,
    }.items():
        monkeypatch.setattr(triplet, name, value)
    rows = triplet.manifest_from_index(raw)
    for name in rows:
        os.chmod(cache / name, 0o400)
    receipt_path = tmp_path / "triplet-control" / triplet.VERIFIED
    receipt_path.parent.mkdir()
    receipt = {
        "status": relay.TRIPLET_STATUS,
        "relay_version": triplet.VERSION,
        "node": "evc40",
        "cache": str(cache),
        "index_sha256": triplet.INDEX_SHA256,
        "manifest_sha256": triplet.canonical_sha256(rows),
        "members": len(rows),
        "bytes": triplet.total_bytes(rows),
        "every_original_byte_size_SHA_count_EOF_layout_verified": True,
        "GPU_used": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    link = t.python.parent / "python3"
    link.symlink_to("python")
    for name, value in {
        "TRIPLET_RECEIPT": receipt_path,
        "TRIPLET_MANIFEST_SHA256": triplet.canonical_sha256(rows),
        "TRIPLET_FILES": len(rows),
        "TRIPLET_BYTES": triplet.total_bytes(rows),
        "ROOT": t.root,
        "ORIGINAL_FUNDING_SHA256": t.original["funding_sha256"],
        "NODE_LOCAL_PREFIX": str(tmp_path) + "/",
    }.items():
        monkeypatch.setattr(relay, name, value)
    stop_original(t)
    record(t)
    path = t.dir / "record.json"
    monkeypatch.setattr(triplet.os, "uname", lambda: os.uname_result(("L", "evc40", "", "", "")))
    for key in ("SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "908454")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    src, dst = tmp_path / "src-control", tmp_path / "dst-control"
    dst.mkdir(mode=0o700)
    return SimpleNamespace(
        t=t,
        tmp=tmp_path,
        record=path,
        record_sha=relay.sha(path),
        src=src,
        dst=dst,
        away=tmp_path / "evc31",
        cached={str(cache / n) for n in rows},
        link=link,
    )


def stream(s):
    agg = relay.plan(s.record, s.record_sha, s.src)
    out = io.BytesIO()
    relay.source(out, s.record, s.record_sha, s.src / relay.PLAN, agg["plan_sha256"])
    return agg, out.getvalue()


def depart(s, keep=()):
    """Model the fresh destination: every node-local member except the relayed cache absent."""
    value = json.loads(s.record.read_bytes())
    for cls in relay.NODE_LOCAL:
        if cls in keep:
            continue
        paths = [Path(value["member_roots"][cls])]
        if cls == "public_cache":
            paths = [
                Path(m["path"])
                for m in value["manifest"]
                if m["class"] == cls and m["path"] not in s.cached
            ]
        for path in paths:
            moved = s.away / path.relative_to("/")
            moved.parent.mkdir(parents=True, exist_ok=True)
            os.rename(path, moved)


def funding(s, agg, **edit):
    value = {
        "status": relay.FUNDING_STATUS,
        "funded_by": "root",
        "issue": 80,
        "relay_version": relay.VERSION,
        "source_node": "evc31",
        "node": "evc40",
        "control": str(s.dst),
        "transition_record_sha256": s.record_sha,
        "transition_record_bytes": agg["transition_record_bytes"],
        "coordinator_sha256": agg["coordinator_sha256"],
        "transfer_list_sha256": agg["transfer_list_sha256"],
        "stream_files": agg["stream_files"],
        "stream_bytes": agg["stream_bytes"],
        "triplet_receipt_sha256": relay.sha(relay.TRIPLET_RECEIPT),
        "envelope_bytes": agg["stream_bytes"],
        "control_bytes": 2**20,
        "reserve_bytes": 20 * GIB,
        "existing_liabilities_bytes": 50 * GIB,
        "shared_future_bytes": 0,
        "deletion_credit_bytes": 0,
        "GPUs_funded": 0,
        "GPU_reserved": False,
    }
    value["required_free_bytes"] = 50 * GIB + agg["stream_bytes"] + 2**20 + 20 * GIB
    value.update(edit)
    path = s.dst / "funding.json"
    path.unlink(missing_ok=True)
    return path, write(path, value)


def target(s, data, path, digest, **kw):
    return relay.target(io.BytesIO(data), s.dst, path, digest, **kw)


def pr76_admit(s):
    ours = (s.record.parent / "transport-receipt.json").read_bytes()
    fund(s.t, transport_receipt_sha256=hashlib.sha256(ours).hexdigest())
    assert json.loads((s.record.parent / "transport-receipt.json").read_bytes()) == json.loads(ours)
    (s.record.parent / "transport-receipt.json").unlink()
    (s.record.parent / "transport-receipt.json").write_bytes(ours)
    return admit(s.t)


def private_rows(s):
    return list(json.loads((s.src / relay.PLAN).read_bytes())["rows"])


def untouched(s):
    return not (s.dst / relay.intent_name(1)).exists() and not s.t.root.exists()


def test_pins_match_pr76_and_the_actual_triplet_relay():
    assert relay.ROOT == tr.ROOT and relay.SCHEMA == tr.SCHEMA
    assert relay.RECORD_STATUS == tr.RECORD_STATUS and relay.TRANSITIONS == tr.TRANSITIONS
    assert relay.NODE_LOCAL == tr.NODE_LOCAL
    assert relay.ORIGINAL_FUNDING_SHA256 == tr.ORIGINAL["funding_sha256"]
    assert (relay.SOURCE_NODE, relay.SOURCE_JOB) == (tr.ORIGINAL["node"], tr.ORIGINAL["parent_job"])
    assert (relay.SOURCE_NODE, relay.TARGET_NODE, relay.SOURCE_JOB) == (
        triplet.SOURCE_NODE,
        triplet.TARGET_NODE,
        triplet.SOURCE_JOB,
    )
    plan = json.loads((TRIPLET_AUDIT / "source-plan-verified.json").read_text())
    assert plan["manifest_sha256"] == relay.TRIPLET_MANIFEST_SHA256
    assert (plan["relayed_files"], plan["relayed_bytes"]) == (3404, 27668082285)
    assert (relay.TRIPLET_FILES, relay.TRIPLET_BYTES) == (3404, 27668082285)
    assert relay.TRIPLET_RECEIPT == triplet.CONTROL / triplet.VERIFIED
    assert relay.TRIPLET_STATUS in Path(triplet.__file__).read_text()
    assert relay.CONTROL != triplet.CONTROL and relay.RESERVE_BYTES == 20 * GIB


def test_walk_matches_pr76_semantics(tmp_path):
    (tmp_path / "d/e").mkdir(parents=True)
    (tmp_path / "d/e/f").write_text("x")
    (tmp_path / "d/l").symlink_to("e")
    (tmp_path / "d/m").symlink_to("e/f")
    assert relay.walk(tmp_path / "d") == tr._walk(tmp_path / "d")
    with pytest.raises(ValueError):
        relay.walk(tmp_path / "d/l")


def test_relay_reuses_triplet_cache_and_satisfies_pr76_admission(s):
    inodes = {n: os.lstat(n).st_ino for n in s.cached}
    agg, data = stream(s)
    names = private_rows(s)
    assert names[:2] == [str(s.record), str(s.t.coordinator)]
    assert not set(names) & s.cached and agg["already_relayed_cache_files"] == len(s.cached)
    assert len(data) == HEADER + agg["stream_bytes"] + len(relay.END)
    assert data.startswith(relay.MAGIC) and data.endswith(relay.END)
    assert all(npz(uid) not in data for uid in UIDS)
    depart(s)
    path, digest = funding(s, agg)
    result = target(s, data, path, digest)
    assert result["new_files"] == agg["stream_files"] and result["existing_exact_files"] == 0
    assert result["links_created"] == 1 and s.link.is_symlink()
    assert result["already_relayed_cache_files"] == len(s.cached)
    assert {n: os.lstat(n).st_ino for n in s.cached} == inodes
    value = json.loads(s.record.read_bytes())
    assert result["transport_members"] == value["transport_members"]
    assert result["transport_bytes"] == value["transport_bytes"]
    text = json.dumps(agg) + json.dumps(result)
    assert str(s.tmp) not in text and "npz" not in text
    assert pr76_admit(s)["members_verified"] == value["members"]
    assert (s.away / s.t.root.relative_to("/")).is_dir()  # originals retained at source


def test_existing_exact_destination_members_are_proven_never_overwritten(s):
    agg, data = stream(s)
    depart(s, keep=("runtime",))
    before = os.lstat(s.t.python)
    path, digest = funding(s, agg)
    result = target(s, data, path, digest)
    assert result["existing_exact_files"] == 1 and result["links_created"] == 0
    after = os.lstat(s.t.python)
    assert (after.st_ino, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    pr76_admit(s)


@pytest.mark.parametrize("change", ["differ", "extra", "link"])
def test_differing_or_extra_destination_member_refused_before_any_write(s, change):
    agg, data = stream(s)
    depart(s, keep=("runtime",))
    if change == "differ":
        s.t.python.write_text("#!other\n")
    elif change == "extra":
        (s.t.python.parent / "extra").write_text("x")
    else:
        s.link.unlink()
        s.link.symlink_to("/usr/bin/python3")
    path, digest = funding(s, agg)
    with pytest.raises(ValueError):
        target(s, data, path, digest)
    assert untouched(s)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d[:5] + b"X" + d[6:],  # magic
        lambda d: d[: len(relay.MAGIC)] + bytes(32) + d[len(relay.MAGIC) + 32 :],  # digest
        lambda d: d[: HEADER - 16] + (1).to_bytes(8, "big") + d[HEADER - 8 :],  # count
        lambda d: d[: HEADER - 8] + (1).to_bytes(8, "big") + d[HEADER:],  # byte cap
        lambda d: d[: HEADER + 3],  # truncated record
    ],
)
def test_header_or_record_mismatch_refused_before_intent(s, mutate):
    agg, data = stream(s)
    depart(s)
    path, digest = funding(s, agg)
    with pytest.raises(ValueError):
        target(s, mutate(data), path, digest)
    assert untouched(s)


@pytest.mark.parametrize("cut", ["truncated", "corrupt", "trailing"])
def test_interrupted_or_corrupt_stream_is_uncertain_and_never_promoted(s, cut):
    agg, data = stream(s)
    last = private_rows(s)[-1]
    depart(s)
    path, digest = funding(s, agg)
    end = len(data) - len(relay.END)
    bad = {
        "truncated": data[: end - 1],
        "corrupt": data[: end - 1] + bytes([data[end - 1] ^ 1]) + data[end:],
        "trailing": data + b"\0",
    }[cut]
    with pytest.raises(ValueError):
        target(s, bad, path, digest)
    assert (s.dst / relay.intent_name(1)).exists() and not (s.dst / relay.VERIFIED).exists()
    assert not (s.record.parent / "transport-receipt.json").exists()
    assert Path(last).exists() == (cut == "trailing")
    with pytest.raises(ValueError, match="no automatic retry"):
        target(s, data, path, digest)
    with pytest.raises(ValueError, match="prefix admission"):
        target(s, data, path, digest, attempt=2)
    with pytest.raises(Exception):
        pr76_admit(s)


def prefix(s, fsha, rec, **edit):
    value = {
        "status": relay.PREFIX_STATUS,
        "funded_by": "root",
        "issue": 80,
        "relay_version": relay.VERSION,
        "interrupted_attempt": 1,
        "reconcile_sha256": rec["reconcile_sha256"],
        "transition_record_sha256": s.record_sha,
        "funding_sha256": fsha,
        "automatic_retry": False,
    }
    path = s.dst / relay.prefix_name(1)
    path.unlink(missing_ok=True)
    return path, write(path, dict(value, **edit))


def test_reconcile_is_read_only_and_root_prefix_admits_one_bounded_attempt(s):
    agg, data = stream(s)
    depart(s)
    path, fsha = funding(s, agg)
    with pytest.raises(ValueError):
        relay.reconcile(s.dst, path, fsha, 1)  # nothing interrupted yet
    with pytest.raises(ValueError):
        target(s, data[: len(data) - len(relay.END) - 1], path, fsha)
    promoted = {n: os.lstat(n).st_ino for n in private_rows(s)[:-1]}
    rec = relay.reconcile(s.dst, path, fsha, 1)
    assert rec["read_only"] and rec["record_present"]
    assert rec["verified_files"] == len(promoted) and rec["absent_files"] == 1
    assert rec["unpromoted_staging_parts"] == 1 and rec["differing_files"] == 0
    assert str(s.tmp) not in json.dumps(rec)
    with pytest.raises(FileExistsError):
        relay.reconcile(s.dst, path, fsha, 1)
    stale, stale_sha = prefix(s, fsha, rec, reconcile_sha256="0" * 64)
    with pytest.raises(ValueError):
        target(s, data, path, fsha, attempt=2, prefix=stale, prefix_sha=stale_sha)
    retry, retry_sha = prefix(s, fsha, rec, automatic_retry=True)
    with pytest.raises(ValueError):
        target(s, data, path, fsha, attempt=2, prefix=retry, prefix_sha=retry_sha)
    admitted, admitted_sha = prefix(s, fsha, rec)
    result = target(s, data, path, fsha, attempt=2, prefix=admitted, prefix_sha=admitted_sha)
    assert result["existing_exact_files"] == len(promoted) and result["new_files"] == 1
    assert {n: os.lstat(n).st_ino for n in promoted} == promoted
    pr76_admit(s)
    with pytest.raises(ValueError, match="forbids repetition"):
        target(s, data, path, fsha, attempt=3)


@pytest.mark.parametrize(
    "edit",
    [
        {"reserve_bytes": 20 * GIB - 1},
        {"required_free_bytes": 1},
        {"shared_future_bytes": 1},
        {"deletion_credit_bytes": 1},
        {"GPUs_funded": 1},
        {"GPU_reserved": True},
        {"funded_by": "devin"},
        {"issue": 78},
        {"node": "evc31"},
        {"source_node": "evc40"},
        {"existing_liabilities_bytes": 0},
        {"envelope_bytes": 1},
        {"control_bytes": 2 * GIB},
        {"triplet_receipt_sha256": "0" * 64},
        {"transfer_list_sha256": "0" * 64},
        {"transition_record_sha256": "0" * 64},
        {"coordinator_sha256": "0" * 64},
        {"stream_bytes": 1},
        {"stream_files": 3},
        {"extra": 1},
    ],
)
def test_funding_must_be_explicit_root_node_local_and_gpu_free(s, edit):
    agg, data = stream(s)
    depart(s)
    if "stream_bytes" in edit:
        edit["envelope_bytes"] = 1
    path, digest = funding(s, agg, **edit)
    with pytest.raises(ValueError):
        target(s, data, path, digest)
    assert untouched(s)
    with pytest.raises(ValueError):
        target(s, data, path, "f" * 64)


def test_insufficient_or_breached_reserve_refused(s, monkeypatch):
    agg, data = stream(s)
    depart(s)
    path, digest = funding(s, agg)
    low = SimpleNamespace(free=agg["stream_bytes"], total=0, used=0)
    monkeypatch.setattr(relay.shutil, "disk_usage", lambda _: low)
    with pytest.raises(ValueError, match="reserve"):
        target(s, data, path, digest)
    assert untouched(s)
    calls = []

    def shrinking(_):
        calls.append(1)
        return SimpleNamespace(free=10**15 if len(calls) < 4 else 1, total=0, used=0)

    monkeypatch.setattr(relay.shutil, "disk_usage", shrinking)
    with pytest.raises(ValueError, match="breached"):
        target(s, data, path, digest)
    assert (s.dst / relay.intent_name(1)).exists() and not (s.dst / relay.VERIFIED).exists()


@pytest.mark.parametrize("change", ["mode", "hardlink", "receipt", "linked", "missing"])
def test_relayed_cache_must_be_unchanged_since_its_immutable_receipt(s, change):
    agg, data = stream(s)
    depart(s)
    path, digest = funding(s, agg)
    member = Path(sorted(s.cached)[-1])
    if change == "mode":
        os.chmod(member, 0o600)
    elif change == "hardlink":
        os.link(member, s.tmp / "second-name")
    elif change == "receipt":
        os.chmod(relay.TRIPLET_RECEIPT, 0o600)
        relay.TRIPLET_RECEIPT.write_text("{}")
    elif change == "linked":
        os.rename(member, s.tmp / "elsewhere.npz")
        member.symlink_to(s.tmp / "elsewhere.npz")
    else:
        member.unlink()
    with pytest.raises(ValueError):
        target(s, data, path, digest)
    assert untouched(s)


def test_linked_destination_ancestor_refused(s):
    agg, data = stream(s)
    depart(s)
    elsewhere = s.tmp / "elsewhere"
    elsewhere.mkdir()
    s.t.root.symlink_to(elsewhere)
    path, digest = funding(s, agg)
    with pytest.raises(ValueError):
        target(s, data, path, digest)
    assert not (s.dst / relay.intent_name(1)).exists() and not any(elsewhere.iterdir())


def test_destination_transition_already_continued_is_refused(s):
    agg, data = stream(s)
    depart(s)
    later = s.record.parent / "replacement-funding.json"
    later.parent.mkdir(parents=True)
    later.write_text("{}")
    path, digest = funding(s, agg)
    with pytest.raises(ValueError, match="repetition"):
        target(s, data, path, digest)


@pytest.mark.parametrize("change", ["bytes", "extra", "link", "control", "coordinator"])
def test_source_changed_after_plan_or_record_is_refused(s, change):
    agg = relay.plan(s.record, s.record_sha, s.src)
    if change == "bytes":
        os.chmod(s.t.python, 0o700)
        s.t.python.write_text("#!runtime\n")  # same bytes, new signature
    elif change == "extra":
        (s.t.python.parent / "extra").write_text("x")
    elif change == "link":
        s.link.unlink()
        s.link.symlink_to("python2")
    elif change == "control":
        (s.record.parent / "replacement-funding.json").write_text("{}")
    else:
        s.t.coordinator.write_text("# changed\n")
    with pytest.raises(ValueError):
        relay.source(io.BytesIO(), s.record, s.record_sha, s.src / relay.PLAN, agg["plan_sha256"])


def forged(s, edit):
    value = json.loads(s.record.read_bytes())
    edit(value)
    value["manifest_sha256"] = triplet.canonical_sha256(value["manifest"])
    payload = json.dumps(value).encode()
    return payload, hashlib.sha256(payload).hexdigest()


def first_local(value, cls="runtime"):
    return next(m for m in value["manifest"] if m["class"] == cls and m["kind"] == "file")


@pytest.mark.parametrize(
    "edit",
    [
        lambda v: v["original"].update(parent_job="908454"),
        lambda v: v["original"].update(node="evc40"),
        lambda v: v["original"].update(funding_sha256="0" * 64),
        lambda v: v.update(status="original_exit0_complete"),
        lambda v: v.update(original_state_deleted=True),
        lambda v: v.update(transport_bytes=v["transport_bytes"] + 1),
        lambda v: v.update(transport_classes=["original_root"]),
        lambda v: v["member_roots"].update(runtime="/shared/runtime"),
        lambda v: first_local(v).update(path=first_local(v)["path"] + "/../../x"),
        lambda v: first_local(v).update(path="/etc/passwd"),
        lambda v: first_local(v).update(kind="fifo"),
        lambda v: first_local(v, "original_root").update(
            path=str(relay.ROOT / relay.TRANSITIONS / "001/x")
        ),
    ],
)
def test_record_must_be_the_exact_actual_exit75_transition(s, edit):
    payload, digest = forged(s, edit)
    with pytest.raises(ValueError):
        relay.validate_record(payload, digest)
    with pytest.raises(ValueError):
        relay.validate_record(s.record.read_bytes(), "0" * 64)


def test_absent_record_is_never_planned(s):
    s.record.unlink()
    with pytest.raises(FileNotFoundError):
        relay.plan(s.record, s.record_sha, s.src)
    assert not s.src.exists()


@pytest.mark.parametrize(
    "key, value",
    [("SLURM_STEP_GPUS", "0"), ("CUDA_VISIBLE_DEVICES", "0"), ("SLURM_CPUS_PER_TASK", "2")],
)
def test_target_endpoint_is_one_cpu_zero_gpu(s, monkeypatch, key, value):
    agg, data = stream(s)
    depart(s)
    path, digest = funding(s, agg)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="zero-GPU"):
        target(s, data, path, digest)
    assert untouched(s)


def test_pipeline_is_one_cpu_zero_gpu_overlapping_the_original_job():
    command = relay.pipeline(Path("/r"), "a" * 64, "b" * 64, "c" * 64)
    source, target_ = command.removeprefix("set -o pipefail; ").split(" | ")
    for side in (source, target_):
        assert "--cpus-per-task=1" in side and "--gres=none" in side and "--ntasks=1" in side
        assert "gpu" not in side.replace("--gres=none", "").lower()
    assert "--jobid=899766 --overlap" in source and " source " in source
    assert "--nodelist=evc40" in target_ and "--jobid" not in target_
    assert "--prefix" not in target_
    second = relay.pipeline(
        Path("/r"), "a" * 64, "b" * 64, "c" * 64, attempt=2, prefix_sha="d" * 64
    )
    assert "--attempt 2" in second and relay.prefix_name(1) in second
