import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from scripts.newton import triplet_cache_relay as relay

AUDIT = Path("initiative/audits/dense-suffix-admission-root-review-20261014-v1")
UIDS = ["1.2.826.0.1.1", "1.2.826.0.1.2", "1.2.826.0.1.3"]
SIZE = 64
GIB = 2**30


def payload(uid):
    return hashlib.sha256(uid.encode()).digest() * 2


def write_index(cache, members, **extra):
    index = {
        "schema_version": 1,
        "kind": "k",
        "renderer": "r",
        "source_sha256": "a" * 64,
        "node_text_sha256": "b" * 64,
        "pixel_group": {},
        "source_revision": "rev",
        "fitting_uids_sha256": relay.roster_sha256(UIDS[:2]),
        "evaluation_uids_sha256": relay.roster_sha256(UIDS[2:]),
        "cohort_sha256": relay.roster_sha256(members),
        "members": members,
        **extra,
    }
    raw = json.dumps(index, sort_keys=True, indent=1).encode()
    (cache / relay.INDEX).write_bytes(raw)
    return raw


@pytest.fixture
def env(tmp_path, monkeypatch):
    cache = tmp_path / "src" / "merged-cache"
    (cache / relay.MEMBERS_DIR).mkdir(parents=True)
    members = {}
    for uid in UIDS:
        data = payload(uid)
        (cache / relay.MEMBERS_DIR / f"{uid}.npz").write_bytes(data)
        members[uid] = {
            "path": f"members/{uid}.npz",
            "bytes": SIZE,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    for name in ("intent.json", "finalized.json", "receipts/x.json"):
        path = cache / relay.DIRECTORY / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
    raw = write_index(cache, members)
    merged = {
        "index_sha256": hashlib.sha256(raw).hexdigest(),
        "studies": len(UIDS),
        "complete_shards": relay.SHARDS,
        "labels_in_cache": False,
        "exact_member_bytes": SIZE * len(UIDS),
    }
    receipt = cache.parent / "merged-cache-verified.json"
    receipt.write_text(json.dumps(merged))
    for name, value in {
        "STUDIES": len(UIDS),
        "MEMBER_BYTES": SIZE,
        "NPZ_BYTES": SIZE * len(UIDS),
        "INDEX_SHA256": hashlib.sha256(raw).hexdigest(),
        "COHORT_SHA256": relay.roster_sha256(UIDS),
        "FITTING_UIDS_SHA256": relay.roster_sha256(UIDS[:2]),
        "EVALUATION_UIDS_SHA256": relay.roster_sha256(UIDS[2:]),
        "MERGED_RECEIPT_SHA256": relay.sha(receipt),
    }.items():
        monkeypatch.setattr(relay, name, value)
    monkeypatch.setattr(relay, "_free", lambda path: 1000 * GIB)
    monkeypatch.setattr(relay.os, "uname", lambda: os.uname_result(("L", "evc40", "", "", "")))
    for key in ("SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "910001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    target = tmp_path / "dst" / "shards" / "merged-cache"
    src_control, dst_control = tmp_path / "src-control", tmp_path / "dst-control"
    dst_control.mkdir()
    total = len(raw) + SIZE * len(UIDS)
    funding = {
        "status": relay.FUNDING_STATUS,
        "funded_by": "root",
        "issue": 78,
        "relay_version": relay.VERSION,
        "node": "evc40",
        "cache": str(target),
        "control": str(dst_control),
        "index_sha256": relay.INDEX_SHA256,
        "cache_bytes": total,
        "control_bytes": 2**20,
        "reserve_bytes": 20 * GIB,
        "existing_liabilities_bytes": 300 * GIB,
        "required_free_bytes": 300 * GIB + total + 2**20 + 20 * GIB,
        "shared_future_bytes": 0,
        "deletion_credit_bytes": 0,
        "GPUs_funded": 0,
        "GPU_reserved": False,
    }
    (target.parent.parent).mkdir(parents=True)
    return dict(
        cache=cache,
        receipt=receipt,
        target=target,
        src_control=src_control,
        control=dst_control,
        funding=funding,
        raw=raw,
        members=members,
    )


def write_funding(e, **changes):
    path = e["control"] / "funding.json"
    path.write_text(json.dumps(e["funding"] | changes))
    return path, relay.sha(path)


def planned(e):
    aggregate = relay.plan(e["cache"], e["receipt"], e["src_control"])
    return aggregate, e["src_control"] / relay.PLAN


def stream(e):
    aggregate, plan = planned(e)
    buffer = io.BytesIO()
    relay.source(buffer, e["cache"], plan, aggregate["plan_sha256"])
    return buffer.getvalue()


def run_target(e, raw, **changes):
    path, digest = write_funding(e, **changes)
    return relay.target(io.BytesIO(raw), e["target"], e["control"], path, digest, "evc40")


def test_pinned_constants_match_actual_CPU_receipt():
    record = json.loads((AUDIT / "actual-CPU-accepted.json").read_text())
    cohort = record["receipt"]["public_cohort"]
    assert record["receipt_sha256"] == relay.CPU_RECEIPT_SHA256
    assert cohort["index_sha256"] == relay.INDEX_SHA256
    assert cohort["cohort_sha256"] == relay.COHORT_SHA256
    assert cohort["fitting_uids_sha256"] == relay.FITTING_UIDS_SHA256
    assert cohort["evaluation_uids_sha256"] == relay.EVALUATION_UIDS_SHA256
    assert cohort["members"] == relay.STUDIES == 3403
    assert record["receipt"]["merged_cache_receipt_sha256"] == relay.MERGED_RECEIPT_SHA256
    assert relay.NPZ_BYTES == 27667165884
    assert relay.CACHE == Path(record["receipt"]["GPU_wrapper_arguments"]["cache"])


def test_full_relay_exact_bytes_layout_and_aggregate_only_outputs(env):
    aggregate, _ = planned(env)
    assert (
        aggregate["relayed_files"] == 4 and aggregate["not_relayed_producer_bookkeeping_files"] == 3
    )
    (env["src_control"] / relay.PLAN).unlink()
    receipt = run_target(env, stream(env))
    names = sorted(
        str(p.relative_to(env["target"])) for p in env["target"].rglob("*") if p.is_file()
    )
    assert names == sorted([relay.INDEX] + [f"{relay.MEMBERS_DIR}/{u}.npz" for u in UIDS])
    assert all((env["cache"] / n).read_bytes() == (env["target"] / n).read_bytes() for n in names)
    assert receipt["files"] == 4 and receipt["studies"] == 3 and receipt["GPU_used"] is False
    published = json.dumps(
        [aggregate, receipt, json.loads((env["control"] / relay.VERIFIED).read_text())]
    )
    assert not any(uid in published for uid in UIDS)
    assert json.loads((env["control"] / relay.INTENT).read_text())["automatic_retry"] is False


def test_repeat_refused_by_existing_intent_and_target(env):
    raw = stream(env)
    run_target(env, raw)
    with pytest.raises(ValueError, match="Existing intent"):
        run_target(env, raw)
    (env["control"] / relay.INTENT).unlink()
    (env["control"] / relay.VERIFIED).unlink()
    with pytest.raises(ValueError, match="Existing target"):
        run_target(env, raw)


@pytest.mark.parametrize(
    "edit",
    [
        lambda b: b[:-1],
        lambda b: b + b"x",
        lambda b: (
            b[: -len(relay.END) - 1] + bytes([b[-len(relay.END) - 1] ^ 1]) + b[-len(relay.END) :]
        ),
        lambda b: b[:70] + bytes([b[70] ^ 1]) + b[71:],
        lambda b: b[:12] + b"\0" * 32 + b[44:],
        lambda b: b[:44] + (5).to_bytes(8, "big") + b[52:],
        lambda b: b[:52] + (10**6).to_bytes(8, "big") + b[60:],
    ],
    ids=["truncated", "extra", "npz-byte", "index-byte", "header-sha", "count", "total-cap"],
)
def test_tampered_stream_never_verified(env, edit):
    with pytest.raises(ValueError):
        run_target(env, edit(stream(env)))
    assert not (env["control"] / relay.VERIFIED).exists()


def test_extra_missing_linked_or_changed_source_refused_before_any_byte(env):
    extra = env["cache"] / relay.MEMBERS_DIR / "extra.npz"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="extra"):
        planned(env)
    extra.unlink()
    member = env["cache"] / relay.MEMBERS_DIR / f"{UIDS[0]}.npz"
    os.link(member, env["cache"] / "hardlink")
    with pytest.raises(ValueError, match="Linked"):
        planned(env)
    (env["cache"] / "hardlink").unlink()
    aggregate, plan = planned(env)
    member.write_bytes(payload(UIDS[1]))
    buffer = io.BytesIO()
    with pytest.raises(ValueError, match="signature"):
        relay.source(buffer, env["cache"], plan, aggregate["plan_sha256"])
    assert buffer.getvalue() == b""
    member.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        relay.source(buffer, env["cache"], plan, aggregate["plan_sha256"])
    assert buffer.getvalue() == b""


def test_symlinked_member_and_wrong_plan_refused(env):
    member = env["cache"] / relay.MEMBERS_DIR / f"{UIDS[0]}.npz"
    data = member.read_bytes()
    member.unlink()
    (env["cache"] / "real").write_bytes(data)
    member.symlink_to(env["cache"] / "real")
    with pytest.raises(ValueError, match="Linked"):
        planned(env)


def test_traversal_or_changed_index_refused(env, monkeypatch):
    bad = dict(env["members"])
    bad["../escape"] = {"path": "members/../escape.npz", "bytes": SIZE, "sha256": "c" * 64}
    raw = write_index(env["cache"], bad)
    monkeypatch.setattr(relay, "STUDIES", 4)
    monkeypatch.setattr(relay, "COHORT_SHA256", relay.roster_sha256(bad))
    monkeypatch.setattr(relay, "INDEX_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="traversing"):
        relay.manifest_from_index(raw)
    with pytest.raises(ValueError, match="pinned index"):
        relay.manifest_from_index(raw + b" ")


def test_merge_receipt_must_bind_index(env):
    env["receipt"].write_text(json.dumps({"index_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="merge receipt"):
        planned(env)


@pytest.mark.parametrize(
    "changes",
    [
        {"shared_future_bytes": 1},
        {"deletion_credit_bytes": 1},
        {"reserve_bytes": 20 * GIB - 1},
        {"required_free_bytes": 1},
        {"GPUs_funded": 1},
        {"GPU_reserved": True},
        {"node": "evc31"},
        {"funded_by": "devin"},
        {"cache_bytes": 1},
        {"control_bytes": 2**30},
    ],
)
def test_funding_proof_refused_before_target_mutation(env, changes):
    raw = stream(env)
    with pytest.raises(ValueError, match="funding"):
        run_target(env, raw, **changes)
    assert not (env["control"] / relay.INTENT).exists() and not env["target"].exists()


def test_insufficient_reserve_and_wrong_funding_hash_refused(env, monkeypatch):
    raw = stream(env)
    path, _ = write_funding(env)
    with pytest.raises(ValueError, match="funding proof differs"):
        relay.target(io.BytesIO(raw), env["target"], env["control"], path, "0" * 64, "evc40")
    monkeypatch.setattr(relay, "_free", lambda p: env["funding"]["required_free_bytes"] - 1)
    with pytest.raises(ValueError, match="reserve"):
        run_target(env, raw)
    assert not (env["control"] / relay.INTENT).exists() and not env["target"].exists()


def test_reserve_breach_during_stream_stops(env, monkeypatch):
    raw = stream(env)
    calls = iter([1000 * GIB, 1000 * GIB, 1000 * GIB, 0])
    monkeypatch.setattr(relay, "_free", lambda p: next(calls, 0))
    with pytest.raises(ValueError, match="breached"):
        run_target(env, raw)
    assert not (env["control"] / relay.VERIFIED).exists()


@pytest.mark.parametrize(
    "setting",
    [
        ("node", "evc31"),
        ("SLURM_JOB_ID", None),
        ("SLURM_STEP_ID", None),
        ("SLURM_STEP_GPUS", "0"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("SLURM_CPUS_PER_TASK", "2"),
    ],
)
def test_unallocated_wrong_node_or_GPU_endpoint_refused(env, monkeypatch, setting):
    key, value = setting
    if key == "node":
        monkeypatch.setattr(relay.os, "uname", lambda: os.uname_result(("L", value, "", "", "")))
    elif value is None:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="endpoint"):
        run_target(env, stream(env))
    assert not (env["control"] / relay.INTENT).exists()


def test_uncertain_transfer_reconciled_read_only(env):
    raw = stream(env)
    cut = len(raw) - len(relay.END) - SIZE // 2
    with pytest.raises(ValueError):
        run_target(env, raw[:cut])
    before = {p: p.read_bytes() for p in env["target"].rglob("*") if p.is_file()}
    result = relay.reconcile(env["target"], env["control"])
    assert result["verified_prefix_files"] == 3 and result["unverified_last_file"] is True
    assert (
        result["missing_files"] == 1 and result["intent_present"] and not result["verified_present"]
    )
    assert before == {p: p.read_bytes() for p in env["target"].rglob("*") if p.is_file()}
    assert not any(uid in json.dumps(result) for uid in UIDS)
    first = env["target"] / relay.MEMBERS_DIR / f"{UIDS[0]}.npz"
    first.chmod(0o600)
    first.write_bytes(b"\0" * SIZE)
    with pytest.raises(ValueError, match="Earlier"):
        relay.reconcile(env["target"], env["control"])


def test_pipeline_is_one_CPU_zero_GPU_per_endpoint():
    command = relay.pipeline(Path("/tmp/relay"), "a" * 64, "b" * 64)
    source, target = command.removeprefix("set -o pipefail; ").split(" | ")
    for side in (source, target):
        assert "--cpus-per-task=1" in side and "--gres=none" in side and "gpu" not in side
    assert "--jobid=899766 --overlap" in source and "--nodelist=evc40" in target
    assert " source " in source and " target " in target
