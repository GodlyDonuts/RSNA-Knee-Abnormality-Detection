"""Actual frozen derivation and five-generation byte/provenance reuse."""

import copy
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


generation = load("http499_generation_test", HERE / "generation.py")
sys.modules["generation"] = generation
owner = load("http499_owner_test", HERE / "owner.py")
owner.generation = generation
prior = load("http499_inventory_fixture", ROOT / "tests/test_predictive_cache_capacity_recovery.py")


@pytest.fixture
def bundle(monkeypatch):
    actual = owner.git_file
    monkeypatch.setattr(owner, "git_file", lambda pin, name:
                        (ROOT / name).read_text() if pin == "f" * 40 else actual(pin, name))
    decision = (HERE / "decision.accepted.json").read_bytes()
    return owner.freeze("f" * 40, decision_body=decision,
                        decision_sha256=owner.sha(decision),
                        failure_body=(HERE / "actual-terminal.json").read_bytes())


def suffix(bundle):
    result = SimpleNamespace()
    exec(bundle["payloads"]["raw"]["suffix_reconcile"], result.__dict__)
    return result


def test_exact_freeze_preserves_all_numerical_modules_and_correction_envelope(bundle):
    old = owner.previous_bundle()
    for role in ("raw", "context"):
        current, before = bundle["payloads"][role], old["payloads"][role]
        for name, text in before["files"].items():
            if name != owner.HTTP_MODULE:
                assert current["files"][name] == text
        for key in before:
            if key.startswith("source_correction") or key.startswith("source_overlay"):
                assert current[key] == before[key]
        assert current["scratch_cap_bytes"] == 6 * 2**30
        assert current["funding"] == before["funding"]
        assert bundle["programs"][role].count('**suffix.http499_metadata(v)') == 2
    assert bundle["payloads"]["raw"]["previous_source_http_sha256"] == old["payloads"]["raw"]["source_http_sha256"]
    for role, source in bundle["programs"].items():
        compile(source, role, "exec")
    assert bundle["programs"]["collect"] == old["programs"]["collect"]
    for role in ("raw", "context"):
        admission = bundle["programs"]["admit"]
        for generation_name in ("cache", "alias-recovery", "capacity-recovery", "http-recovery"):
            assert f"full-{role}-{generation_name}-20261020" in admission
        assert f'Path(chain["{role}"]["root"])' in admission
    assert '"original_jobs": suffix.HTTP499_PREDECESSORS' in bundle["programs"]["admit"]


def test_all_older_decision_contracts_keep_original_globals_with_new_payload(bundle):
    helper = suffix(bundle)
    value = bundle["payloads"]["raw"]
    helper.http499_contract(value)
    helper.correction_contract(value)
    # The archived guard must see the original HTTP helper through a private
    # payload view, not mutate the newly admitted helper or older decisions.
    captured = []
    helper.predecessor_guard.__globals__["_data_guard"] = lambda v, *args: captured.append(v) or {}
    with pytest.raises(FileNotFoundError):
        helper.predecessor_guard(value, prior.recon,
            lambda argv, **kw: "981958|FAILED|1:0\n981960|FAILED|1:0\n" if argv[0] == "sacct" else "", {})
    assert captured[0]["source_http_sha256"] == helper.PREVIOUS_HTTP_SHA
    assert value["source_http_sha256"] != helper.PREVIOUS_HTTP_SHA
    assert helper.PREDECESSORS == {"raw": "980960", "context": "980969"}
    assert helper.HTTP499_PREDECESSORS == owner.ORIGINALS


def empty_root(path):
    path.mkdir(mode=0o700)
    for part in ("cache", "proofs", "scratch"):
        (path / part).mkdir(mode=0o700)
    return path


def test_five_roots_reuse_baseline_prefix_and_sealed_corrected_case_without_writes(tmp_path, bundle):
    helper = suffix(bundle)
    original, records, initial, _ = prior.fixture.case.__wrapped__(tmp_path)
    extra = [empty_root(tmp_path / f"old-{i}") for i in range(3)]
    data = empty_root(tmp_path / "data-generation")
    effective = copy.deepcopy(records)
    effective[2]["source_files"][0]["bytes"] += 1
    effective[2]["source_files"][0]["sha256"] = "a" * 64
    archive = data / "cache/case-0002.private.npz"
    archive.write_bytes(b"original-corrected-cache")
    row = prior.complete(data, effective[2], archive, 2)
    correction = {"before_sha256": "b" * 64, "after_sha256": "a" * 64}
    row["source_correction"] = correction
    sp = data / "proofs/case-0002.source-proof.private.json.gz"
    proof = json.loads(gzip.decompress(sp.read_bytes()))
    proof["source_correction"] = correction
    compressed = gzip.compress(json.dumps(proof).encode())
    sp.write_bytes(compressed)
    row["source_proof_sha256"] = owner.sha(compressed)
    vp = data / "proofs/case-0002.verified.private.json"
    vp.write_text(json.dumps(row))
    helper.raw_inventory.__globals__.update(HTTP499_DATA_ROOT=data, _effective_records=effective,
                                            _source_corrections={2: correction})
    before = {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    result = helper.raw_inventory(prior.recon, [original, *extra, data], records, initial,
                                  maximum_buffers=6)
    assert result["complete"][2] == row and result["pending"] == []
    assert result["all_buffers"]  # The original defective buffer is still retained/funded.
    assert before == {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    row.pop("source_correction")
    vp.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="correction provenance"):
        helper.raw_inventory(prior.recon, [original, *extra, data], records, initial)


def test_corrected_raw_and_context_rows_must_be_sealed_not_reapplied(bundle):
    helper = suffix(bundle)
    changes = {1740: {"exact": "one"}, 1850: {"exact": "two"}}
    raw = {"complete": {i: {"source_correction": c} for i, c in changes.items()}, "orphan_caches": {}}
    ctx = {"complete": {i: {"provenance": {"source_correction": c}} for i, c in changes.items()},
           "orphan_caches": {}}
    helper.require_unsealed(raw, changes)
    helper.require_corrected_contexts(ctx, changes)
    del raw["complete"][1740]
    with pytest.raises(ValueError, match="sealed"):
        helper.require_unsealed(raw, changes)
    ctx["complete"][1850]["provenance"]["source_correction"] = {"wrong": True}
    with pytest.raises(ValueError, match="sealed"):
        helper.require_corrected_contexts(ctx, changes)


def test_once_transition_requires_actual_terminal_and_never_retries():
    jobs = {j: {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}
            for j in owner.ORIGINALS.values()}
    assert owner.next_action(jobs, {}) == ("dispatch", "raw")
    with pytest.raises(AssertionError):
        owner.next_action(jobs, {"raw": {"returncode": 1, "job_id": "123"}})
    jobs["981958"]["queue_empty"] = False
    assert owner.next_action(jobs, {}) == ("wait",)
