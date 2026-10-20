"""Distinct HTTP-owner integration: immutable generations and no remote actions."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "initiative/research/predictive-cache-http-recovery-20261020-v1"
OLD = ROOT / "initiative/research/predictive-cache-capacity-recovery-20261020-v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generation = load("http_generation_tests", HERE / "generation.py")
sys.modules["generation"] = generation
owner = load("http_owner_tests", HERE / "owner.py")
prior = load("http_prior_tests", ROOT / "tests/test_predictive_cache_capacity_recovery.py")


def frozen_programs():
    return {
        role: owner.git_file(owner.PREVIOUS, owner.PREVIOUS_REL + "/" + name)
        for role, name in (("raw", "program.py.txt"), ("context", "context.py.txt"),
                           ("admit", "admit.py.txt"), ("collect", "collect.py.txt"))
    }


def suffix():
    text = owner.git_file(owner.PREVIOUS, owner.PREVIOUS_REL + "/suffix.py")
    text += "\n" + (HERE / "suffix.py").read_text()
    module = SimpleNamespace()
    exec(compile(text, "http_generation_reconciliation", "exec"), module.__dict__)
    return module


def test_generated_programs_keep_decoding_and_rendering_exact_and_account_all_generations():
    old = frozen_programs()
    new = generation.programs(old)
    for role in new:
        compile(new[role], role, "exec")
    assert new["collect"] == old["collect"]
    # Actual one-case scientific extraction/transformation is byte-identical.
    a, b = 'tree = ast.parse(v["original_program"])', "start = time.monotonic()"
    assert old["raw"][old["raw"].index(a):old["raw"].index(b)] == (
        new["raw"][new["raw"].index(a):new["raw"].index(b)]
    )
    a, b = '    groups, source_proof = cache.load', '    results[i] = entry'
    assert old["context"][old["context"].index(a):old["context"].index(b)] == (
        new["context"][new["context"].index(a):new["context"].index(b)]
    )
    assert "[original, predecessor, capacity_predecessor]" in new["raw"]
    assert "[old_raw, predecessor_raw, capacity_raw]" in new["context"]
    assert "[old_context, predecessor_context, capacity_context]" in new["context"]
    assert "capacity_predecessor / \"scratch\"" in new["raw"]
    assert "control_bytes(capacity_predecessor)" in new["raw"]
    assert "controls_size(capacity_context)" in new["context"]
    assert new["raw"].count("SourceHTTP(revision=revision, token=token,") == 1
    assert 'print(json.dumps({"source_HTTP": event}), flush=True)' in new["raw"]
    assert "http.open_response(row, offset)" in new["raw"]
    assert "requests.get(" not in new["raw"]
    assert 'continuation_chain["raw"]["ancestor_chain"]["raw"]["actual_terminal"]' in new["raw"]


def test_three_generations_reuse_completed_orphan_and_retained_partial_without_writes(tmp_path):
    original, records, initial, data = prior.fixture.case.__wrapped__(tmp_path)
    middle = prior.successor(tmp_path)
    latest = tmp_path / "http-failed-original"
    latest.mkdir(mode=0o700)
    for name in ("cache", "scratch", "proofs"):
        (latest / name).mkdir(mode=0o700)
    destination = latest / "scratch/fitting-study-http-original"
    data.parent.rename(destination)
    (latest / "proofs/case-0002.intent.json").write_bytes(
        (original / "proofs/case-0002.intent.json").read_bytes()
    )
    before = {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    helper = suffix()
    inv = helper.raw_inventory(prior.recon, [original, middle, latest], records, initial,
                               maximum_buffers=6)
    assert inv["pending"] == [2]
    assert inv["buffers"][2] == destination / "data"
    assert inv["buffer_owners"][str(destination / "data")] == latest
    assert inv["orphan_caches"][2] == original / "cache/case-0002.private.npz"
    assert before == {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    expected = prior.complete(latest, records[2], inv["orphan_caches"][2], 2)
    inv = helper.raw_inventory(prior.recon, [original, middle, latest], records, initial,
                               maximum_buffers=6)
    assert inv["complete"][2] == expected and inv["pending"] == []


def decision():
    value = {
        "status": "accepted_original_v4_HTTP_failure_reconciled",
        "original_jobs": owner.ORIGINALS,
        "verified_failure_site": "HTTP_response_status_before_body",
        "all_originals_preserved": True,
        "source_http_sha256": owner.sha((ROOT / owner.HTTP_MODULE).read_bytes()),
        "funding": owner.previous().FUNDING,
        "lane_count": 4, "maximum_HTTP_requests": 8, "scratch_cap_bytes": 6 * 2**30,
    }
    return json.dumps(value).encode()


def test_freeze_reuses_actual_science_and_pins_new_transport_without_replacing_originals(monkeypatch):
    real = owner.git_file
    monkeypatch.setattr(owner, "git_file", lambda pin, path:
                        (ROOT / path).read_text() if pin == "f" * 40 else real(pin, path))
    capacity = (ROOT / "initiative/research/parallel-serving-account-20261020-v1"
                "/capacity-recovery-decision.accepted.json").read_bytes()
    body = decision()
    bundle = owner.freeze("f" * 40, *prior.receipt_arguments(), capacity_body=capacity,
                          capacity_sha256=owner.sha(capacity), http_body=body,
                          http_sha256=owner.sha(body))
    raw, context = bundle["payloads"]["raw"], bundle["payloads"]["context"]
    assert raw["source_git"] == context["source_git"] == "f" * 40
    assert raw["original_program"] == prior.owner.git_file(
        prior.owner.base_module().RAW_SOURCE,
        "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt",
    )
    assert owner.HTTP_MODULE in raw["files"] and owner.HTTP_MODULE not in context["files"]
    predecessor = owner.previous().freeze(
        owner.PREVIOUS, *prior.receipt_arguments(), capacity_body=capacity,
        capacity_sha256=owner.sha(capacity),
    )
    # The historic producer snapshots differ in inactive training code. Each
    # retains its own exact original numerical source, not a synthesized mix.
    for role in ("raw", "context"):
        for name, source in predecessor["payloads"][role]["files"].items():
            assert bundle["payloads"][role]["files"][name] == source
    assert raw["context_module_sha256"] == context["module_sha256"]
    assert raw["source_http_sha256"] == owner.sha(raw["files"][owner.HTTP_MODULE].encode())
    assert raw["mixed_prefix_admission"] is None
    compile(raw["suffix_reconcile"], "complete_reconciler", "exec")


def test_exact_v4_failure_and_ancestry_required_before_recovery(tmp_path, monkeypatch):
    helper = suffix()
    ancestors = {"raw": {"job": "979664"}, "context": {"job": "979665"}}
    # The original capacity guard's own tests cover its real historical intent checks.
    helper.predecessor_guard.__globals__["_capacity_guard"] = lambda *args: ancestors
    roots, hashes = {}, {}
    for role, job in owner.ORIGINALS.items():
        root = tmp_path / role
        root.mkdir(mode=0o700)
        body = json.dumps({"CPU_job": job, "source_git": owner.PREVIOUS,
                           "continuation_chain": ancestors}).encode()
        (root / "intent.json").write_bytes(body)
        roots[role], hashes[role] = str(root), owner.sha(body)
    scope = helper.predecessor_guard.__globals__
    scope.update(HTTP_ROOTS=roots, INTENTS=hashes)
    body = decision()
    value = {"http_recovery_decision_body": body.decode(),
             "http_recovery_decision_sha256": owner.sha(body),
             "source_http_sha256": json.loads(body)["source_http_sha256"]}
    def output(argv, **kwargs):
        return "980743|FAILED|1:0\n980744|FAILED|1:0\n" if argv[0] == "sacct" else ""
    actual = helper.predecessor_guard(value, prior.recon, output, {})
    assert actual["raw"]["ancestor_chain"] == ancestors
    assert actual["raw"]["actual_terminal"]["job_id"] == "980743"
    def active(argv, **kwargs):
        return output(argv, **kwargs) if argv[0] == "sacct" else "980743|RUNNING\n"
    with pytest.raises(ValueError):
        helper.predecessor_guard(value, prior.recon, active, {})
    (Path(roots["raw"]) / "intent.json").write_text("{}")
    with pytest.raises(ValueError):
        helper.predecessor_guard(value, prior.recon, output, {})


def test_owner_transition_and_no_repeat_after_new_failure():
    jobs = {j: {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}
            for j in owner.ORIGINALS.values()}
    assert owner.next_action(jobs, {}) == ("dispatch", "raw")
    jobs["123"] = {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}
    with pytest.raises(AssertionError):
        owner.next_action(jobs, {"raw": {"job_id": "123", "returncode": 1}})


def test_actual_wrapped_payload_uses_new_owner_root_before_ready_signal():
    base = owner.previous().base_module()
    base.REMOTE_ROOTS = generation.ROOTS
    text = base.wrapped("pass\n", "raw")
    assert generation.ROOTS["raw"] in text
    assert generation.PREVIOUS_ROOTS["raw"] not in text
    compile(text, "new_owner_wrapped_start", "exec")


def test_finite_owner_uses_new_journal_and_exact_roots_once(tmp_path, monkeypatch):
    capacity = owner.previous()
    base = capacity.base_module()
    base.ROOT = tmp_path
    (tmp_path / "artifacts").mkdir()
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setattr(owner, "CONTROL", control)
    monkeypatch.setattr(capacity, "base_module", lambda: base)
    monkeypatch.setattr(owner, "previous", lambda: capacity)
    bundle = {"commit": "f" * 40, "sha256": {}, "transport": object(),
              "payloads": {"raw": {}, "context": {}}}
    monkeypatch.setattr(owner, "freeze", lambda *a, **kw: bundle)
    calls, phases = [], {}

    def observe(_socket, ids, _transport, quota=False):
        jobs = {j: {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}
                for j in (*owner.ORIGINALS.values(), "979573", "979604")}
        for phase in phases.values():
            jobs[phase["job_id"]] = {"state": "COMPLETED", "exit_code": "0:0",
                                    "queue_empty": True}
        return jobs, "/home/sa305415 100 40000000 0 - 100 6000 0"

    def launch(name, role, payload, _bundle, _journal, _socket, _quota):
        assert name not in phases
        calls.append((name, role, payload.copy()))
        result = {"job_id": str(990000 + len(calls)), "returncode": 0,
                  "ready": True, "accepted": role == "collect"}
        phases[name] = result
        return result

    monkeypatch.setattr(base, "observe", observe)
    monkeypatch.setattr(base, "launch", launch)
    monkeypatch.setattr(base, "refresh", lambda *args: None)
    monkeypatch.setattr(base, "credential", lambda: "dummy-stdin-credential")
    monkeypatch.setattr(base.time, "sleep", lambda _seconds: None)
    p = tmp_path / "receipt.json"
    p.write_text("{}")
    argv = ["owner.py", "--source-git", "f" * 40]
    for receipt in ("regression", "prior-attempt", "prefix-admission"):
        argv += ["--" + receipt + "-receipt", str(p), "--" + receipt + "-sha256", "0" * 64]
    for receipt in ("capacity-decision", "http-decision", "mixed-prefix-admission"):
        argv += ["--" + receipt, str(p), "--" + receipt + "-sha256", "0" * 64]
    monkeypatch.setattr(sys, "argv", argv)
    owner.main()
    assert [c[0] for c in calls] == ["raw", "raw_collect", "context", "context_collect"]
    assert calls[1][2]["root"] == generation.ROOTS["raw"]
    assert calls[3][2]["root"] == generation.ROOTS["context"]
    journal = tmp_path / "artifacts/control/owner-result.json"
    assert json.loads(journal.read_text())["new_GPU_models_or_competition_entries"] == 0
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        owner.main()
    assert len(calls) == 4
