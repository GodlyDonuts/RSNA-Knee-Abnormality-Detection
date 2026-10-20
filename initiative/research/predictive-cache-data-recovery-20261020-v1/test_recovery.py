"""Exact-source continuation contracts; no remote operations or MRI decoding."""
import ast
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generation = load("data_generation_test", HERE / "generation.py")
sys.modules["generation"] = generation
owner = load("data_owner_test", HERE / "owner.py")
prior = load("data_capacity_tests", ROOT / "tests/test_predictive_cache_capacity_recovery.py")
owner.generation = generation


def body(path):
    return (ROOT / path).read_bytes()


def bundle(monkeypatch):
    real = owner.git_file
    monkeypatch.setattr(owner, "git_file", lambda pin, name:
                        (ROOT / name).read_text() if pin == "f" * 40 else real(pin, name))
    capacity = body("initiative/research/parallel-serving-account-20261020-v1/capacity-recovery-decision.accepted.json")
    http = body("initiative/research/predictive-cache-http-recovery-20261020-v1/decision.accepted.json")
    decision = (HERE / "decision.accepted.json").read_bytes()
    regression = body("initiative/research/predictive-source-decode-failure-20261020-v1/actual-authoritative-overlay-reproduction.json")
    return owner.freeze("f" * 40, *prior.receipt_arguments(),
        capacity_body=capacity, capacity_sha256=owner.sha(capacity),
        http_body=http, http_sha256=owner.sha(http),
        correction_body=decision, correction_sha256=owner.sha(decision),
        plan_body=body("initiative/audits/source-pixel-representation-v1/source_correction_plan.json"),
        receipt_body=body("initiative/audits/source-pixel-representation-v1/hf-source-correction-receipt.json"),
        correction_regression_body=regression, correction_regression_sha256=owner.sha(regression),
        corrected_sources=json.loads(decision)["corrected_sources"])


def test_actual_evidence_freeze_exact_unchanged_numerical_modules(monkeypatch):
    b = bundle(monkeypatch)
    raw, ctx = b["payloads"]["raw"], b["payloads"]["context"]
    for role, payload in (("raw", raw), ("context", ctx)):
        compile(b["programs"][role], role, "exec")
        for name, source in payload["files"].items():
            assert owner.sha(source.encode()) == payload["module_sha256"][name]
    assert raw["source_correction_helper_sha256"] == owner.sha((HERE / "source_correction.py").read_bytes())
    assert raw["source_correction_decision_sha256"] == owner.sha((HERE / "decision.accepted.json").read_bytes())
    assert raw["source_http_sha256"] == ctx["source_http_sha256"]
    assert raw["scratch_cap_bytes"] == 6 * 2**30
    assert raw["suffix_reconcile"].count("len(roots) <= 4") == 2
    assert raw["original_program"] == prior.owner.git_file(prior.owner.base_module().RAW_SOURCE,
        "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt")
    for role in ("raw", "context"):
        assert b["programs"][role].count('**correction_meta') == 2
    admitted = b["programs"]["admit"]
    assert 'full-raw-cache-20261020-v1' in admitted
    assert 'full-raw-alias-recovery-20261020-v3' in admitted
    assert 'full-raw-capacity-recovery-20261020-v1' in admitted
    assert 'Path(chain["raw"]["root"])' in admitted  # Original failed HTTP owner.
    assert generation.ROOTS["raw"] in admitted  # Only absence guard points at new output.


def suffix_from(b):
    module = SimpleNamespace()
    exec(b["payloads"]["raw"]["suffix_reconcile"], module.__dict__)
    return module


def test_four_original_generations_preserve_retained_buffers_and_no_writes(tmp_path, monkeypatch):
    b = bundle(monkeypatch)
    helper = suffix_from(b)
    original, records, initial, data = prior.fixture.case.__wrapped__(tmp_path)
    extra = []
    for n in range(3):
        root = tmp_path / ("generation-" + str(n)); root.mkdir(mode=0o700)
        for name in ("scratch", "cache", "proofs"):
            (root / name).mkdir(mode=0o700)
        extra.append(root)
    before = {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    inv = helper.raw_inventory(prior.recon, [original, *extra], records, initial, maximum_buffers=6)
    assert inv["pending"] == [2] and inv["buffers"][2] == data
    assert before == {str(p): (p.read_bytes(), p.stat()) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Finite distinct"):
        helper.raw_inventory(prior.recon, [original, *extra, original], records, initial)


def test_original_one_case_only_adds_exact_provenance(monkeypatch):
    b = bundle(monkeypatch); helper = suffix_from(b)
    original = b["payloads"]["raw"]["original_program"]
    tree = ast.parse(original)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "one")
    text = ast.get_source_segment(original, function)
    text = text.replace('p = D / "cache" / ("case-" + str(i).zfill(4) + ".private.npz")', 'p = output_path(i)')
    changed = helper.extend_one(text)
    parsed = ast.parse(changed).body[0]
    parsed.body = [n for n in parsed.body if not (isinstance(n, ast.If) and ast.unparse(n.test) == "i in corrections")]
    assert ast.dump(parsed, include_attributes=False) == ast.dump(ast.parse(text).body[0], include_attributes=False)


def test_original_failure_ancestry_and_live_queue_rejected(tmp_path, monkeypatch):
    b = bundle(monkeypatch); helper = suffix_from(b)
    ancestors = {"raw": {"job": "980743"}, "context": {"job": "980744"}}
    scope = helper.predecessor_guard.__globals__
    scope["_http_guard"] = lambda *a: ancestors
    roots, hashes = {}, {}
    for role, job in owner.ORIGINALS.items():
        root = tmp_path / role; root.mkdir(mode=0o700)
        data = json.dumps({"CPU_job": job, "source_git": owner.PREVIOUS, "continuation_chain": ancestors}).encode()
        (root / "intent.json").write_bytes(data)
        roots[role], hashes[role] = str(root), owner.sha(data)
    scope.update(DATA_PREDECESSOR_ROOTS=roots, INTENTS=hashes)
    def output(argv, **kwargs):
        return "980960|FAILED|1:0\n980969|FAILED|1:0\n" if argv[0] == "sacct" else ""
    chain = helper.predecessor_guard(b["payloads"]["raw"], prior.recon, output, {})
    assert chain["raw"]["ancestor_chain"] == ancestors
    def live(argv, **kwargs):
        return output(argv, **kwargs) if argv[0] == "sacct" else "980960|RUNNING\n"
    with pytest.raises(ValueError):
        helper.predecessor_guard(b["payloads"]["raw"], prior.recon, live, {})
    (Path(roots["raw"]) / "intent.json").write_text("{}")
    with pytest.raises(ValueError):
        helper.predecessor_guard(b["payloads"]["raw"], prior.recon, output, {})


def test_corrected_cases_cannot_replace_completed_orphan_and_unknown_terminal(monkeypatch):
    helper = suffix_from(bundle(monkeypatch))
    for inv in ({"complete": {1740: {}}, "orphan_caches": {}},
                {"complete": {}, "orphan_caches": {1850: {}}}):
        with pytest.raises(ValueError, match="already accepted"):
            helper.require_unsealed(inv, {1740: {}, 1850: {}})
    chain = {"raw": {"root": "HTTP", "actual_terminal": {"job": 1}, "ancestor_chain": {
        "raw": {"root": "capacity", "actual_terminal": {"job": 2}}}}}
    assert helper.buffer_terminal(chain, "capacity", {}) == {"job": 2}
    with pytest.raises(ValueError, match="no exact terminal"):
        helper.buffer_terminal(chain, "wrong", {})


def test_once_only_owner_transitions_do_not_retry_uncertain_mutations():
    jobs = {j: {"state": "FAILED", "exit_code": "1:0", "queue_empty": True} for j in owner.ORIGINALS.values()}
    assert owner.next_action(jobs, {}) == ("dispatch", "raw")
    with pytest.raises(AssertionError):
        owner.next_action(jobs, {"raw": {"returncode": 1, "job_id": "123"}})
