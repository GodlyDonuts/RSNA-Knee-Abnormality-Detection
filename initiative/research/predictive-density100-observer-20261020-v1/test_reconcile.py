import ast
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("terminal_reconciliation", HERE / "reconcile_terminal.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_original_decision_serializations_and_reject_changed_content():
    body = module.git_file(module.PILOT, "initiative/research/predictive-density100-pilot-20261020-v1/decision.json").encode()
    written = (json.dumps(json.loads(body), sort_keys=True, allow_nan=False) + "\n").encode()
    assert body != written
    assert module.exact_serializations(body, written, module.sha(body))["original_decision_unchanged"]
    for bad in (body, written.rstrip(), written.replace(b'100.0', b'101.0') if b'100.0' in written else written+b' '):
        with pytest.raises(AssertionError):
            module.exact_serializations(body, bad, module.sha(body))
    with pytest.raises(AssertionError):
        module.exact_serializations(body, written, module.sha(written))


def test_reconciliation_preserves_original_numerical_export_and_readonly_checks():
    source = module.program()
    ast.parse(source)
    for original in ('c.acceptance(value)', 'json.loads(original_terminal) == value["terminal"]',
                     'c.require(terminal["state"] in c.TERMINAL and terminal["live_queue_empty"]',
                     '"original_files_written": 0'):
        assert original in source
    assert 'out["scientific_decision_serialization"] = exact_serializations(' in source
    assert 'torch.load' not in source and 'sbatch' not in source
