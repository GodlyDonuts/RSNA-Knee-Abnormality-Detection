"""Only host/transport changes; unchanged original numerical recovery functions."""

import ast
from pathlib import Path

D = Path("initiative/research/teacher-pair-earlier45-20261020-v1")
OLD = Path("initiative/research/teacher-pair-real-recovery48-20261019-v2")


def functions(source):
    return {
        n.name: ast.dump(n)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name != "main"
    }


def test_numerical_worker_run_is_identical():
    assert functions((D / "worker.py.txt").read_text()) == functions(
        (OLD / "worker.py.txt").read_text()
    )


def test_complete_state_and_prediction_decoder_identical():
    assert functions((D / "driver.py.txt").read_text()) == functions(
        (OLD / "driver.py.txt").read_text()
    )


def test_original32_input_math_byte_identical():
    assert (D / "inputs32.py.txt").read_bytes() == Path(
        "initiative/research/teacher-pair-earlier26-20261019-v1/inputs32.py.txt"
    ).read_bytes()


def test_same_pending_old_owner_must_be_held_before_dispatch():
    source = (D / "relocate_once.py.txt").read_text()
    assert source.index("scontrol hold 974412") < source.index("t.envelope(")
    assert "RunTime=00:00:00" in source and "Reason=JobHeldUser" in source
    assert "scancel" not in source and "automatic_retry" in source


def test_both_assigned_GPU_wrappers_bind_new_host_and_original_gradient():
    source = (D / "stage_recovery.py.txt").read_text()
    assert (
        "--nodelist=evc45" in source
        and "--partition=preemptable" in source
        and "--time-min=00:15:00" in source
    )
    assert "973601|COMPLETED|0:0" in source and "974412" in source
    for name in ("worker.py.txt", "driver.py.txt", "stage_recovery.py.txt", "relocate_once.py.txt"):
        compile((D / name).read_text(), name, "exec")
