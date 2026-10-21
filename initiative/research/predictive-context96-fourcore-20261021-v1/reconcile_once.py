"""Read existing original output after observer-only missing-NumPy failure.

Reuses the unchanged original coordinator/payload, replacing only its output
reader with standard-library NPY decoding. Status-only: no prepare, push or entry.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = "f8d2380365e98666aa0d3afb3c3b754b8ebdeb10"


def main():
    assert len(sys.argv) == 1
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in ("reconcile.py", "reconcile_once.py"):
        rel = (HERE / name).relative_to(ROOT).as_posix()
        assert (HERE / name).read_text() == subprocess.check_output(
            ["git", "show", pin + ":" + rel], cwd=ROOT, text=True
        )
    s = importlib.util.spec_from_file_location("dense96_output_reader", HERE / "reconcile.py")
    repair = importlib.util.module_from_spec(s)
    s.loader.exec_module(repair)
    rel = HERE.relative_to(ROOT).as_posix()
    body = subprocess.check_output(
        ["git", "show", ORIGINAL + ":" + rel + "/run_once.py"], cwd=ROOT, text=True
    )
    anchor = '        control.derive(frozen("initiative/research/predictive-context96-T4-20261021-v1/account.py"))'
    # Formatting of the already frozen original is accepted exactly.
    if anchor not in body:
        anchor = '        control.derive(\n            frozen("initiative/research/predictive-context96-T4-20261021-v1/account.py")\n        )'
    assert body.count(anchor) == 1
    body = body.replace(anchor, anchor.replace("control.derive(", "repair.program(control, "))
    body = body.replace('choices=("prepare", "push", "status")', 'choices=("status",)')
    assert body.count('PAYLOAD = ROOT / "artifacts" / HERE.name') == 1
    body = body.replace(
        'PAYLOAD = ROOT / "artifacts" / HERE.name',
        'PAYLOAD = ROOT / "artifacts" / HERE.name / "observer-dependency-reconciliation"',
    )
    assert body.count('"source_git": args.source_git,') == 1
    body = body.replace(
        '"source_git": args.source_git,',
        '"source_git": args.source_git, "observer_source_git": ' + repr(pin) + ",",
    )
    # Helpers are loaded locally; staged source and original scientific hash are unchanged.
    namespace = {
        "__name__": "fourcore_readonly_reconciliation",
        "__file__": str(HERE / "run_once.py"),
        "repair": repair,
    }
    exec(compile(body, "original_status_with_dependency_free_reader", "exec"), namespace)
    sys.argv = [sys.argv[0], "status", "--source-git", ORIGINAL]
    namespace["main"]()


if __name__ == "__main__":
    main()
