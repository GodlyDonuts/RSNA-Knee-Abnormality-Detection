"""Explicit Colab final admission/deployment, no owner loop or mutation retries."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def namespace():
    spec = importlib.util.spec_from_file_location("Colab_deployment_controls", HERE / "controls.py")
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    original = ctl.git(ctl.ORIGINAL + "run_once.py")
    original = ctl.once(
        original, "own = controls(args.source_git)", "own = controls(args.source_git, value)"
    )
    original = ctl.once(
        original,
        'ACCOUNT_PHASES = ("dataset", "dataset-reconcile", "push", "status", "submit", "outcome")',
        'ACCOUNT_PHASES = ("adopt", "push", "status", "submit", "outcome")',
    )
    original = ctl.once(
        original, 'choices=("qualification", "B", "C", "C0")', 'choices=("qualification", "C")'
    )
    original = ctl.once(
        original,
        'target = "cpu.py" if args.phase in CPU_PHASES else "account.py"',
        'target = "adopt.py" if args.phase == "adopt" else ("cpu.py" if args.phase in CPU_PHASES else "account.py")',
    )
    original = ctl.once(
        original,
        '"--job-name=rsna-predictive-final-" + args.phase',
        '"--job-name=rsna-Colab-final-" + args.phase',
    )
    ns = {"__file__": str(HERE / "run_once.py"), "__name__": "Colab_explicit_single_phase"}
    exec(compile(original, "original_Colab_single_phase_coordinator", "exec"), ns)

    def controls(pin, value):
        paths = [HERE / n for n in ("controls.py", "run_once.py", "prepare.py", "adopt.py")]
        paths += [ctl.SERVING / n for n in ("compile.py", "contract.py")]
        paths += [
            ctl.STORAGE / n for n in ("colab_storage.py", "actual-A10040-checkpoint-lease.json")
        ]
        paths.append(HERE.parent / "predictive-density100-deployment-20261021-v1/primary_budget.py")
        for path in paths:
            if ctl.git(path.relative_to(REPO).as_posix(), pin) != path.read_text():
                raise ValueError("Execute the exact committed independent Colab controls")
        return ctl.build(value)

    ns["controls"] = controls
    return ns


if __name__ == "__main__":
    namespace()["main"]()
