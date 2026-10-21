"""One explicit C100 phase using the canonical account coordinator; no retries."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def namespace():
    spec = importlib.util.spec_from_file_location("C100_account_controls", HERE / "controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.git(module.ORIGINAL + "run_once.py")
    original = module.replace_once(
        original, "own = controls(args.source_git)", "own = controls(args.source_git, value)"
    )
    original = module.replace_once(
        original, 'choices=("qualification", "B", "C", "C0")', 'choices=("qualification", "C")'
    )
    original = module.replace_once(
        original,
        '"--job-name=rsna-predictive-final-" + args.phase',
        '"--job-name=rsna-predictive-C100-" + args.phase',
    )
    ns = {"__file__": str(HERE / "run_once.py"), "__name__": "C100_single_phase_coordinator"}
    exec(compile(original, "original_single_phase_C100_coordinator", "exec"), ns)

    def controls(pin, value):
        for path in (
            HERE / "run_once.py",
            HERE / "controls.py",
            HERE / "primary_budget.py",
            module.SERVING / "compile.py",
            module.SERVING / "contract.py",
        ):
            if module.git(path.relative_to(REPO).as_posix(), pin) != path.read_text():
                raise ValueError("Execute the exact committed C100 deployment source")
        return module.build(value)

    ns["controls"] = controls
    return ns


if __name__ == "__main__":
    namespace()["main"]()
