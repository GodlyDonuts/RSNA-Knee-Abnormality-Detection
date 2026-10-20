"""Single G100 coordinator using pinned original operational controls.

Status is read-only. No phase runs automatically, no pilot is repeated, and a
full run needs actual pilot review, complete original inputs and new funding.
"""

import importlib.util
import subprocess
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace():
    derive = load("G100_full_derive", HERE / "derive.py")
    controls = load("G100_full_controls", HERE / "controls.py")
    contract = derive.local_contract(REPO)
    texts = controls.sources(REPO, contract)
    admission = types.ModuleType("G100_admission")
    exec(compile(texts["admission.py"], "derived_G100_admission", "exec"), admission.__dict__)
    a = admission
    source = texts["coordinator.py"]
    anchor = "import admission as a\nimport derive\n"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "a = ADMISSION\nderive = DERIVATION\n", 1)
    ns = {
        "__file__": str(HERE / "coordinator.py"),
        "__name__": "G100_operations_coordinator",
        "ADMISSION": admission,
        "DERIVATION": derive,
    }
    exec(compile(source, "derived_G100_coordinator", "exec"), ns)

    def git(pin, name):
        return subprocess.check_output(["git", "show", pin + ":" + name], cwd=REPO)

    def frozen(args):
        for name in ("coordinator.py", "controls.py", "derive.py", "contract.py"):
            path = HERE / name
            a.require(
                git(args.source_git, path.relative_to(REPO).as_posix()) == path.read_bytes(),
                "Invoke the exact committed G100 coordinator source",
            )
        files = derive.build(REPO)
        placement = git(ns["PLACEMENT_PIN"], ns["PLACEMENT_PATH"]).decode()
        payload = dict(
            source_git=args.source_git,
            files=files,
            code_sha256={n: a.sha(v.encode()) for n, v in files.items()},
            placement_source=placement,
            placement_sha256=a.sha(placement.encode()),
        )
        program = (
            "import sys,json,types\n"
            "a=types.ModuleType('admission');sys.modules['admission']=a\n"
            f"exec(compile({texts['admission.py']!r},'admission.py','exec'),a.__dict__)\n"
            "o=types.ModuleType('operations');sys.modules['operations']=o\n"
            f"exec(compile({texts['operations.py']!r},'operations.py','exec'),o.__dict__)\n"
            "v=json.load(sys.stdin)\n"
            "try:\n"
            " result=a.main(v) if v['phase']=='admit' else o.main(v)\n"
            " print(json.dumps(result,allow_nan=False),flush=True)\n"
            "except BaseException as error:\n"
            " tb=error.__traceback__\n"
            " while tb.tb_next:tb=tb.tb_next\n"
            " print(json.dumps({'error_type':type(error).__name__,'site_line':tb.tb_lineno}),file=sys.stderr,flush=True)\n"
            " raise SystemExit(1)\n"
        )
        transport = types.ModuleType("G100_stdin_transport")
        name = "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
        exec(compile(git(a.BASE, name), "pinned_stdin_transport", "exec"), transport.__dict__)
        return payload, program, transport

    ns["frozen"] = frozen
    return ns


if __name__ == "__main__":
    namespace()["main"]()
