"""Prepare the full-data G100 worker/publisher; never allocate or admit a run."""

import importlib.util
import subprocess
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "473a9a94d90890e8a97e7ed46d157dcf417ad173"
C100 = "initiative/research/predictive-density100-full-20261020-v1/"
PILOT = "initiative/research/predictive-generative-routing-20261020-v1/"


def build(repo):
    repo = Path(repo)

    def git(name, pin=PIN):
        return subprocess.check_output(["git", "show", pin + ":" + name], cwd=repo)

    original = git(C100 + "derive.py").decode()
    base = types.ModuleType("pinned_C100_derivation")
    exec(compile(original, "pinned_C100_derivation", "exec"), base.__dict__)
    pilot_builder = types.ModuleType("pinned_G100_pilot_builder")
    exec(
        compile(git(PILOT + "worker_builder.py"), "pinned_G100_pilot_builder", "exec"),
        pilot_builder.__dict__,
    )
    raw = git(base.ORIGINAL + "worker.py.txt", base.BASE)
    worker = base.derive_worker(raw)
    worker = worker.replace("density100_contract", "generative_contract")
    worker = worker.replace("density100_telemetry", "generative_telemetry")
    worker = base.replace_once(
        worker,
        '        "generative_contract.py",\n',
        '        "generative_contract.py",\n        "base_C100_derivation.py",\n'
        '        "rsna/competitive/predictive/generative_routing.py",\n',
    )
    # Reuse the exact adapter used by original GPU986011. This only changes
    # the loss function inside this new isolated worker process.
    adapter = pilot_builder.ROUTING_ADAPTER
    expected = 'if role == "G100":\n    training_engine.two_mask_loss = routed_two_mask_loss\n'
    assert adapter.count(expected) == 1
    adapter = adapter.replace(
        expected,
        'assert binding["branch"] == "G100"\n'
        "training_engine.two_mask_loss = routed_two_mask_loss\n",
    )
    indented = "\n".join("    " + line if line else "" for line in adapter.splitlines()) + "\n"
    anchor = "    from rsna.competitive.predictive.train import Recipe, fit\n"
    worker = base.replace_once(worker, anchor, anchor + indented)

    publisher = base.derive_publisher(git(base.ORIGINAL + "publish_epoch.py.txt", base.BASE))
    publisher = publisher.replace("density100_contract", "generative_contract")
    publisher = publisher.replace("c100_provenance", "g100_provenance")
    publisher = base.replace_once(
        publisher,
        "    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n",
        "    module = importlib.util.module_from_spec(spec)\n"
        '    base_path = reference({"path": str(code / "base_C100_derivation.py"),\n'
        '        "sha256": binding["code_sha256"]["base_C100_derivation.py"]})\n'
        '    base_spec = importlib.util.spec_from_file_location("base_C100_derivation", base_path)\n'
        "    base_module = importlib.util.module_from_spec(base_spec)\n"
        "    base_spec.loader.exec_module(base_module)\n"
        '    previous_base = sys.modules.get("base_C100_derivation")\n'
        "    try:\n"
        '        sys.modules["base_C100_derivation"] = base_module\n'
        "        spec.loader.exec_module(module)\n"
        "    finally:\n"
        "        if previous_base is None:\n"
        '            del sys.modules["base_C100_derivation"]\n'
        "        else:\n"
        '            sys.modules["base_C100_derivation"] = previous_base\n',
    )
    publisher = publisher.replace('funding["branch"] == "C100"', 'funding["branch"] == "G100"')
    publisher = publisher.replace("predictive-C100-full3887-", "predictive-G100-full3887-")

    files = {}
    for name, digest in base.NUMERICAL_SHA.items():
        raw = (
            b""
            if name in ("rsna/__init__.py", "rsna/competitive/__init__.py")
            else git(name if name.startswith("rsna/") else base.ORIGINAL + name, base.BASE)
        )
        assert base.sha(raw) == digest
        files[name] = raw.decode()
    routing = git("rsna/competitive/predictive/generative_routing.py")
    assert base.sha(routing) == pilot_builder.ROUTING_SHA
    files.update(
        {
            "worker.py": worker,
            "publish_epoch.py": publisher,
            "base_C100_derivation.py": original,
            "generative_contract.py": (HERE / "contract.py").read_text(),
            "generative_telemetry.py": git(C100 + "telemetry.py").decode(),
            "rsna/competitive/predictive/generative_routing.py": routing.decode(),
        }
    )
    for name, value in files.items():
        compile(value, name, "exec")
    return files


def local_contract(repo):
    """Load the prospective contract for local review only, without deployment."""
    spec = importlib.util.spec_from_file_location("prospective_G100_contract", HERE / "contract.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    base = types.ModuleType("base_C100_derivation")
    raw = subprocess.check_output(["git", "show", PIN + ":" + C100 + "derive.py"], cwd=repo)
    exec(compile(raw, "pinned_C100_derivation", "exec"), base.__dict__)
    old = sys.modules.get("base_C100_derivation")
    try:
        sys.modules["base_C100_derivation"] = base
        spec.loader.exec_module(module)
    finally:
        if old is None:
            del sys.modules["base_C100_derivation"]
        else:
            sys.modules["base_C100_derivation"] = old
    return module
