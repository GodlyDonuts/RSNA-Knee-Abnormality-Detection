"""Deliver verified original Colab output to the current finite owner only."""

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = "initiative/research/colab-C100-deployment-owner-20261021-v1/collect_saved_output.py"
source = subprocess.check_output(
    ["git", "show", "c9fbbb20:" + ORIGINAL], cwd=ROOT, text=True
)
old = 'destination = ROOT / "artifacts" / HERE.name / "epoch2-successor" / h.FILENAME'
assert source.count(old) == 1
source = source.replace(old, 'destination = ROOT / "artifacts" / HERE.name / "owner" / h.FILENAME')
exec(compile(source, str(HERE / "collect_saved_output.py"), "exec"), globals())
