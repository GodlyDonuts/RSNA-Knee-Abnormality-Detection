"""Prepare, but do not dispatch, the ten-epoch fold-2 single-H100 run."""

from __future__ import annotations

import json
from pathlib import Path

from build import one

ROOT = Path(__file__).resolve().parent


def build() -> None:
    original = json.loads((ROOT / "fold1-reference.json").read_text())
    assert original["fold"] == 1 and original["accumulation_steps"] == 4
    assert original["batch_size"] == 1 and original["epochs"] == 10
    assert original["diagnostic_study_limit"] is None
    production = dict(original)
    production.update(fold=2, accumulation_steps=8)
    (ROOT / "production-config.json").write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )

    script = (ROOT / "pilot.sbatch").read_text()
    for old, new in (
        ("rsna-dinov3-fold2-1gpu-pilot", "rsna-dinov3-fold2-1gpu-production"),
        ("#SBATCH --nodelist=evc32", "#SBATCH --exclude=evc27,evc38,evc43,evc44,evc49,evc50"),
        ("#SBATCH --time=01:00:00", "#SBATCH --time=08:00:00"),
    ):
        script = one(old, new, script)
    assert script.count("pilot-config.json") == 2
    script = script.replace("pilot-config.json", "production-config.json")
    assert script.count("dinov3-l16-fold2-b74060b-1gpu-pilot-v1") == 2
    script = script.replace(
        "dinov3-l16-fold2-b74060b-1gpu-pilot-v1",
        "dinov3-l16-fold2-b74060b-1gpu-production-v1",
    )
    if "1gpu-pilot" in script:
        raise ValueError("Pilot output remains in production launcher")
    (ROOT / "production.sbatch").write_text(script)

    watcher = (
        ROOT.parent / "dinov3-fold1-recovery-v1/retry-843935/archive_odd_epochs.py"
    ).read_text()
    watcher = one('config.get("fold") != 1', 'config.get("fold") != 2', watcher)
    watcher = one('"fold": 1,', '"fold": 2,', watcher)
    (ROOT / "archive_odd_epochs.py").write_text(watcher)


if __name__ == "__main__":
    build()
