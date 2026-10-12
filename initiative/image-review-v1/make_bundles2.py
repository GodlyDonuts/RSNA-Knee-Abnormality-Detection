"""Rebuild bundles: montage JPEGs + enriched studies.json (report + priors)."""

import json
import zipfile
from pathlib import Path

ROOT = Path("/home/ubuntu/rsna-review")
MONT = ROOT / "montages"
OUT = ROOT / "bundles2"
OUT.mkdir(exist_ok=True)
batches = json.loads((ROOT / "manifest/batches.json").read_text())
enr = json.loads((ROOT / "manifest/enriched.json").read_text())

for b in batches:
    out = OUT / f"{b['batch_id']}.zip"
    if out.exists():
        continue
    studies_meta = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for study in b["studies"]:
            d = MONT / study
            idx = json.loads((d / "index.json").read_text())
            meta = {"study_uid": study, "side": idx["side"], "slots": idx["slots"]}
            meta.update(enr[study])
            studies_meta.append(meta)
            for jpg in sorted(d.glob("*.jpg")):
                z.write(jpg, f"{study}/{jpg.name}")
        z.writestr("studies.json", json.dumps(studies_meta))
    print(out.name, f"{out.stat().st_size / 1e6:.1f}MB", flush=True)
print("done")
