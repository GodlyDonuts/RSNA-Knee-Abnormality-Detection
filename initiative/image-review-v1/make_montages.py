"""Render per-channel montage JPEGs for every study .npy in npz/p1,npz/p2.

Output: montages/<study>/<SLOT>.jpg plus index.json with side + slots.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SRC = Path("/home/ubuntu/rsna-review/npz")
DST = Path("/home/ubuntu/rsna-review/montages")
PAD = 6
COLS = 4
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
         "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]


def montage(arr: np.ndarray, label: str, out: Path):
    n, h, w = arr.shape
    rows = (n + COLS - 1) // COLS
    canvas = Image.new("L", (COLS * w + (COLS + 1) * PAD,
                            rows * (h + 18) + PAD), 0)
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        r, c = divmod(i, COLS)
        x = PAD + c * (w + PAD)
        y = PAD + r * (h + 18)
        canvas.paste(Image.fromarray(arr[i]), (x, y + 14))
        draw.text((x + 2, y), f"{label}[{i}]", fill=255)
    canvas.convert("L").save(out, quality=95)


def render_one(args):
    npy_path, side = args
    p = Path(npy_path)
    study = p.stem
    out_dir = DST / study
    idx = out_dir / "index.json"
    if idx.exists():
        return study, "cached"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        a = np.load(p)
        assert a.shape[0] == 6
        for i, slot in enumerate(SLOTS):
            montage(a[i], slot, out_dir / f"{slot}.jpg")
        idx.write_text(json.dumps({"study_uid": study, "slots": SLOTS,
                                   "side": side if isinstance(side, str) else ""}))
        return study, "ok"
    except Exception as e:
        return study, f"error:{e}"


if __name__ == "__main__":
    m = pd.concat([pd.read_csv("npz/p1/manifest.csv"),
                   pd.read_csv("npz/p2/manifest.csv")])
    side_of = dict(zip(m.StudyInstanceUID, m.side))
    files = sorted(SRC.rglob("*.npy"))
    args = [(str(p), side_of.get(p.stem, "")) for p in files]
    print(f"{len(args)} studies")
    ok = err = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for study, status in ex.map(render_one, args):
            if status in ("ok", "cached"):
                ok += 1
            else:
                err += 1
                print(study, status, file=sys.stderr)
    print(f"done ok={ok} err={err}")
