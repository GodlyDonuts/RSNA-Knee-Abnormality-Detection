"""Aggregate image-review JSONL results, compare vs current labels + gold."""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/ubuntu/rsna-review")
REPO = Path("/home/ubuntu/repos/RSNA-Knee-Abnormality-Detection")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
STATES = ["positive", "negative", "uncertain", "insufficient_evidence"]

# current released labels (v2.1 family merged labels live in study_labels.csv)
cur = pd.read_csv(REPO / "initiative/releases/report-supervision-v1/study_labels.csv",
                  low_memory=False)
cur = cur.set_index("study_uid")
splits = json.loads((ROOT / "manifest/splits.json").read_text())

# gold labels (58 studies): from study_labels.csv rows split==gold_evaluation
# (its label columns carry the official gold values for those rows)
gold_rows = cur[cur.split == "gold_evaluation"]


def load_results(res_dir):
    rows = []
    for f in sorted(Path(res_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"bad line {f.name}: {e}", file=sys.stderr)
    return rows


def state_to_bin(s):
    return {"positive": 1.0, "negative": 0.0}.get(s)


def main(res_dir):
    rows = load_results(res_dir)
    print(f"{len(rows)} study reviews")
    seen = set()
    out = []
    gold_stats = {"n": 0}
    gold_target_stats = {t: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unc": 0}
                         for t in TARGETS}
    changes = []
    for r in rows:
        uid = r["study_uid"]
        if uid in seen:
            print(f"dup review {uid}", file=sys.stderr)
        seen.add(uid)
        rec = {"study_uid": uid, "split": splits.get(uid, "?"),
               "needs_full_res": bool(r.get("needs_full_res")),
               "notes": r.get("notes", "")}
        st = r["states"]
        ev = r.get("evidence", [""] * 12)
        for i, t in enumerate(TARGETS):
            rec[f"{t}__img_state"] = st[i] if i < len(st) else "insufficient_evidence"
            rec[f"{t}__img_ev"] = ev[i] if i < len(ev) else ""
            b = state_to_bin(st[i])
            rec[f"{t}__img"] = b
            if uid in cur.index:
                curv = cur.loc[uid, t]
                rec[f"{t}__cur"] = curv
                if b is not None and pd.notna(curv) and abs(b - curv) > 0.5:
                    changes.append((uid, t, curv, st[i]))
            if uid in gold_rows.index and b is not None:
                g = gold_rows.loc[uid, t]
                if pd.notna(g):
                    gs = gold_target_stats[t]
                    if g >= 0.5:
                        gs["tp" if b >= 0.5 else "fn"] += 1
                    else:
                        gs["fp" if b >= 0.5 else "tn"] += 1
                else:
                    gold_target_stats[t]["unc"] += 1
            elif uid in gold_rows.index and st[i] == "uncertain":
                gold_target_stats[t]["unc"] += 1
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results/image_review.csv", index=False)
    missing = set(cur.index) - seen
    print(f"unique studies reviewed {len(seen)}; missing {len(missing)}")
    if missing:
        Path(ROOT / "results/missing_studies.json").write_text(json.dumps(sorted(missing)))
    print(f"flips vs current labels: {len(changes)}")
    pd.DataFrame(changes, columns=["study_uid", "target", "current", "img_state"]).to_csv(
        ROOT / "results/flips_vs_current.csv", index=False)
    print("\nGold-58 reviewer performance (image review vs official gold):")
    print(f"{'target':<18} tp   fp   tn   fn   unc  sens  spec")
    sens_l, spec_l = [], []
    for t in TARGETS:
        g = gold_target_stats[t]
        sens = g["tp"] / max(1, g["tp"] + g["fn"])
        spec = g["tn"] / max(1, g["tn"] + g["fp"])
        if g["tp"] + g["fn"]:
            sens_l.append(sens)
        if g["tn"] + g["fp"]:
            spec_l.append(spec)
        print(f"{t:<18} {g['tp']:<4} {g['fp']:<4} {g['tn']:<4} {g['fn']:<4} {g['unc']:<4} "
              f"{sens:.2f}  {spec:.2f}")
    print(f"macro sens {sum(sens_l)/max(1,len(sens_l)):.3f}  macro spec {sum(spec_l)/max(1,len(spec_l)):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/rsna-review/results")
