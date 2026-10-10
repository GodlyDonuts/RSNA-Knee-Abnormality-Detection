"""Build deterministic review batches over all 4,407 train studies."""
import hashlib, json, random
from pathlib import Path
import pandas as pd

REPO = Path("/home/ubuntu/repos/RSNA-Knee-Abnormality-Detection")
OUT = Path("/home/ubuntu/rsna-review")
BATCH_SIZE = 30

labels = pd.read_csv(REPO / "initiative/releases/report-supervision-v1/study_labels.csv",
                     usecols=["study_uid", "split"], low_memory=False)
studies = labels.study_uid.tolist()
split_of = dict(zip(labels.study_uid, labels.split))
assert len(studies) == 4407 and len(set(studies)) == 4407

rng = random.Random(20261007)
order = studies[:]
rng.shuffle(order)

batches = []
for i in range(0, len(order), BATCH_SIZE):
    members = order[i:i + BATCH_SIZE]
    batches.append({
        "batch_id": f"img-{len(batches):03d}",
        "studies": members,
        "n": len(members),
        "n_gold": sum(1 for s in members if split_of[s] == "gold_evaluation"),
        "n_quarantine": sum(1 for s in members if split_of[s] == "quarantine"),
    })

(OUT / "manifest").mkdir(exist_ok=True)
(OUT / "manifest/batches.json").write_text(json.dumps(batches, indent=1))
(OUT / "manifest/splits.json").write_text(json.dumps(split_of, indent=0, sort_keys=True))
print(f"{len(batches)} batches; gold-in-batches {sum(b['n_gold'] for b in batches)}; "
      f"quarantine {sum(b['n_quarantine'] for b in batches)}")
