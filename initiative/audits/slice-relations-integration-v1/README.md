# Opt-in relation-loss integration, October1

The model now optionally returns raw-feature slice-distance loss alongside its
twelve disease logits. The trainer normalizes it across the complete effective
batch/DDP world, maintains separate classification/relation logs and persists
the policy, learned auxiliary head, AdamW state and per-rank RNG in checkpoints.
Inference reconstructs the head for strict checkpoint partition validation but
does not execute it. Removing the dormant head after restoration preserves
classification logits in the synthetic engineering check; no transformed
deployment export or official-weight runtime acceptance is claimed.

Exact prepared controls and remaining admission gates are in
`../../experiments/competitive-v1/slice-relations-v1/README.md`. Existing live
sampling source/configs/jobs and their private epoch publishers remain unchanged.
No relation GPU job, real-MRI relation training or competition submission was
made for this milestone. No external data was added.

The focused integration checks exercise actual LoRA modules, AdamW, frozen-base
delta files and CPU two-process DDP. Trainer recovery tests additionally cover
physical/shuffled/zero policies, mid-epoch and externally requested stops, both
outer-fold and all-development modes. Inference runs on synthetic MRI-cache
fixtures with the auxiliary forward forbidden. These checks show implementation
consistency, not clinical accuracy, H100 performance or Kaggle acceptance.

An initial disk-checkpoint test fixture omitted schema-format/world-size binding
fields. The strict loader correctly refused it; the fixture was corrected without
relaxing the loader. An attempted focused test invocation referenced a nonexistent
filename and ran no tests; the correct focused suite then passed79 tests. The
extended integration suite passed14 before adding the final seed/config gates;
the terminal full-suite receipt is the authoritative final QA result.

Current training checkpoints newly preserved by their original coordinator:
uniform epoch1 at private HFfcf7073ea2c1d1f29d2a2e2979758c9780f7a96c
and stratified epoch2 at HF3c384fb5e26298c7614a253504f37b8dc0e96faf.
All five objects in each publication were fully re-downloaded/size/SHA verified.
CPU publishers866759 and866760 completed0:0. Original first stratified epoch
HF227bdb38dd0fa5f2bd18422acd4876912ce647ab remains preserved and was not repeated.
See `../sampling-epochs-v1/` for exact original sidecars and publication journals.
