# Steven v6 / current public-label overlay: provenance correction

Verified 2026-10-14 at approximately18:45–18:48UTC. Documentation-only inspection of the public Data Card and version-history notes; no CSV, Data Explorer rows, report text, imaging data or participant-level records were read or downloaded.

## Result

The current public overlay has a documented **report-label ensemble lineage**, rather than wholly unknown ancestry. Repository receipts establish that the deployed Shio-Koube file is byte-identical to Steven dataset version6's `llm_labels_v4_blend.csv`. The live Steven card identifies this as a probability-space blend of Steven v2 and lixin73's GPT-5.6-Sol report labels. It does **not** document an MRI image-prediction teacher. The public dataset and named second parent display **CC0**.

This changes the strategy diagnosis: a proposed fold-safe MRI-teacher experiment would add a different kind of signal from the current two-LLM report-target blend. It would not merely add another text extractor. However, Steven's target engineering and comparisons were guided by Gold58, so these targets are not independent of that validation set.

## 1. Exact file identity, from existing repository receipts

Pinned repository commit: `8300cfd20c2acaeef9d684a81dbc7fe977282a62`.

- `initiative/audits/adapter-source-research-v1/sources.json`, `confirmed_lineage`, links Hugging Face `Shio-Koube/knee3` revision `373f4e1daf36984fa48bf3fc8fd9883fe17cb584`, member `llm_labels_v4_blend.csv`, to Steven Kaggle dataset version6
- Member byte count: **630153**
- Member SHA256: **eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4**
- Independent collection receipt: `initiative/audits/public-solution-intelligence-20261007-v1/collection.json` records the same v6 member/hash and CC0
- Recorded source archive SHA256: `573f11c299a30b15c13f0f19368543f2d0ee81810809541b188ed83b359a9aa0`

The dedicated provenance audit inspected these receipts; this follow-up did not redownload files or independently rehash clinical target rows. The separate consolidated audit is `/workspace/shared/rsna-research-ledger-20261014/label-provenance.md`.

## 2. Live primary documentation and version confirmation

[Steven dataset version6 entry](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6) redirected to the [canonical card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels). The live version-history control explicitly showed **Version6**, so the page was not assumed to be an arbitrary latest version.

Version6 history note describes the v4 probability-space blend with independent lixin73 labels, evaluated at0.8927 against gold. Publication timestamp: **August8,2026 23:16:46 PDT**, equivalent to **August9,2026 06:16:46 UTC**. Earlier version2 note says undecided Synovitis cells were filled from Effusion. Version3/4 add documentation; version5 marks publication. Kaggle dataset version6 and label recipe v4 are different versioning systems.

The expanded card identifies:

1. `llm_labels_full.csv`: original v1
2. `llm_labels_v2.csv`: author's revised labels
3. `llm_labels_v4_blend.csv`: recommended v2 + lixin73 GPT-5.6-Sol blend

The named [lixin73 parent card](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56) is titled “RSNA Knee LLM Report Labels (GPT-5.6-Sol)” and displays CC0, but currently has no description. It does not independently disclose extraction prompt, model settings or processing details.

## 3. What v2 and v4 actually do

### Report extraction

The Steven provenance paragraph says an LLM reads the competition's training `Report` field and returns a probability for each of12 findings, with an explicit not-addressed option. It says no external data was used. It does not name the author's own v1/v2 LLM or provide its exact prompt/settings in the inspected card.

The card assigns **0.5 to not-addressed findings**, rather than meaning calibrated50% clinical risk. It recommends confidence weighting `2*abs(p−0.5)`, which gives these undecided cells zero weight. It reports25.4% of v1 cells exactly0.5. This is the creator's recommended semantics, not proof our current training loss applies that weight.

### v2 Synovitis proxy

The author compared report Synovitis and report Effusion against Gold58 and found Effusion informative for Synovitis. V2 fills **only undecided Synovitis cells** from the Effusion field, preserving explicit Synovitis statements. The version-history note calls out83.7% of the original Synovitis corpus as undecided. The card reports Synovitis label AUC improving0.678→0.790 and12-target label macro-AUC0.8780→0.8873.

This is a **cross-target report-derived proxy**, not image pseudo-labeling. The documentation does not fully specify whether the transferred probability is copied directly or transformed by a formula, so exact numerical reconstruction remains open.

### v4 blend

The card says v4 blends Steven v2 with lixin73 GPT-5.6-Sol label probabilities. It compares Steven v2 (0.8873), Pilkwang (0.8700), lixin73 (0.8347), and the chosen v4 blend (0.8927) against Gold58. The reported rank correlations with Steven are0.834 for Pilkwang and0.718 for lixin73, motivating the latter's complementary errors despite lower standalone label AUC.

The operation is explicitly **probability-space blending**, not percentile-rank blending. The creator warns that rank conversion would destroy the intended relation between `2*abs(p−0.5)` and confidence.

**Undisclosed:** the exact scalar or per-target mixture coefficients, alignment/missing-cell branch rules, clipping, source-preprocessing details, author-v2 LLM identity/prompt and a complete reproducible v4 generation script. Do not claim a50/50 blend merely because it is a blend. No MRI teacher or out-of-fold image-prediction parent appears in the documented lineage; absent undisclosed processing, this is report-target ensembling.

## 4. Gold58 independence and licensing

The card explicitly describes checking report targets on the58 annotated studies, choosing a Synovitis proxy through those comparisons and comparing alternate blend parents there. These are not independent held-out performance estimates. The0.8927 figure is **label-vs-gold AUC**, not a trained MRI model's public leaderboard score. The card itself warns that a better label metric need not yield a better image model, and that an initial label swap failed to help until other pipeline issues were fixed.

Both live cards display [CC0 Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/). Combined with exact-byte source receipts, lack of a separate license card on the Hugging Face mirror no longer supports saying the underlying labels have no located license. The public source declares CC0; that is distinct from a guarantee that it can waive competition-data obligations or any third-party rights. Do not extend the declaration to report text, images or unrelated assets.

The project's protected-gold policy remains relevant: excluding Gold58 rows from our fit does not remove upstream target-engineering choices informed by Gold58. This does not prove every fitting cell contains a gold label; it establishes selection dependence at the recipe level.

## 5. Corrections to the high-score report and next audit

- Replace “current overlay ancestry unknown” with **“verified report-probability blend ancestry; exact extraction settings and mixture implementation incomplete”**
- Replace “no located source license” with **“byte-matched upstream Steven v6 declares CC0; mirror itself lacks a license card”**
- Retain uncertainty about exact blend coefficients, authors' own model/prompt and complete generation reproducibility
- Distinguish our current two-report-labeller blend from colum/Archit/Raymond's MRI image-pseudo-label training
- **Current weighting mismatch verified by the constraint audit:** the full-public overlay's audited2247-row configuration gives all26964 fitting cells mask/weight1. It does not apply `2*abs(p−0.5)`. `effective_batch_loss` averages, across active targets, each target's masked BCE sum divided by its effective-batch mask sum; that path has no separate positive/class weighting. Thus a target0.5 contributes a full-weight gradient toward0.5 rather than being ignored. Evidence: `initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/public-run.json`, `rsna/competitive/label_source.py` and `rsna/competitive/loss.py`, pinned8300cfd. A separate teacher-confidence helper is not the production overlay. These are the2247-row configuration's receipt counts, not a freshly computed all3403-row count
- Compare source-faithful confidence weighting with full-weight soft BCE under a fixed mask, split and model. This is an experimentally testable semantic difference, **not automatically a bug fix**: full-weight uncertain labels may regularize, and confidence weighting may discard useful weak signal or amplify extraction bias. Do not conflate per-cell confidence with class positive weighting or change effective-batch normalization inadvertently
- Keep the existing protected data boundaries and evaluate new image-teacher strategies without using Gold58 to pick target weights or thresholds

