# Exact public-label provenance: RSNA active overlay

Research checked 2026-10-14, updated with live publisher-card verification at approximately 18:46 UTC. Repository evidence is pinned to `8300cfd20c2acaeef9d684a81dbc7fe977282a62`. This is a metadata/documentation review, not an audit of patient rows or a new training experiment.

## Decision

**The exact identity, declared report-label parents, and publisher's CC0 declaration are established by the combined receipt and live-card evidence.** The active table is a recorded byte-identical mirror of Steven's Kaggle version-6 v4 probability-space blend of his v2 report labels and lixin73's GPT-5.6-Sol report labels. Its upstream synovitis procedure was evaluated/guided using the official 58 cases. No MRI-teacher parent is documented. Remaining gaps concern exact reproducibility and upstream rights/evaluation exposure, rather than an entirely unknown source or missing publisher license.

This is more informative than either “unknown source because Hugging Face lacks a card” or “report-only and independent because the filename says LLM.” In particular, the official 58-study gold set is **not a pristine independent evaluation of this label-generation procedure**. That conclusion does not itself prove MRI-teacher contamination of the project's outer fold or confirmation partition.

## 1. Exact object and recorded chain

- HF dataset: `Shio-Koube/knee3`
- HF revision: `373f4e1daf36984fa48bf3fc8fd9883fe17cb584`
- Member: `llm_labels_v4_blend.csv`
- File SHA-256: `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`
- File bytes: `630153`
- Recorded table size: 4,407 studies, one UID plus twelve target columns
- Upstream dataset identified by the existing receipts: `stevenleehans/rsna-knee-llm-report-labels`, version **6**

The pinned [September 29 source catalog](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/sources.json) contains three mutually consistent entries:

1. `confirmed_lineage` explicitly names both repositories, version 6, `same_downloaded_bytes: true`, and the full SHA above
2. HF `download_receipts` records the exact pinned revision, member name, bytes and SHA
3. `kaggle_source_receipts` records version 6, the same member name/bytes/SHA, and `license_declared: CC0: Public Domain`

The [October 7 collection receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-solution-intelligence-20261007-v1/collection.json) independently records a retrieval of the same Kaggle dataset version 6, public status and CC0 declaration, and the same member hash. Its archive is 426,025 bytes, SHA-256 `573f11c299a30b15c13f0f19368543f2d0ee81810809541b188ed83b359a9aa0`.

That archive's recorded members are:

- `llm_labels_full.csv`, 534,442 bytes, SHA-256 `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`
- `llm_labels_v2.csv`, 548,231 bytes, SHA-256 `3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144`
- `llm_labels_v4_blend.csv`, 630,153 bytes, SHA-256 `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`

The [active comparison provenance](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/experiments/competitive-v1/public-label-source-v1/provenance.json) binds that same CSV and HF revision to the 2,247-study fitting overlay, SHA-256 `c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51`. It records 1,156 outer studies, zero missing fitting studies, and no protected/outer labels in the local overlay.

**Evidence level:** these are durable, pinned project records of earlier full-download verification. I read the receipts, not the original CSV bytes, and did not independently repeat their hash computation. The evidence is stronger than a filename association, but it is not a newly obtained publisher-signed provenance attestation.

## 2. Generation ancestry and what “gold-guided” means

The same source catalog records the v4 parents as the author's v2 extraction and lixin73's GPT-5.6-Sol extraction. It explicitly records `gold_guided_changes_declared: true` and `synovitis_proxy_fill_from_effusion_declared: true`. The [source-audit README](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/README.md) describes the same byte match and upstream changes. The [adaptation/supervision plan](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/docs/dinov3-adapter-and-supervision-plan.md) specifies that those changes were guided by the **official 58 cases**.

**Live primary-source verification, 2026-10-14 approximately 18:46 UTC:** the documentation-only browser review opened [the version-6 URL](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6), which redirected to the [canonical Data Card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels). The actual UI identified **Version 6**, and the expanded card displayed **CC0: Public Domain**. It names `llm_labels_v4_blend.csv` as the recommended table: the author's v2 plus lixin73 GPT-5.6-Sol **report-label probabilities**, blended in probability space, not rank space. Exact mixing coefficients are not disclosed. The provenance says an LLM reads the competition Report column and declares no outside competition data. V2 changes only undecided synovitis using report-derived effusion, preserving explicit synovitis statements. Its diagnosis/evaluation uses the official 58 cases. The reported v4 **0.8927** is label macro AUC against those cases, **not an image-model leaderboard score**. No MRI-teacher parent is documented.

Additional primary context in the indexed [Steven Kaggle discussion profile](https://www.kaggle.com/stevenleehans/discussion) distinguishes a rejected v3 experiment: it fit cross-label ridge imputation on the report-label corpus, using gold to score. That separate v3 procedure should not be silently attributed to the recommended v4, whose named parent is v2.

Important distinctions:

- Gold-guided **selection or postprocessing** can compromise independence without any gold row being directly inserted into optimizer batches
- Report-derived weak supervision is not equivalent to human MRI ground truth
- An author's description of report extraction does not document every parent model, prompt, transformation, or prior evaluation exposure
- The current evidence does **not** establish that v4 copies gold values verbatim, uses a model trained on the project's outer images, or exposes any of the project's 74-case/report review sets. Do not assert those stronger claims

The October 7 receipt also pins lixin73's dataset version 1 and its CC0 declaration, including `labels_llm_gpt56sol.csv` SHA-256 `a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae`. It does not demonstrate which exact lixin parent bytes entered v4 or supply a reconstructible v4 generator.

The documentation-only browser follow-up completed at approximately 18:48 UTC. The linked [lixin73 card](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56) confirms the GPT-5.6-Sol report-label title and CC0, but has no dataset description; its prompt/settings are not recovered. Steven's version-6 history dates publication to **2026-08-09 06:16:46 UTC** (August 8, 23:16:46 PDT) and explicitly records the v4 probability-space blend. Dataset version **6** and label-recipe version **4** are separate version numbers.

The expanded Steven card also recommends per-cell confidence weighting `2*abs(p-0.5)` and defines original-extraction 0.5 as not addressed. This is a documented source convention, not proof that distance from 0.5 is calibrated clinical confidence. Because v4 mixes sources, its final numeric 0.5 need not identify the original missingness state. The exact proxy formula, mixture coefficients and missing-cell combination rules remain undisclosed. Do not reconstruct missingness masks or change losses solely from equality to 0.5.

## 3. Why the later HF metadata audit does not erase this chain

The [October 8 metadata receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-card-20261008-v1/metadata.json) reports the exact HF revision public/ungated, `card_data: null`, and no matching documentation files. Its [README](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-card-20261008-v1/README.md) correctly leaves annotation origin and terms unresolved from **HF metadata alone**. It does not invalidate the earlier cross-host byte receipts.

Conversely, `RSNA_competition_reports_only: true` in [reviewed inference code](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/scripts/kaggle/reviewed_infer.py) is an admission assertion, not new upstream evidence. The local comparison's `gold_informed_source_blend_or_tier_imported: false` and README statement that no gold-selected tier is introduced can describe the absence of a **new local** gold-selected blend. They cannot certify that the imported source has no upstream gold guidance. Preserve that distinction in future provenance metadata.

## 4. Consequences for nested teacher validation

1. **Existing local row exclusion remains meaningful.** Fitting-only overlay admission prevents direct import of outer/protected source rows through that loader. It does not undo global upstream tuning or establish how a hypothetical image teacher was trained
2. **Official-58 results are diagnostics, not independent confirmation of v4.** The public procedure used those cases for evaluation/selection. Removing the 58 from the local fit does not make the source-generation choices independent of them
3. **Do not infer outer-image leakage without evidence.** The publisher names report extractions and label-level postprocessing. No image-teacher parent is documented. The missing exact generator/parent pins prevent a full reproducibility audit; they are not positive evidence of an image teacher
4. **For a new image-teacher/student experiment, nest the entire teacher path.** Every teacher fit, image adaptation, calibration, blend choice, threshold and pseudo-label generation must exclude the student's outer leakage components. Inner OOF targets must be generated inside that outer-training pool. Globally produced OOF predictions may still encode outer-label information through teachers used on other training rows
5. **A teacher trained on this exact v4 is not an independent annotator.** Compare it with observed-only/control supervision on unchanged labels/masks; do not count agreement with its own ancestor table as independent corroboration. Preserve missingness semantics and separately document gold-exposed recipe choices
6. **A final all-development model can still be evaluated by genuinely unseen competition data.** Upstream gold exposure is a limit on internal validation claims; it does not alone prove the final submission invalid or erase a held-out leaderboard result

## 5. License and competition redistribution

**Established:** two pinned receipts record CC0 for the exact byte-matched Kaggle version 6, and the live version-6 publisher Data Card independently displays CC0: Public Domain. **Not established:** an HF license declaration at the pinned revision, the complete licensing chain for every generation input, or an unrestricted right to redistribute underlying competition reports/images/identifiers. The absent HF card is no longer a reason to describe the exact table as having no known publisher license.

[Creative Commons' CC0 explanation](https://creativecommons.org/publicdomain/zero/1.0/) says the affirmer waives their rights and permits reuse, while other persons' rights and privacy/publicity restrictions remain; it gives no warranty that a particular work is clear of third-party rights. A downstream CC0 declaration therefore cannot alone cancel upstream competition restrictions.

The pinned [competition contract audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/docs/competition-contract-20260920.md) records restrictions on redistribution of competition data to nonparticipants, a reference to RSNA's MIRA terms, and separate winner reproducibility/weight-publication obligations. The live [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) did not render substantive text in this web session, so the September 20 audit is not a substitute for a fresh rules review before publication.

Operational recommendation: retain source attribution, exact versions/hashes and private receipts. Refer participants to the official competition and upstream dataset rather than republishing source rows. Treat training use, competition eligibility, release of learned weights, and redistribution of label CSVs or reports as separate decisions. This review grants no new external-sharing authorization and reaches no definitive legal conclusion on derived-label redistribution.

## 6. Remaining proof obligations

The version-6 Data Card now closes the high-level parent/method/license questions. Remaining reproducibility and distribution questions are narrower:

- [Steven dataset](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), version 6; exact v4 coefficients, parent versions/hashes, complete generator/model/prompt, and full evaluation-exposure record
- [lixin73 parent dataset](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56); documented extraction model/prompt/input modality and exact member used by v4
- [Pinned HF tree](https://huggingface.co/datasets/Shio-Koube/knee3/tree/373f4e1daf36984fa48bf3fc8fd9883fe17cb584); upload commit message or publisher attribution, if any
- A reconstructible parent-generation chain consistent with the declared report-only inputs; if later evidence reveals image predictions, teacher training membership and cross-fitting lineage would then be needed
- Current competition/MIRA permission applicable to derived-label and weight redistribution

No public primary generator or exact upstream parent-version mapping was recovered. Thus **the exact table is linked to a live CC0-declared report-label blend with documented official-58 exposure; exact reproducibility and broad redistribution clearance remain open**. Report-only origin is supported by the publisher's explicit declaration. Do not convert either that declaration into guaranteed validation independence or the remaining implementation gaps into a claim of MRI-teacher contamination.

## Access and scope log

GitHub search/fetch_file worked; GitHub generic fetch failed with transport errors. HF metadata/tree/history and version-6 Kaggle API/card requests were inaccessible or empty through the web reader. A separate documentation-only cloud-browser review subsequently verified the expanded live publisher Data Card and version/license UI; it did not open Data Explorer or CSVs. No source CSV, image, checkpoint, clinical record, gold-label file or secret was intentionally opened or downloaded, and no training or external mutation was performed. One HF landing-page response and some search snippets unexpectedly included automatic dataset/case previews; those contents were not analyzed, quoted, retained in this artifact, or used for any conclusion. Subsequent HF requests were limited to metadata/tree/history paths.
