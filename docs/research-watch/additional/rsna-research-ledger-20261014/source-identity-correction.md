# Correction addendum: active public-label source identity

**October 14, 2026. Applies to the experiment ledger delivered earlier today and the accompanying historical supplement.** The original files remain unchanged so their versions stay identifiable. This correction uses an existing source receipt at the same pinned snapshot 8300cfd20c2acaeef9d684a81dbc7fe977282a62; it is not a new label download or training result.

## Corrected conclusion

The active Shio-Koube/knee3 llm_labels_v4_blend.csv is **byte-identical to Steven's Kaggle version6 file**, according to the repository's September29 original-download receipt. It is incorrect to leave their identity as unknown or to imply that the active source's licensing/provenance evidence consists only of a missing Hugging Face card.

Verified identity:
- Hugging Face repository: Shio-Koube/knee3
- Immutable source revision:373f4e1daf36984fa48bf3fc8fd9883fe17cb584
- File:llm_labels_v4_blend.csv
- Size:630,153bytes;4,407rows
- SHA256:eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4
- Matching Kaggle source:stevenleehans/rsna-knee-llm-report-labels, version6
- Original receipt explicitly records same_downloaded_bytes=true
- That Kaggle source declares CC0/Public Domain

The same receipt records **gold-guided changes declared**, **Synovitis proxy filling from Effusion declared**, and two declared parents: author's v2 extraction and lixin73 GPT-5.6-Sol extraction. Exact parent transformations and full upstream derivation remain incompletely established. Known immediate identity must not be downgraded to unknown simply because deeper parent provenance needs work.

Primary repository evidence: [original sources.json](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/sources.json). Its checked_date is2026-09-29. Its historical diagnostic_only/not_admitted status describes that discovery stage, not the later training history.

## Connection to the current training and public-score artifacts

This is not merely a matching unused file in a repository. The current reviewed inference binding records the same source revision and source_csv_sha256. The all-development source-overlay worker asserts that exact SHA before constructing the3,403-study overlay. This connects the established Steven-v6 identity to the active public-supervision line.

[Reviewed inference binding](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/scripts/kaggle/reviewed_infer.py) · [All-development overlay worker](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-final-inputs-20261013-v1/worker.py.txt)

The header statement RSNA_competition_reports_only in a local binding does not negate declared gold-guided upstream processing. Fitting-only row selection protects local data access and membership, but cannot remove source-level dependence already incorporated into label-generation rules.

## Exact earlier passages requiring correction

### Experiment ledger, section4, source paragraph (line95 in the43,488-byte local version)

Earlier text:

“The source is Shio-Koube/knee3 at373f4e1daf36984fa48bf3fc8fd9883fe17cb584. It was publicly accessible/ungated at the audited revision; no dataset card/license was found. Access is not proof of redistribution rights, label provenance, teacher independence or competition eligibility.”

Correction: the missing-card observation is specific to the HF repository. Add the confirmed byte identity to Steven Kagglev6, declaredCC0, gold-guided source changes, Synovitis proxy fill and declared parents. Do not imply no license or lineage evidence exists. The cautions about full upstream rights and independence remain relevant; their factual basis must include the known provenance.

### Historical supplement, section5, heading and source-identity paragraph

Earlier heading: “Three public label sources must not be conflated.”

Clarification: Pilkwang's earlier YES/NO/UNK diagnostic source remains distinct, and JEV is not automatically the same. But **Shio's specific active blend file and Stevenv6 are the same downloaded bytes**, not merely similarly named sources.

Earlier text:

“Metadata for Steven's and JEV-Knee labels described blending/target choices tuned on58 or74 gold-like reports; those were not admitted in this diagnostic. Neither same filenames nor source-page claims establish identical ancestry or clean protected-cohort use.”

Correction: “not admitted in this diagnostic” describes only the earlier diagnostic stage. It must not be read as saying Steven's exact bytes are absent from the current trained model. Identity here is established by equal size and SHA, independently of filename. Gold-independent validation is still not established, and known gold-guided source transformations make that limitation concrete.

### Historical supplement, section7, recipe-gap interpretation

The linked October8 recipe-gap README warns that public descriptions cannot certify exact Shio origin/license. Preserve that as the historical source's statement, but add that the earlier September29 source receipt supplies a specific stronger identity/declared-license fact overlooked by that warning. The ledger's current conclusion must reconcile both rather than repeat the weaker warning as the full evidence state.

## What changes, and what does not

- **Changes:** provenance is partly established, not wholly unknown. The active supervision has known gold-guided source-level processing and a declared source license. Proposed validation must account for this ancestry rather than calling source-filtered fitting automatically clean.
- **Does not change:** measured training/public outcomes, source hashes, fixed-mask/coverage intervention definitions, negative experiment results, or the fact that neither failed ablation was fold-safe image-teacher soft blending.
- **Still unknown:** complete reproducible parent extraction/blending logic and all downstream rights/competition-eligibility implications. The archived declaration does not itself adjudicate compliance or prove a violation.
- **Required before stronger claims:** trace the declared parent transformations and exact gold influence; distinguish direct row exclusion from upstream rule/threshold selection. Do not infer that the held-out weak-development rows were used by the source without evidence, and do not call the upstream gold influence nonexistent.

No GitHub write, source-data acquisition, case-level inspection, training change or competition submission was performed for this addendum.

