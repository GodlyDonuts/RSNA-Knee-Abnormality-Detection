# RSNA historical run expansion

Historical evidence pin: **8300cfd20c2acaeef9d684a81dbc7fe977282a62**. A separate final section uses **c83ffa8b58a3bb28dc63e7f10f4805faab0a8893**. The two snapshots are not silently merged.

This supplement fills missing scored checkpoints and failed submissions, makes older all-development training completions explicit, and reconciles the meaningful model/input/supervision families against the full audit-directory inventory. It does not discover a new best model: the best verified own solo public score remains **0.872**.

## Additional scored frozen-model checkpoints

The earlier ledger omitted the complete frozen global-CLS fold2 checkpoint curve. These five scored submissions came from **one** completed training run, Slurm **844305**, not five independently trained models. The run used source **b74060b98f01bff1b924070ccba5ce44dd002f40**, 2,281 fitting / 1,122 development studies, ten epochs and 2,860 updates. Its fixed-final internal report-label macro AUC was **0.7590158139992688**.

| Epoch | Updates | Submission ID | Actual public score |
|---:|---:|---:|---:|
| 2 | 572 | 56664013 | 0.611 |
| 4 | 1,144 | 56663969 | 0.628 |
| 6 | 1,716 | 56663606 | 0.637 |
| 8 | 2,288 | 56663742 | 0.638 |
| 10 | 2,860 | 56663180 | 0.635 |

The September29 receipt independently checked ListSubmissions and GetSubmission: all five were COMPLETE, with nonzero output and no error. No private or per-target scores were supplied. E8 being best on this observed public curve does not establish the best private-test checkpoint; the three-decimal E6–E10 range is only 0.003.

The original source and inference are shared, so this is a meaningful within-run training-progress comparison. It is not a backbone comparison against the earlier copied multi-model notebooks. The internal0.7590/public0.635 difference is descriptive across different labels/populations, not a directly measured generalization gap.

Sources: [five-checkpoint score receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-checkpoint-comparison-v1/README.md); [fold2 completed training receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/dinov3-fold2-completion-v1/README.md)

## Failed submissions that must remain visible

The original ledger named some of these failures but omitted three older submitted identities. The reviewed own-model history now explicitly contains the following **12 failed submissions**, each unscored. Zero output is not an AUC of zero.

| Submission ID | Candidate | Observed terminal outcome |
|---:|---|---|
| 56482890 | Frozen fold0 E1 | Generic hidden notebook error, zero output, no score |
| 56483451 | Frozen fold0 E2 | Same |
| 56483949 | Frozen fold0 E4 | Same |
| 56483731 | Frozen fold0 E6 | Same |
| 56485356 | Frozen fold0 E10 | Same |
| **56507997** | Frozen fold0 E8, resilient-series wrapper | Generic hidden notebook error, zero output, no score |
| **56528793** | Frozen fold2 E8, pixel-payload recovery | Generic hidden notebook error, zero output, no score |
| **56546143** | Same fold2 E8, study-fallback wrapper | Generic hidden notebook error, zero output, no score |
| 56550973 | Same fold2 E8, metadata-recovery wrapper | Generic hidden notebook error, zero output, no score |
| 56833454 | Effusion seed1 original-label control, E3 | Exceeded runtime, zero output, no score |
| 56833935 | Effusion seed2 original-label control, E3 | Exceeded runtime, zero output, no score |
| 56833831 | Effusion seed2 reviewed-label candidate, E3 | Exceeded runtime, zero output, no score |

The three bold IDs are newly explicit historical omissions. The seed1 control was later deployed successfully as **56889124, public0.680**, using the same trained weights with corrected exact-FP32 prefetch execution; this supplements, rather than erases, failed56833454. Seed1 candidate56833608 scored0.685. The two seed2 timeouts leave that paired public comparison incomplete even though both internal training/evaluation arms completed.

For the older generic errors, source fixes reproduced particular malformed-input defects but did not expose or prove the hidden traceback. Public three-study execution success was insufficient. The initially proposed sample/test roster mismatch was subsequently strongly disfavored by source evidence from successful notebooks; it must not be retold as a diagnosed cause.

Sources: [original five failures](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-hidden-rerun-failure-v1/README.md); [fold0 E8 outcome](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-epoch8-resilient-v1/README.md); [fold2 pixel recovery outcome](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-fold2-pixelfix-v1/README.md); [both later fold2 outcomes](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-series-metadata-recovery-v1/README.md); [three runtime failures](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-runtime-budget-20261005-v1/README.md)

### Count reconciliation

The main ledger's 17 scored own-solo rows plus the five frozen fold2 checkpoints above make **22 scored submission identities**. Together with these 12 failed identities, the reviewed combined ledger accounts for **34 terminal own-model submission outcomes**. This is a reconciled count of the explicit identities in these documents, not an assertion that the account's entire submission history was audited. It excludes copied third-party ensembles, private diagnostic notebooks, unsubmitted preparations and uncertain notebook pushes.

Neither checkpoint count nor submission count equals training-run count. One trained checkpoint can have several deployment attempts; one training run can produce several scored checkpoints. Do not add them to the20 principal scientific-comparison rows as if they were independent trials.

## Older all-development runs are completed fits

### Original rank32 at224

Production **865263**, source **8911f90910ab8892b8324f6985c18d3ab146ec0c**, reached the immutable **epoch10 / 4,260-update** snapshot on all **3,403 eligible studies**. Checkpoint SHA256:
**54a1415dc9326c4b2206b52e4416380dfcdeace99ace02e09ea5792494b5e093**.

The four-object epoch10 package was privately downloaded/size/SHA verified at **2026-09-30T14:27:53Z**, revision **8ddd8a87c3360b7fba3439eefa61c2f1bbb0470d**. The historical README still opens with an epoch8/epoch9 progress observation; the later immutable endpoint supersedes that progress statement. This review establishes the completed scheduled training endpoint and preservation, without inventing a separate scheduler-terminal receipt.

The exact recipe retains BF16, rank32, six series/16 slices,224 input, effective batch8 and warmup213. It starts from the official base rather than giving an already trained fold0 model another ten epochs. Its later public result is **56711823=0.792**.

Sources: [population and recipe](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/dinov3-adapter-a32-all-development-v1/README.md); [production config](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/dinov3-adapter-a32-all-development-v1/production-config.json); [epoch10 snapshot](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/dinov3-adapter-a32-all-development-v1/epoch-publication/epoch-010/snapshot.json); [epoch10 verification](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/dinov3-adapter-a32-all-development-v1/epoch-publication/epoch-010/verified.json)

### Original-label native384

Production **868093** completed **0:0**, ten epochs / **4,260 contiguous finite updates**, with all ten immutable epochs privately verified. The fixed-final checkpoint SHA256 is:
**8f6da3a1b401d9056fe846eb85e52fdb596fbadcc39ce57e1bc7b8cfe553a3a7**.

It preserves C9 science source **c9fcc7eefb09103942be89bf4fa87410eb9ec5ab**, the accepted original masks and 3,403-study membership. The historical admission README's “not dispatched” text is superseded by the completed epoch-preservation receipt. Its scored E5/E10 outputs are **0.804/0.802** under IDs56783855/56833260.

Both all-development runs intentionally have **zero internal holdout studies**, no training-set AUC and no outer-prediction endpoint. Their lack of internal AUC is not an evaluation failure. They exclude the 501 fully masked original training rows, 441 reserved confirmation, 58 official evaluation and four quarantine studies.

Sources: [native384 admission contract](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/native384-all-development-v1/README.md); [actual completed ten-epoch receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/native384-all-development-epochs-v1/README.md)

## Previously omitted or underdeveloped experiment families

### Physical slice-distance auxiliary supervision

This is a distinct auxiliary-loss hypothesis, not the already tested neighboring-image-channel input.

- Geometry survey **866739 completed0:0** on2,247 fitting studies and 11,639 selected series. All selected series had usable increasing physical geometry. The uniform 16-slice grid supplied 1,392,640 possible within-series pairs before the capped sampling policy
- Original full224 BF16 gradient probe **866846 failed1:0** because the launcher omitted the existing Python.h/CPATH prerequisite. No completed gradient measurement or optimizer update was produced
- Distinct corrected H100 probe **867310 completed0:0 in12:34**. Across two fitting-only effective batches, auxiliary weight 0.05 contributed **2.0466–2.0707%** of disease encoder-gradient norm. Objective cosine was mildly negative, **−0.111 to−0.072**; combined encoder norm 0.2720–0.2813 did not trigger encoder-only clipping. Physical/shuffled/zero classification gradients, logits and pair rosters agreed
- CPU **866929 completed0:0** for the separate 16-fitting + 16-development engineering fixture. The real production-trainer full/stop/resume orchestrator was prepared, but **no completed full clinical comparison is established by these receipts**

This was successful numerical/scale investigation following a failed compiler prerequisite. It is not a failed accuracy experiment, demonstrated anatomy learning or proof that task conflict is clinically important. Physical and shuffled objective scales were nearly equal at initialization; later learning and complete-model clipping remained untested.

Sources: [geometry survey](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/slice-relations-geometry-survey-v1/README.md); [original failure and corrected identity](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/slice-relations-gradient-header-recovery-v1/README.md); [actual corrected terminal](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/slice-relations-gradient-header-recovery-v1/terminal/slurm-terminal.txt); [aggregate gradient-scale decision](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/slice-relations-gradient-header-recovery-v1/training-only-scale-decision.json); [CPU fixture and remaining recovery gate](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/slice-relations-trainer-inputs32-v1/README.md)

### Whole-series neighboring-feature context

The completed whole-series result 0.8603399208 is the **no-context** coverage control. The same family separately implemented a **66,433-parameter** zero-initialized neighboring-feature message module, using physical gaps when known and ordinal offsets otherwise. Its proposed production contrast remained prepared only. Therefore the no-gain coverage result does not reject explicit sequence interaction.

Source: [whole-series context protocol and operational update](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/experiments/competitive-v1/whole-series-context-v1/README.md)

### Anatomical task sharing

A fitting-only support audit found observed positive/negative/unknown counts:
- ACL-or-MCL:176 /1,505 /566
- Medial-or-lateral meniscus:996 /879 /372
- Contusion-or-fracture:124 /372 /1,751

These logical union tasks create no new MCL or fracture labels. The audit also counted 117 MCL-negative/ACL-positive, 533 lateral-meniscus-negative/medial-meniscus-positive and 12 fracture-negative/contusion-positive observations, supporting disease-specific discrimination rather than copying sibling predictions. Shared anatomical queries with disease residuals were **proposed, not integrated or trained**.

Source: [anatomical-sharing support](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/anatomical-sharing-support-v1/README.md)

### Patch geometry and artificial padding

CPU876287 audited geometry metadata for all 2,247 fitting studies  / 12,476 cached series. Mean wholly artificial letterbox patch fraction was **0.9248%**; 607 series had any wholly artificial patch cells. This is negative diagnostic evidence against artificial letterboxing being the principal score-gap explanation. It does not prove all inputs are well localized or establish cross-series registration. The plane-memory prototype in this earlier receipt was later trained under its separate already-recorded family.

Source: [patch geometry and prototype status](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/patch-geometry-research-v1/README.md)

### Original hard witness and smooth successor

The original hard selector's tied zero-score initialization gave nonzero gradient to **576/663,552 score cells, 0.0868056%**, while the global CLS path retained gradients. Subsequent engineering summaries report original hard/BF16 recovery, but no completed production accuracy contrast was identified for that exact hard selector. The separate smooth/FP16 successor did reach the completed three-epoch comparison already in the ledger. Do not count its full-fit-v1 preparation and full-fit-v2 result as two scientific experiments.

Sources: [measured initialization mechanism](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/witness-initial-gradient-v1/README.md); [hard-selector protocol](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/experiments/competitive-v1/patch-witness-v1/README.md); [smooth comparison preparation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/smooth-witness-full-fit-v1/README.md)

### Twenty-model shared-prefix teacher execution

The archive establishes more than a single-model synthetic fingerprint:
- Twenty original public DINOv2 members shared 113 tensors / 11,404,416 elements in their common frozen prefix
- On **eight fitting studies × twenty models × ten windows × twelve targets**, shared-prefix execution matched original full-model fingerprints, window logits and probabilities **bit for bit** on the H100
- The separate single-teacher CPU/GPU comparison had maximum cross-precision difference 0.000780702; bitwise CPU/GPU equality was not claimed

This is successful execution-equivalence engineering. It does not establish teacher training-cohort independence, a full 2,247-study collection terminal, teacher accuracy, student training or distillation benefit. The existing full-collection observation 1900/2247 remains incomplete evidence.

Sources: [common prefix identity](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-teacher-committee-prefix-20261013-v1/README.md); [actual shared-prefix acceptance](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-teacher-shared-prefix-GPU-20261013-v1/README.md); [single-teacher GPU parity](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-teacher-GPU-acceptance-20261013-v1/README.md)

### Earlier backbones and legacy claims

The accepted early ConvNeXt-Tiny receipt used complete official weights but only a deterministic synthetic grayscale ramp on CPU. It produced finite 1×12 logits and finite gradients for 18 head tensors while the encoder stayed frozen. DINOv2's contemporary receipt verified 224 tensor keys/shapes from the header only; it explicitly did **not** download/hash the whole weights or execute a pretrained forward.

The legacy unifiedtheory code describes physical ALiBi, cached-token adapters, report distillation, raw-DICOM specialists and ensemble plans. Its enclosing archive explicitly says upstream commit and claimed benchmark/result provenance were not verified. Those descriptions are preserved hypotheses/legacy claims, not admissible completed current RSNA runs. The later source-faithful DINOv3 slot result is a different experiment.

Sources: [ConvNeXt actual scope](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/competitive-pretrained-local-check-v1/convnext-actual-cpu-receipt.json); [DINOv2 header-only scope](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/competitive-pretrained-local-check-v1/dinov2-header-compatibility.json); [legacy provenance warning](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/legacy/README.md); [legacy protocol descriptions](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/legacy/unifiedtheory/README.md)

### Training throughput proposal

The training profiler is implemented and had 64 corrected root tests, but no actual isolated MRI/H100 setting was promoted. Its native control timing of 70,589.364seconds to 2,810 updates is an observed existing-run motivation. Proposed worker, microbatch and encoder-chunk changes still need actual same-setting recovery and matched-arm acceptance. Exact-FP32 inference/prefetch optimization is separately implemented; it is not an untried training-throughput result.

Source: [profiler status and numerical limits](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/experiments/competitive-v1/throughput-profile-v1/README.md)

## Coverage and remaining uncertainty

The complete historical inventory has **1,090 root entries**, including 1,058 directories and 32 files. This expansion assigns every entry a name-based navigation category:
- 230 entries in 43 model/input/evidence groups
- 313 deployment or preservation entries
- 484 annotation/data-governance entries
- 63 general engineering/planning entries

These are folder/file counts under a new explicit taxonomy, not numbers of experiments or individually read artifacts. All 22 competitive-v1 directories are mapped, including the folds metadata directory. The separate family-coverage document records every group, representative evidence and the finite remaining checks.

The registered historical source review covers **76 aggregate source files**. Long docs were reviewed for relevant history/result sections; this does not claim every sentence or nested artifact was inspected. Two separately pinned newer READMEs are listed below. No raw clinical case rows, reports, images, protected validation records or individual prediction files were read. No dataset/weight downloads, training, repository writes, publication or competition submissions were performed.

Remaining terminal gaps are concrete: public-repeat candidate evaluation; dense canceled-state/promoted checkpoint and whether its matched suffix fit started; patch-focus GPU terminal; full teacher collector completion; all-data public epoch10 terminal; and primary aggregate evidence for legacy claims. Engineering-only preparation and preservation duplicates can stay summarized unless an exact event-level job census is requested. Later progress needs its own pin.

Known provenance correction remains in force: the active Shio CSV is byte-identical to Steven Kaggle v6, a report-LLM blend with declaredCC0 and Gold58-guided source design. No MRI-teacher ancestor is documented. The newer e4f288d6 focal-window module has 127 engineering tests and no accuracy result; it does not retrospectively change these historical outcomes.

## Separately pinned October14 update

Newer source pin: **c83ffa8b58a3bb28dc63e7f10f4805faab0a8893**.

1. Original all-development native-public run908454 completed **epoch9 / 3,834 training updates**. Checkpoint **f00817598c47267d64356ddbb523a72ee8764fc1b6274ca266df19c130af33a4**, 125,556,557bytes, was privately preserved once; all five objects were downloaded/size/SHA verified at HF ba1ae09a3ad2a802b17e99e30c8ffd383c6eef4a. This advances the historical pin's eight-epoch observation. It establishes neither epoch10 completion nor an E9 score
2. E8 private Dataset **12589606/version1** passed complete68-object download/size/SHA verification. Its notebook push had one exclusive intent but actual push result/public acceptance remained unobserved. **No competition submission occurred**. Dataset readiness is not a score

Sources: [epoch9 preservation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/initiative/audits/final-public-source-epoch9-20261014-v1/README.md); [epoch8 prevalidation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/initiative/audits/kaggle-epoch8-prevalidation-20261014-v1/README.md)

The best verified own solo public score remains **0.872**. No focal-window, patch-focus, dense-capacity or teacher-distillation gain follows from this update.

