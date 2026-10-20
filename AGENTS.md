# RSNA Knee agent guide

## October20 09:30UTC live source/context pipeline

Actual978232 finished all16 source caches:2,391,033,794 original bytes,
1,091,912,852cachebytes/62,032regions, exact both masks/targets/priors, all original
labels/masks unchanged. These16 caches were transferred once, fully SHA-verified
byCPU978296/978304, and GPU978305 is actually training B/C on27 (24updates/arm,
12train/4disjoint fitting engineering, full/stop/resume). No clinical result yet.
Do not duplicate it. Both-head LR reduction is a training-only engineering choice.
An observed fractional-warmup overshoot is clamped only in FUTURE local train.py;
existing staged GPU/source/cache programs remain byte-frozen. Twelve scoped CPU
checks pass, including complete recovery and fractional-LR bounds.

Actual CPU978371 measured18.656x faster CPU input loading from EXACT FP32 cached
masked contexts (58.611s load+render versus3.142s cached load over16 cases). All
images, targets, metadata, support priors and RNG matched bitwise;263,296,123bytes.
This is input-speed evidence, NOT training-speed or accuracy. New context_cache.py
has no encoder features, teacher logits, labels or lossy casting.

Original full raw CPU978318 RUNNING on33: two concurrent source studies/4CPU/48GiB,
all3887 fitting studies/720851 original DICOMs. It reuses16 verified raw caches;
320GiB TOTAL includes paid originals,4GiB own scratch/256MiB controls plus separate
24GiB future model/publication reserve; all old liabilities/20GiB reserve retained.
Original full context consumer978414 RUNNING on33/oneCPU/12GiB, pipelines exact
24-window FP32 inputs as raw proofs complete, reuses16 context caches.96GiB TOTAL
includes263MB paid contexts and replaces completed unused2GiB pilot reservation;
32MiB controls. No GPU waits for data and no full predictive fit is dispatched.

Original978318 has12hour wall limit (ends21:07UTC). One attempt to extend it was
rejected by scheduler permissions; no change/restart/pause occurred. Read
predictive-full-cache-observation-20261020-v1. At actual terminal, collect completed
registry if0:0; ifTIMEOUT reconcile completed raw/context proof prefixes and at
most two private source buffers before continuing only missing work. Never repeat
verified case downloads/cache writes. First use actual throughput, not linear
extrapolation alone. New source/context stages are independent of healthy fits.

Source/private GPU milestone de165575 is already67-object/352183byte privately
full-download/size/SHA verified atHF b6c14f013ea200fa573bad8ff25e4adb5fe24793,
CPU978283. Never repeat. Newer full-cache/context/warmup changes are not covered
until a distinct new receipt. Original975297 and finite successor46627/46628 stay
active; source-zero E3 kernel139786921 now actuallyRUNNING, no score yet.


## October20 actual learning milestone and live next work

Original real-MRI B/C GPU978218 COMPLETED0:0/79seconds on healthy ordinary evc27.
Both arms passed complete model/AdamW/history/cursor/all-RNG and disjoint-process
prediction-byte recovery; all full encoder parameters trained after warmup.
Full-phase4-update times B2.213s/C1.930s, peak~5.414GB. This one-case engineering
pass is NOT a clinical/competition gain. C training exam loss briefly reached
21.46; preserve that negative numerical evidence. New bounded16-case learning
uses BOTH arms headLR3e-5 (tenfold lower),24context windows/mask, complete source
region/support priors,12fitting training/4disjoint engineering cases,8epochs/
24updates/effective4/head-only3. No protected/gold cohort or teacher loss. It is
prepared under predictive-raw16-learning-20261020-v1, finite owner47201; inspect
actual-inputs/submit/terminal before inferring execution. It admits inputs first,
reserves no GPU while waiting, and never repeats uncertain transfers/jobs.

CPU978232 is actually building16 complete source-level lossless fitting caches
on33. At latest observation10/16 complete:1,370,769,026 original bytes ->
633,162,775 cache bytes,37,104complete descriptor regions. Source fixture reused,
other originals acquired once with pinned individual SHA; no full-cohort cache
or full predictive production yet. New16GiBcache/2GiBowned transient/16MiBcontrols
funded after all old liabilities and20GiBreserve. Only consumed job-owned scratch
may be reclaimed; originals/cache/proofs retained. Read original raw16 controls.

Interrupted978084 retained2.392GB; CPU978186/978187 completed exact suffix-only
recovery to27,55,685files/12.434GB total,10.042GB transferred, no GPU held during
copy. Do not repeat any replica or978218. Node47 remains CUDA-unusable;27 passed.

Source-zero E3 kernel139786921v1 was pushed ONCE after fresh10739.90free VM seconds,
6000fit cap/3000final-serving reserve; original E2 fully6-object privately verified
atHF ee73db671130b6f850f551844e0a16a9ac80fcec,972updates. Distinct finite successor
46627 handles only remaining E3 preservation/qualification/one entry/outcome;
46628 handles only remaining teacher_rank epochs and CPU final compilation.
Original ended owners/intents remain preserved. Do not restart them or duplicate
any phase. Original H100975297 remains healthy. Best actual0.958external ensemble,
own adaptation0.949; no new original accuracy gain claimed. Git milestones after
1172d429 are not covered by prior private HF mirror until a new receipt exists.


## October20 08:55UTC actual recovery handoff

Portable978084 on evc27 passed actual allocated CUDA context/memory, then FAILED
12:03 before model/optimizer because Slurm stdout runtime copy timed out. CPU978163
proved retained26319files/2392247988bytes, no model-start or replica acceptance.
Preserve partials. New predictive-private-resume-20261020-v1 uses CPU-only private
socket endpoints, exact SHA/EOF and existing prefix reuse; no GPU during copy and
no overwrite/reacquisition. Inspect actual-resumed.json/intent before action.
CPU978128 actually accepted lossless original67DICOM cache:14159093bytes,
1072regions, bit-equal source pixels and both contexts/targets/supports/priors.
This is one-case engineering, not cohort admission or accuracy.
Source-zero E2 GPU139756869 completed972updates; original publisher977754 completed
0:0/91seconds and verified all6objects privately atHF
 ee73db671130b6f850f551844e0a16a9ac80fcec. Do not repeat publication. Original local
owners32102/27713/33801/38630 ended after transport loss; reconcile receipts and
use distinct successors for only remaining phases, never restart old controls.


Updated October20,2026,06:23UTC. Goal active: strongest legitimate competition model. Read [current state](docs/current-state.md), [continuation](docs/codex-continuation.md), then the referenced original receipts. Earlier snapshots remain in Git history; live service/job state governs action.

## Current priorities

- **Device update supersedes earlier pending977678:** original cancelled PENDING/zero/never-started after actual node47 replica;977901 FAILED before model/optimizer. Actual977943/977963 prove no usable CUDA context on47 (one device missing, remaining CUDA999 even with both allocated/assigned steps). Exclude47/38; no reset or isolation bypass. Use only the single new portable original diagnostic coordinator if not already dispatched; actual-submit.json governs. Original684b02a7 numerical loop unchanged; eleven scoped CPU tests cover new separate lossless raw-cache source, not an actual cohort cache or GPU pass. See current-state and predictive-portable-recovery-20261020-v1.

- Predictive source1172d429/actual input+dispatch sources privately fully45object download/size/SHA verified atHFb5554649716ec81e4b72f8a98a7fcdcb41c0cb80 byCPU977741; do not repeat. Original metadata-only977688 cancelled PENDING/0seconds/never started, not a model run. MRI-only readout has explicit precision controls; actualT4 still unaccepted.

- **Latest actual predictive dispatch:** CPU977675 accepted original raw MRI/fitting labels/full MRI encoder. GPU977678 submitted ONCE, PENDING Priority evc33/1H100/2CPU/20min; sourceafca3ce7 and root `/tmp/rsna-predictive-real-trainer-20261020-v1`. Inspect actual result; never repeat v1/v2 coordinators. See `predictive-real-trainer-20261020-v2`. Original CPU977645 label-subset join failure was corrected without changing labels/numerical modules; no GPU started in that attempt. No actual GPU pass or predictive accuracy yet. Queued numerical sources frozen.

- **Owner-directed primary NEXT branch:** predictive-evidence B/C, not further cached-attention production. Read the new current-state opening and `initiative/research/predictive-evidence-proposal-20261020-v1/README.md`. Actual raw-input CPU977456 passed67 original DICOMs/1072 regions with target intervention/context-byte equality. New full-encoder B/C/C0 modules and distinct optimizer/recovery loop pass10 CPU synthetic checks; no actual MRI model/GPU/accuracy acceptance yet. Next real-MRI parent/raw-input tiny trainer/recovery and earlyT4 throughput before a funded matched full fit. No old teacher loss/parent-logit shortcut; complete target priors; B receives actual observed descriptors. Existing healthy jobs untouched. Deadline actualOct22,23:59UTC; latest final-launch targetOct22,12UTC.
- Pointwise3 now COMPLETE1458updates/private six-object fully verified HFe36bb88ffa2aa4ac62680a29a5d7ba0fdd7e7bf6/CPU977516, modelda61c534; compiledCPU977528, no actualT4/entry. Do not repeat P1–3/compile. Owners27713/38630 continue. Historical pointwise2/older next-actions below are superseded by this receipt.

- Changed-target E1/E2/E3 COMPLETE486/972/1458updates and all privately fully verified. E3 modelb9686536 atHF25c8cada7265f7c33f9551d1d0bc6511517bebc5, CPU976299, sixobjects. Do not repeat any push/publication. Compiler25232 finished; final qualification/public run COMPLETE and entry57422143 accepted once. Original entry57422143 now COMPLETE/scored0.949, no error. Evaluator21960 ended normally after both predeclared outcomes; never restart.
- Actual code entry57418131 COMPLETE/scored0.948, below older own0.949. E3 scored0.949, tying the existing adaptation; no teacher-supervision gain established. Corrected qualification139716471/public139717947 both COMPLETE.
- Matched full pointwise/ranking GPU975297 is RUNNING on cached evc48, completed pointwise epoch2/972 finite optimizer updates; old975129 cancelled while zero-runtime pending after actual admission. Never duplicate. Both full numerical sources frozen.
- Actual975724 selected-reader admission passed,2.82x full-case CPU speed and exact inputs/RNG; not used in healthy975297. Epoch preservation owner27713 privately verified pointwise epochs1/2 once; epoch2 atHF755515ba5576fcadff371273d056370338ac6d3e viaCPU976879, sixobjects. Never repeat either. Four later checkpoints remain.
- Source-zero epoch1 kernel139742882v1 COMPLETE486updates; all six immutable checkpoint/provenance objects privately fully verified atHF1df252b385869b4ec8372526f39ad169f33e2e6e byCPU977111. Never repeat epoch1. Original32102 admitted fresh15,623.57freeVMseconds/capacity and pushed epoch2 kernel139756869v1 ONCE, slug godlydonuts/rsna-zero-full-e2-e98af321d8b0/sourcee98af321. ActualRUNNING, target972of1458updates; all3887/two views/source weight0. Final serving reserves3000seconds; futureE3 still needs fresh admission. Preserve originals/owner, no competing dispatch. Original139736683 actual two-T4 recovery and139738563 reader acceptance remain authoritative; no diagnostic repetition or revival of cancelled975954.

- New final-only source-zero serving account controls staged once byCPU976721/source93e19967; sole finite evaluation owner33801 waits for actual final fitting/private verification/pointer before CPU compile, bothT4 qualification, public inference and at mostONEcodeentry. No source-zero serving/score exists yet; never restart owner.
- Private integrity Dataset12722964v1 is fully download/hash verified once. CPU-only Kaggle reader probe139738563v1 actually COMPLETE/accepted:32studies/64views/exact tensors/targets/RNG,3.244x case-loading speed. No GPU training-speed or accuracy gain claimed. Never repeat the upload or push.
- Selected new full-source-zero/final-serving sources and actual0.949 receipt are privately fully46object download/size/SHA verified once atHF381ae720b6498227c4991f20d130d9f634c175e0, CPU976734/source56747336. Never repeat mirror.
- New cached-attention ratio CPU976806 rejected on2invalid denominators atlargest synthetic drift. Distinct linearized correction CPU976828 passed all32fitting-feature/12target/fourlevel proxy gates, MSE ratios.001779/.027942/.162722, no invalid outputs. These are synthetic-drift approximation results, not actual student drift or accuracy. Source6dec668b; next separately funded actual-encoder diagnostic/matched deterministic modes, no GPU production yet.
- Cached-attention mechanism sources/results and UI optimizer progress privately20object byteverified atHFaa111c78483a5f0b79ff9b2d56be187fdd470294, CPU976868/source4eb1a39a. Never repeat prefix. Actual encoder/readout and privately aligned full3887 cache are now implemented/admitted; GPU977100 recovery is queued. No production fit or accuracy result yet.
- Best Kaggle0.958 external reproduction; own adaptations0.949, no original improvement established. Keep actual results and negative evidence current.

- New cached-attention source9e76db5d/full3887 private reference cache accepted; actual CPU977068 admitted real-encoder recovery. GPU977100 submitted ONCE on evc48/oneH100/twoCPU/30min, lastPENDING Priority. Never repeat dispatch. Root /tmp/rsna-cached-attention-encoder-20261020-v1; inspect original logs/receipts. Reference/linear17train/15disjoint fitting predictions/sixupdates require actual complete recovery; no GPU acceptance, production fit or accuracy gain yet. See cached-attention-encoder-20261020-v1. Original975297/27713/32102/33801 verified live unchanged.

- New encoder sourcea08571df and23 sanitized small objects privately fully re-download/size/SHA verified atHFe94663ca412ba58c9ceccbb1bb1dbffe33ea2b9f viaCPU977113, canonical lock; never repeat. No private reference arrays/UIDs/weights included. Source-zero first-epoch publication977111 COMPLETED/privately verified; original32102 now follows actualRUNNING epoch2 kernel139756869. Do not duplicate.

- Actual v2 paired-final compiler stagedCPU977228, sole finite owner38630/source ddf1b7e4/SHA b6c60b2ef513394afb229131783810abaddac75a5cb687517ad0b4ac14c40444 verified live. Waits for each role private final3/1458, max2CPU compilations/30h, zero GPU/Dataset/entries. Root original run48/final-serving-v2; no restart. V1CPU977213 refused beforewrite; RO977215 proved absence and237paid teacher files. V2 budgets19remaining of256, allreserves retained. New H100 models still need private Dataset bridge and actualT4 qualification before scoring; do not invent Kaggle training kernel. See teacher-pair-final-serving-20261020-v2.

## Standing execution rules

Accepted labels/masks/folds/base/order stay unchanged; protected gold/confirmation unused. Public teachers are not independent OOF evidence. Daenerys outputs are review candidates, never automatic labels. Stanford-derived weights private/noncompetition; no newly admitted KneeCoT. Failed MedGemma protocols remain negative evidence.

Inspect actual jobs/source/output bindings, quota, sidecars and private journals before mutations. Never duplicate transfers, jobs, publications or submissions. Existing intent/uncertainty requires read-only reconciliation; observation loss is not terminal evidence. Physical4/8GPU nodes prohibited even requesting one; ordinary1/2GPU nodes allowed. Minimum useful CPUs. Retain20GiB shared/node reserve and5000shared files plus funded metadata/liabilities. No paid credits, deletion credit or double counting paid bytes.

Each actual immutable production epoch requires exact provenance and one private independent full download/size/SHA verification under the canonical lock. No raw clinical data, IDs, private arrays/signatures, restricted source bodies or weights in Git/public logs. Credentials only ignored0600.env/RAM/stdin/ASKPASS, never argv/logs.

Kaggle uses the canonical Newton ledger/single account claim and fresh authoritative allowance/max5daily. Require real source/runtime/T4/input acceptance, offline MRI-only inference and actualsubmission.csv. Never submit engineering diagnostics, public three-row CSVs, failed/uncertain versions or source-derived Stanford weights. Qualification/public execution does not prove a hidden score.

Keep live controls frozen; see continuation. No Devin or new agents. Relevant scoped checks for operational changes; full numerical checks for numerical changes. Do not use test counts as progress toward accuracy. Report meaningful evidence/results/failure, remain quiet on unchanged state.
