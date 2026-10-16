# RSNA Knee Abnormality Detection

Reproducible report annotation, DICOM auditing, study-level training, and offline
Kaggle inference for the [RSNA competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

**Start with [current state](docs/current-state.md)** for timestamped actual scores,
training jobs, model/data recipe and what is still untested. The
[research brief](docs/research-brief.md) explains findings and limitations;
[execution handoff](docs/codex-continuation.md) contains exact continuation bindings.

At the October 16, 09:26 UTC check: reproduced **external author-trained 0.949**;
prior best **own-trained 0.872**; new own CoAtNet **epoch 4 scored 0.821**, epoch8 **0.869**, epoch12 submitted/unreported. Original fixed-16 training is running; separate source-neighbor
and positive-weight fits are queued. These are different evidence categories.

**Start with [the initiative archive](initiative/README.md) and
[the continuation guide](docs/continuation.md).** GitHub contains source reports,
metadata, blinded batch inputs, actual agent annotations/reviews, provenance,
inventories, audit evidence, decisions, and recovery instructions alongside the
code. [GodlyDonuts/RSNA-Data](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
holds the large DICOMs, caches, model weights, and mirrored versioned releases.
Credentials and temporary authenticated URLs must stay outside both repositories.

Continue from `main`; the owner authorized direct main integration and both
earlier PRs are integrated. `initiative/reviews-v2/index.json` records current
review coverage. Run `python -m rsna.archive` to verify preserved file identities
and `python -m rsna.review_progress` with the v2 paths in the continuation guide
to check the current review run.

## Historical report-supervision release

The counts below describe the original accepted release, **not the active fitting
population or latest supervision**. Current N3 uses 2,247 fitting studies and dense
public soft labels. Correspondence-filtered expansion capacities and all current
score claims are in [current state](docs/current-state.md).

The published [report-supervision release](docs/data-readiness-report.md) passed
full export verification and actual training/resume/inference acceptance. It contains
3,904 training studies, 441 weak-validation studies, 58 official evaluation studies
and four quarantined studies. All 569 corrections are preserved; the three semantic
QA gates are resolved through a separately reviewed conservative mask policy.
The latest verified software checks are recorded in the current continuation guide. Rare targets remain sparsely supervised, so data quality
and competitive benefit must be measured rather than inferred from effort.

The accepted data/code release is Git `f41879a3`. Its [private Hugging Face release](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/a177562d5f49458e287752b94f085dcb061f6495/releases/report-supervision-v1) is pinned to `a177562d5f49458e287752b94f085dcb061f6495`; all 19 direct-download objects passed full byte verification. The [publication record](initiative/provenance/report-supervision-v1-publication.json) links the separate complete-tree recovery archive and both restoration proofs.
The [expanded mission](docs/mission-and-training-plan.md) also includes a competitive
[training strategy](docs/training-strategy.md), experiments and compute plan for
UCF Newton: compare DINOv2/DINOv3 fairly, then test MRI adaptation, native detail
and a complementary 3D branch. Competitive training is underway; those original strategy proposals are historical.
Current results and research priorities are in the research brief.

The original mirror contains 4,407 training studies (58 with complete official
labels), 819,078 training DICOMs, and 557 DICOMs from the three released sample test
studies. The competition's hidden test studies are not in this mirror.

**Do not use the legacy hard labels for a large training run.** The non-gold
`rsna_manual_labels.csv` has zero positives for ACL, MCL, lateral OA, and PF OA.
Earlier `rsna_pseudo_labels*`, `rsna_custom_labels*`, `rsna_manual_labels*`, and
image-row metadata exports are historical experimental artifacts, not the corrected
release. Their calibration also used the 58 gold studies. These legacy results
must not be presented as independent evaluation of those studies.

The original report rubric counted trace effusion and insufficiently graded
ligament injuries as positive, which does not match the official severity thresholds.
Those drafts remain historical evidence. Corrected annotation must use the frozen
[`protocols/report_weak_v2.md`](protocols/report_weak_v2.md), with new outputs under
`initiative/reviews-v2/` and separate provenance for both passes. The
[verified competition contract](docs/competition-contract-20260920.md) records the
official thresholds and the limitations of report-derived supervision.

The accepted data release establishes technical operability and auditable report
supervision. It does not claim a competitive score or a diagnostic rereading of every
image. Keep original sources, masks and exact provenance with every experiment.

## Setup

Python 3.10 or newer; Python 3.10 is used in CI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1
# CPU development / CI. For GPU training install the matching CUDA build instead.
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy rsna
python -m pytest -q
```

The runtime includes `pydicom` and JPEG/JPEG2000 decoders. No report language model,
API call, pretrained-weight download, or internet access is needed during training
or inference. Report annotation is a separate preparation stage governed by the
[current provider-neutral procedure](protocols/agent_tasks.md); the originating
Devin workflow is preserved as historical evidence.

## Data contract

Targets, in exact order:

```
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

`studies.parquet` has one row per study: UID, split, report, duplicate groups, twelve
labels, per-target `__mask`, `__state`, `__evidence`, and `__provenance` columns.

| Report state | Training value | Mask |
| --- | ---: | ---: |
| positive | 1 | 1 only if accepted for supervision |
| negative (explicit absence) | 0 | 1 only if accepted for supervision |
| uncertain | placeholder 0 | 0 |
| not_mentioned | placeholder 0 | 0 |

Always multiply the loss by the mask. A stored zero with mask zero is **not a
negative diagnosis**. These categorical states are not calibrated probabilities.
Official gold values are preserved exactly, with mask one.

Report duplicates (NFKC, case-fold, whitespace normalization) and nonconstant
identical-pixel duplicates form connected study groups. A SHA-256 of each group's
smallest study UID assigns approximately 90% to `train` and 10% to
`weak_validation`. All 58 official studies go to `gold_evaluation`; non-gold
members of their duplicate groups go to `quarantine`. A deterministic whole-group
repair restores missing class coverage after masks; the accepted release moved one
study. See [the policy](docs/semantic-supervision.md). Train/test pixel duplicates
block label release. Study IDs do not establish patient-disjoint cohorts.

Weak validation measures agreement with reviewed reports, not diagnostic accuracy.
Keep gold evaluation separate from training, threshold fitting, and hyperparameter
selection; report any repeated use honestly. Per-target ROC-AUC is unavailable when
only one class is observed. Official macro ROC-AUC is reported only when all twelve
targets are evaluable.

## Preparation

Use a pinned source revision. The initial source inventory used:

```
52f995f4568e36bfa5fbbaf69a89af41226b4f5d
```

Keep the HF clone outside this repository. A pointer-only clone avoids downloading
the entire 569.8 GB source:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/GodlyDonuts/RSNA-Data "$HOME/rsna-hf-repo"
python -m rsna.inventory \
  --repo "$HOME/rsna-hf-repo" \
  --revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d \
  --metadata-root "$HOME/rsna-hf-repo" \
  --original-paths data/zip_files.txt --output data/inventory/dicom_inventory.parquet
python -m rsna.preprocess \
  --manifest data/inventory/dicom_inventory.parquet --output data/cache256 \
  --source-revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d
python -m rsna.audit \
  --inventory data/inventory/dicom_inventory.parquet --cache data/cache256 \
  --revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d --output data/audit
```

`zip_files.txt` is the original Kaggle ZIP member-name inventory, one path per line;
obtain it from the [verified source inventory](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/86df08e5be2149aef0dc855f0ebffd376e164a19/audits/source-inventory-52f995f)
or independently enumerate the source ZIP. That folder also contains the pinned
DICOM inventory, original study/series coverage counts, and artifact checksums.
It is a historical source-inventory milestone; use the corrected source revision
and accepted manifest in the readiness report for new training.
Without it, the original-archive path equality check cannot run.
The inventory records SHA-256, byte size, storage type, and UID/path identity for
every DICOM. HF downloads use LFS object batches with verified size and hashes;
the 144 directly stored Git blobs use pinned resolve URLs. The current HF dataset is private; use an authorized `HF_TOKEN` from ignored
local secret storage for authenticated downloads.
Optional direct Kaggle access uses `KAGGLE_API_TOKEN`; local unpacked
input uses `--root /path/to/competition`.

Caching checks UID consistency, rescale parameters, monochrome interpretation,
physical slice geometry, transfer syntax, pixel finiteness, and dimensions. It
resizes with preserved physical aspect ratio and center padding, scales by the
volume's minimum and maximum, and stores normalized uint16 volumes. Geometry ordering
falls back to unique InstanceNumber only when geometry is incomplete; warnings
remain in metadata. Geometry is used for ordering, not full physical-space
resampling or canonical anatomical reorientation.

The current cache version is `series-u16-v3`. Older cache pairs require rebuilding;
their prior validation cannot establish the current image-integrity checks.
Each cache has an atomic `.npz`/`.json` pair with source revision, path fingerprint,
transform version, content checksum, source slice hashes, and decoded pixel hashes.
Reruns reuse verified pairs; incompatible or corrupt pairs are rejected. Move an
orphaned pair aside and rerun that series after investigating the interruption.
`progress.jsonl` records successes and failures. Failed series are never silently
included in a release. Reserve sufficient storage; caching stops below 3 GiB free.

For the annotation stage, see [the review procedure](docs/report-review.md). After
all batches and the DICOM audit complete, use the corrected inventory and audit
from [source recovery](docs/source-recovery.md). The existing verified release
can be loaded directly; use a new output directory for any rebuild:

```bash
python -m rsna.labels --train initiative/source/train.csv \
  --reviews initiative/reviews-v2/reviewed --drafts initiative/reviews-v2/drafts \
  --review-provenance initiative/reviews-v2/provenance.json \
  --rubric protocols/report_weak_v2.md \
  --inventory ../cache256-v3-official-recovery-v1/inventory_effective.parquet \
  --source-revision 0b5e410288232f570c69ba4f44723356e6ae6ab7 \
  --dicom-audit initiative/audits/full-image-recovery-v1/audit/dicom_audit.json \
  --artifact-root . \
  --output data/release
```

The command requires exact study coverage, verbatim evidence, distinct reader/reviewer
identities, and provenance hashes binding both passes to the frozen rubric and exact
blinded reports. It also checks the full image-audit inventory, revision, cache version,
study identities, counts, and series-index checksum. A partial audit or copied draft
cannot satisfy release gates. It writes
`studies.parquet`, `study_labels.csv`, `report_annotations.parquet`,
`report_annotations_original.parquet`, `adjudication_trace.json`,
`correspondence_trace.json`, and `label_audit.json`. Keep the draft and reviewed
JSONL files for provenance.
The fixed adjacent `reviews-v2/adjudications/registry.json` is also required,
including when explicitly empty. Registered corrections must match preserved
independent decisions and exact source/review hashes; pending semantic QA blocks
release. Effective annotations preserve original states/evidence and per-target
correction traces, while gold remains authoritative. See the
[correction procedure](docs/adjudications.md).

The fixed `initiative/audits/correspondence-release-registry.json` must also
cover every preserved correspondence policy. Excluded reports receive zero loss
masks while original annotations, official gold labels and duplicate groups stay
intact. Missing, changed or omitted evidence blocks export. See the
[correspondence procedure](docs/correspondence.md).

## Train and evaluate

The reference model encodes slices with a small shared 2D CNN, then aggregates
slices and series using attention and MRI sequence metadata. It selects series by
plane before filling with additional fluid-sensitive sequences. This is an
auditable baseline, not a tuned competition-winning architecture. Do not interpret
smoke-run metrics as evidence of model quality.

Use the accepted release and the already exported
[reviewed sampling table](docs/series-preferences.md). It preserves exact-series
deduplication and reviewed plane metadata while preferring clearer or more focused
acquisitions within a limited series budget. Use the same exported CSV throughout
training, resume and heldout inference.

```bash
python -m rsna.train --studies initiative/releases/report-supervision-v1/studies.parquet \
  --series initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv --cache data/cache256 \
  --output runs/baseline --epochs 20 --device cuda \
  --side 224 --slices 16 --max-series 6 --batch-size 2 --workers 4
python -m rsna.train --studies initiative/releases/report-supervision-v1/studies.parquet \
  --series initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv --cache data/cache256 \
  --output runs/baseline --epochs 30 --device cuda \
  --side 224 --slices 16 --max-series 6 --batch-size 2 --workers 4 \
  --resume runs/baseline/checkpoint.pt
python -m rsna.infer --checkpoint runs/baseline/checkpoint.pt \
  --studies initiative/releases/report-supervision-v1/studies.parquet --split gold_evaluation \
  --series initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv --cache data/cache256 \
  --output runs/baseline/gold_predictions.csv --device cuda
```

Training uses masked BCE, AdamW, AMP on CUDA, gradient clipping, deterministic
epoch seeds, and study-level sampling. All-unknown studies are excluded from
supervised training. Checkpoints include optimizer/scaler states, seed, RNG
states, epoch/step, target order, manifest/series hashes, and preprocessing
configuration. Resume rejects changed manifests/settings or incomplete smoke
epochs. Exact epoch-boundary continuation is tested on CPU; GPU/distributed
reproducibility and throughput require measurement on the target hardware.

For multiple GPUs, run the same command with
`torchrun --standalone --nproc-per-node=4 -m rsna.train ...`.
Rank zero writes checkpoints and evaluates the full weak-validation split.
Masked target counts are reduced across ranks so missing labels do not change
the per-target weighting between single-process and distributed training.

## Offline Kaggle inference

Prepare an offline wheel bundle in the Kaggle notebook's input dataset if its
installed dependencies do not match this package. Install from that bundle with
`pip install --no-index --find-links /kaggle/input/your-wheels ...`. Include this
package and the checkpoint as notebook inputs; preflight all imports offline.

```bash
python -m rsna.infer --checkpoint /kaggle/input/your-model/checkpoint.pt \
  --series /kaggle/input/rsna-knee-abnormality-detection/test_series.csv \
  --template /kaggle/input/rsna-knee-abnormality-detection/sample_submission.csv \
  --dicom-root /kaggle/input/rsna-knee-abnormality-detection \
  --cache /kaggle/working/cache --output /kaggle/working/submission.csv \
  --source-revision kaggle-hidden-test --device cuda
```

Do not pass training reports to inference. With a template, the command predicts
every listed study and verifies exact columns, unique IDs, row order, finite
values, and probabilities in [0, 1]. A prebuilt matching cache can be supplied
without `--dicom-root`. The metrics sidecar records preprocessing and inference
timings. The released three-study sample cannot establish hidden-test performance.

The competition requires internet disabled and a nine-hour notebook runtime.
Measure the complete notebook, including preprocessing and loading, on Kaggle
hardware before submission. Local CPU timings are not a Kaggle runtime guarantee.

## Releases

See [release procedure](docs/releases.md). Pin **both** the GitHub code commit and
HF data revision in experiment records. Never replace a versioned data release in
place or promote an incomplete review/audit to the default training dataset.
