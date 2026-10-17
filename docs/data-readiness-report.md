> **Historical contract or plan.** This document preserves its original dated
> release or experiment. For current scores, fitting populations and active work,
> read [current state](current-state.md), [research brief](research-brief.md) and
> [research-agent handoff](research-agent-handoff.md). Old “next” instructions
> below are not live job or submission authority.

# Accepted report-supervision release

The published `report-supervision-v1` release passed its data and runtime acceptance
on September 22, 2026. It is ready for supervised training with its supplied masks.
The accepted data/code release is Git `f41879a3`. Its [private Hugging Face release](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/a177562d5f49458e287752b94f085dcb061f6495/releases/report-supervision-v1) is pinned to `a177562d5f49458e287752b94f085dcb061f6495`; all 19 direct-download objects passed full byte verification. The [publication record](../initiative/provenance/report-supervision-v1-publication.json) links the separate complete-tree recovery archive and both restoration proofs.

This is an auditable weak-supervision release. It does not certify perfect clinical
labels or better data than competitors. The [training strategy](training-strategy.md)
will test whether its conservative choices improve generalization.

## What to load

- [Final study manifest](../initiative/releases/report-supervision-v1/studies.parquet)
  and its [verification receipt](../initiative/releases/report-supervision-v1/verified_release.json).
- [Reviewed series selection](../initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv).
- Immutable `series-u16-v3` caches for source revision
  `0b5e410288232f570c69ba4f44723356e6ae6ab7`, described in
  [source recovery](source-recovery.md) and [series preferences](series-preferences.md).
- The complete repository supplies the current policy, source, review and code
  dependencies. Training verifies them before selecting any rows. Legacy label
  exports are not alternatives to this release.

The manifest SHA-256 is
`2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7`;
the receipt SHA-256 is
`6453a8ff31373c40556dc857e15fa0025f122195388a2b46c81ecd9cff0b8833`.
The release contains the annotations before and after adjudication, all correction
and exclusion traces, the coverage audit, and the deterministic split repair.

## Measured coverage

| Split | Studies | Leakage groups | Studies with observed targets | Observed target cells |
| --- | ---: | ---: | ---: | ---: |
| Training | 3,904 | 3,761 | 3,403 | 21,044 |
| Weak validation | 441 | 436 | 394 | 2,475 |
| Official evaluation | 58 | 58 | 58 | 696 |
| Quarantine | 4 | 1 | 0 | 0 |

All 4,407 reports have original annotator, independent reviewer and source-first
preread provenance. The release preserves all 569 accepted corrections and three
retirements. Both priority bone passes cover 1,549 reports; priority cartilage
adjudication covers 260 reports. These are report-reading counts, not image diagnoses.

The accepted semantic ledger accounts for all 52,884 cells, 23,804 overlapping
review obligations and 4,412 hold records. It retains 23,844 definite report
observations before official-label replacement, quarantine and other filtering;
3,629 definite observations are held out. Original opinions and disagreements
remain intact. All three global semantic QA gates are resolved against this exact
policy, not by declaring the held findings clinically correct.

Every target has both classes in training and weak validation. This is a minimum
computability condition, not evidence of adequate statistical power:

| Target | Training positive / negative | Weak-validation positive / negative |
| --- | ---: | ---: |
| ACL | 256 / 2,561 | 30 / 293 |
| MCL | 7 / 2,762 | 1 / 321 |
| Medial Meniscus | 1,244 / 1,662 | 163 / 183 |
| Lateral Meniscus | 446 / 2,386 | 54 / 279 |
| Medial OA | 69 / 1,410 | 8 / 167 |
| Lateral OA | 27 / 1,507 | 2 / 193 |
| PF OA | 57 / 1,249 | 8 / 142 |
| Effusion | 25 / 1,928 | 3 / 237 |
| Synovitis | 364 / 15 | 29 / 3 |
| Baker's | 9 / 1,132 | 1 / 140 |
| Contusion | 182 / 1,042 | 19 / 117 |
| Fracture | 5 / 699 | 1 / 81 |

## Isolation and correspondence

The [persisted-table check](../initiative/audits/final-persisted-release-invariants-v1/validation.json)
confirms exact official labels, no training/validation leakage-group overlap, and
quarantine of non-gold studies connected to official evaluation. Quarantine shares
the related gold group deliberately; neither enters training. One whole group
containing one study moved from training to weak validation to restore missing
MCL class coverage. Gold values did not select that move.

All 18 registered report exclusions have twelve zero masks. The 24 unresolved
correspondence cases comprise 23 non-gold studies with twelve zero masks and one
official evaluation study (032:35), whose official labels and evaluation masks
remain intact. None supplies report-based training loss. Earlier statements that
all 24 exported masks must be zero were imprecise about this official-gold case.
Original source text and target states/evidence for all 24 were rechecked exactly.

## Images and runtime evidence

The full technical audit covers 819,635 slices in 24,386 series, including the
released sample test images. The corrected audit has zero warnings. Its complete
uint16 NPZ cache totals 82,555,132,978 bytes; metadata and temporary working space
are additional. Two invalid mirror encodings were recovered from official files.
Original source and recovery provenance remain preserved. Targeted visual review
covered the escalated cases; there was no diagnostic reread of every MRI.

[Actual final-input acceptance](../initiative/audits/final-input-training-acceptance-v1/acceptance.json)
ran on the local Apple M1 CPU, using Python 3.11.0 and PyTorch 2.14.0. A deliberately
small diagnostic model used 32 training and 32 weak-validation studies, selected
only after complete release verification, with two slices, up to three series,
32-pixel inputs and width 8. In 235.7 seconds across five commands it demonstrated:

- Two complete epochs and 16 optimizer steps, with weights changed from initialization.
- Exact equality of the complete checkpoint after uninterrupted versus resumed
  training, including optimizer and random state.
- Zero gradients for every masked target in the full training manifest.
- Valid finite probabilities for all 58 official evaluation studies and the exact
  three-study sample-submission template.

The small run's scores are not competitive performance estimates. Software checks
passed 573 tests, lint, formatting and type checking before this acceptance.
The upload workflow also requires successful CI for the exact published commit.

## Limits that remain relevant to training

Report mention patterns are selective. Masking uncertain findings improves auditability
but reduces supervision and may introduce selection bias. Some targets have too few
observations for stable validation. The 58 official studies were exposed in historical
legacy calibration and must not be described as globally untouched evaluation.
The corrected run keeps them outside training and model selection.

Study/group isolation is verified; patient identity is not established solely by
anonymized study identifiers. Language, report style and scanner groupings are proxies,
not confirmed institution identities. The latest authenticated original-source
checks and the separately preserved mirror-recovery checks have distinct scopes;
no fresh official-byte confirmation of all 576 mirrored source files is claimed
after the earlier rate-limit interruption.

No GPU throughput benchmark, complete hidden-test runtime rehearsal or competitive
training has yet been performed. Those are first experiments in the modeling plan,
not reasons to misrepresent this diagnostic run as a trained competition model.

## Portable cache rebuild and first training command

After restoring the pinned Git tree and installing its dependencies, use a separate
scratch volume with sufficient free space. The source dataset is private; load the
authorized HF credential from ignored local secret storage. These commands do not
need the originating workspace paths:

```bash
mkdir -p /scratch/rsna
python -c "import gzip; from pathlib import Path; Path('/scratch/rsna/inventory.parquet').write_bytes(gzip.decompress(Path('initiative/audits/final-release-technical-inputs-v1/inventory_effective.parquet.gz').read_bytes()))"
python -m rsna.preprocess --manifest /scratch/rsna/inventory.parquet \
  --output /scratch/rsna/cache256 --side 256 --source hf \
  --source-revision 0b5e410288232f570c69ba4f44723356e6ae6ab7 \
  --min-free-gib 20
python -m rsna.train \
  --studies initiative/releases/report-supervision-v1/studies.parquet \
  --series initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv \
  --cache /scratch/rsna/cache256 --artifact-root . \
  --output /scratch/rsna/diagnostic --epochs 2 --device cpu \
  --width 8 --side 32 --slices 2 --max-series 3 --batch-size 4 \
  --workers 0 --seed 42 --diagnostic-study-limit 32
```

Use the existing verified cache directly when available. The rebuild command is
provided for portability; it was not repeated after the completed full-image audit.
The training command is the accepted diagnostic configuration, not the proposed
competitive pretrained architecture. Never change a cache file shared by hardlinks.

## Immutable publication and cluster handoff

Use code/data Git `f41879a33b38e47f0cbd96025edac7e2176a0a2f` and the direct release at HF revision
`a177562d5f49458e287752b94f085dcb061f6495`. `PUBLICATION.json` in that folder binds the unchanged 11 release
files and six companions to the accepted Git tree. The extra README and publication
index make 19 fully downloaded and verified objects. This folder is a convenient
data export; runnable training still requires the full repository and its relative
policy/dependency paths. Default legacy dataset configurations are not this release.

The complete recovery checkpoint is at HF revision `c0b2c8b041120ae6d27891145f1a19e48a6d498f` under
`preparation-checkpoints/f41879a33b38e47f0cbd96025edac7e2176a0a2f/`. Its 444,510,430-byte delta
needs the pinned base archive listed in its manifest. Distinct peer and root
executions reconstructed all 11,145 Git files identically. Remote archive LFS
objects were verified by content-addressed SHA256 and size; ordinary objects were
streamed. The separate direct release above was fully streamed, including LFS.

The current documentation/receipt commit follows the immutable release commit;
it records publication results without claiming the release contains its own
future receipt. Training code, accepted manifests and the reviewed Newton strategy
remain unchanged. The full cache arrays are not duplicated in the compact export;
use the pinned source and rebuild instructions above. No competitive GPU run or
Kaggle performance score is claimed.
