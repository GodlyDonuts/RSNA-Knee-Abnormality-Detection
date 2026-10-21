# RSNA Knee agent guide

Updated **October 21, 2026, 17:28 UTC**. Read [current state](docs/current-state.md), then [execution continuation](docs/codex-continuation.md). Actual original journals and service state supersede snapshots. The owner authorizes independent training, useful Kaggle entries and **300 Colab resource units**. No subagents or Devin.

## October 21, 17:28 UTC deployment update

Original Newton consumer3332 stopped after a successful qualification compile:
its inherited parser expected an older status name. The existing private
Dataset12762386v1 and exact compiled source were reconciled read-only; no
creation/upload/compile was repeated. Distinct finite continuation **8547**,
source4297039c, completed exact private two-T4 qualification **140062137v1** and public
competition execution **140062846v1**. Submission **57491555** was accepted
once at17:26:19UTC from `godlydonuts/rsna-predictive-c100-c-e6-055a8f8ed7e0` v1.
Initial authenticated hidden status/score are null: pending/unknown, not a score.
[Actual qualification, public execution and entry](initiative/research/predictive-C100-final-entry-20261021-v1/README.md).

Only the waiting local Colab consumer3333 was explicitly stopped before any
handoff/account phase. Corrected consumer **8603**, same source4297039c, reads
the same original handoff and protects Newton's budget. The actual Colab A100
training and queued publisher remain untouched; latest observed step2300/2916.
Never restart3332/3333 or repeat the already-pushed qualification.
[Exact continuation and receipt](initiative/research/predictive-final-status-recovery-20261021-v1/README.md).

Existing passive Newton logs show a nonzero diagnosis-conditioned contribution
(final sampled logBF RMS0.3729, mean posterior shift7.54percentage points), with
mixed fitting BCE differences across the ten sampled updates. The final seven
fitting studies show−0.03783BCE, but this is not held-out accuracy or proof of a
gain. [Original aggregate evidence](initiative/research/predictive-C100-passive-final-20261021-v1/README.md).

## Current execution

- **Newton C100 GPU995351 COMPLETED0:0 on evc27 at16:51:31UTC**, all six epochs/2,916 updates on3,887 fitting studies. All six immutable model checkpoints are privately verified: HF1–3 unchanged,4–6 in existing private Kaggle Dataset12762120v1. Original backup1000318 created it once then stopped on immediate GetDatasetStatusHTTP403. CPU1000389 reconciled that same READY Dataset and fully downloaded/size/SHA verified16objects, with zero new creation/upload. Original failure retained. Original final consumer3332 stopped on a response-name mismatch; successor8547 completed qualification/public acceptance and submitted57491555 once. [Actual final/preservation](initiative/research/predictive-C100-private-backup-20261021-v1/actual-final-preserved.json).
- **Independent Colab A100 C100 is training epoch5, update2,300/2,916**, with finite encoder/head gradients. Seed20261021, all3,887 fitting studies,73,271,124 CoAtNet/predictive parameters. This branch is not DINOv3. Its exact step972 recovery performs no network operations during fitting. A paced publisher is queued afterward, retaining all original checkpoints and requiring seven private publication proofs. [Actual progress/recovery](initiative/research/colab-C100-epoch2-recovery-20261021-v1/README.md).
- **Final continuations8547 (Newton) and8603 (Colab), source4297039c, are live**; Newton has accepted its fully preserved final, Colab is still training. Both use the separate tested faster/resilient inference successor and require their own actual two-T4 qualification. Qualification140062137v1 and public140062846v1 passed; submission57491555 is pending/unknown with no score. Earlier86015/95310/89950/76338/67205 are stopped; do not restart them. [Newton](initiative/research/predictive-fast-newton-deployment-20261021-v1/README.md), [Colab](initiative/research/predictive-fast-colab-deployment-20261021-v1/README.md).
- **B57482618 and C57482739 FAILED** with an unhandled hidden-notebook error, zero output bytes and no score. **C0 57482858 remains pending/unknown** at the latest authenticated observation. Original outcome consumer81657 stopped on a JSONDecodeError; a new strictly read-only query at16:48 confirmed unchanged exact outcomes. Do not restart the old consumer. Never resubmit those versions. [Actual failures](initiative/research/predictive-hidden-input-audit-20261021-v1/actual-hidden-outcomes.json).
- Actual CPU audit140048037v1 checked203,001 headers from1,087/4,407 accessible training studies; all tested assumptions passed. This weakens, but does not eliminate, the metadata hypothesis. Hidden failure cause remains unproven. New serving preserves normal input bytes, retries numerical FP16 failure at FP32, and recovers usable MRI from certain rejected metadata. Actual trained qualification remains mandatory. [Evidence and limits](initiative/research/predictive-inference-resilience-20261021-v1/README.md).
- Private CPU-only largest-study stress140052135v1 is COMPLETE: all seven size-selected studies passed original and successor preparation, with bitwise-equal complete contexts. It inspected2,830/4,407 studies for selection; this does not establish hidden robustness or the earlier failure cause. Observer4458 completed; never repeat. [Actual result](initiative/research/predictive-large-study-CPU-20261021-v1/README.md).
- Private CPU four-core experiment140055931v1 completed six trials. Two threads per preparation lane reduced balanced warm time27.0233→21.6329seconds (19.95%) on three public studies. Image difference was1.1920928955078125e-7, remaining fields exact. Retain as a contingency requiring actual trained prediction comparison; current numerical serving remains unchanged; corrected controllers8547/8603 are active. [Evidence and decision](initiative/research/predictive-thread-budget-CPU-20261021-v1/README.md).
- Best actual Kaggle: **external R6 reproduction0.958; own adaptations0.949**. Source-zero/pointwise/ranking each0.948. No predictive B/C/C0/C100/Colab score or accuracy improvement is established. Deadline **October22 23:59UTC**; latest full-scored launch target **October22 noonUTC**.

## Working rules

- Use `/Users/sairamen/projects/RSNA-Knee-Abnormality-Detection`. Inspect exact original job/source/output bindings and journals before action. Never duplicate jobs, transfers, notebook pushes, uploads or entries. Existing intent/uncertainty requires read-only reconciliation. Lost observation is not terminal evidence.
- Freeze healthy numerical sources and configurations. Use distinct explicit successors for genuine corrections. Preserve original failed attempts and negative results. Do not alter accepted labels/masks/fitting order or claim weak-label agreement is independent validation. Parent exposure and historical Gold58 selection confound this branch's internal evaluation.
- Ordinary physical one-/two-H100 nodes are authorized; physical four-/eight-GPU nodes remain prohibited. Use minimum useful CPUs. No driver/device-isolation bypass. Preserve the original Colab ten-hour fit bound and300-unit limit. No Kaggle purchases or billing changes.
- Keep20GiB shared/node reserve and5,000 shared files after all liabilities. Latest shared quota leaves22,872,809,472bytes/5,371files before123MiB/370future files. Both final deployment scopes and the new private backup are explicitly funded; fresh state governs. No deletion credit or double-counting paid bytes.
- Keep raw MRI, reports, IDs, arrays, signatures, private predictions, external restricted source and weights off Mac/Git/public logs. Large artifacts stay on authorized private cluster/Colab/Kaggle storage. Credentials only ignored0600 `.env`, RAM/stdin/ASKPASS, never argv or logs.
- Preserve each immutable epoch once with complete source/base/license/runtime/fitting provenance and independent full download/size/SHA verification. HF storage is currently exhausted; private Kaggle backup is an explicit distinct successor, not an HF success. Do not repeat existing whole-byte verification without a concrete concern.
- Kaggle mutations use the canonical Newton ledger and single claim, fresh authoritative allowance/free GPU capacity and maximum five daily entries. At15:55UTC two slots and3,056.1426641 free GPU-VM seconds remained. Final Newton protects1,920seconds; Colab waits for its exact entry before using remaining capacity. Never submit public three-row CSVs, diagnostics, failed/uncertain versions or Stanford-derived weights. Public completion is not a hidden score.
- Preserve best0.958 and reconcile the two final selections before deadline. Latest selection observation was0/2 manually selected, automatic best-score policy active; this is not a manual lock.
- Keep GitHub snapshots current and link original receipts. Current source is **not HF-mirrored**; latest verified small-source mirror remains v7. Notify meaningful progress/results/failures or required input. Do not promise a win or call software checks an accuracy gain.
