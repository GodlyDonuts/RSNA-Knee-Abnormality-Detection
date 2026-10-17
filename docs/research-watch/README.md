> **Historical research archive.** Live scores, jobs and current mechanisms are in
> [current state](../current-state.md) and the [research-agent handoff](../research-agent-handoff.md).
> The October 15 status and N3 proposal below have been superseded: N3 epoch 16
> actually scored 0.894; the attributed external parent scored 0.949. Read the
> [October 16 source-contract correction](additional/rsna-public-training-contract-update-2026-10-16.md).
> Preserve the original reports; do not execute their historical “next” instructions.

# RSNA Research Watch full research archive

This collection preserves the complete available research reports, proposal text, experiment-ledger reviews, public-source analyses, provenance registers, metadata inventories and synthetic-only checks assembled through October 15, 2026. It is a research archive, not a declaration that every proposal was implemented or every inventory entry was inspected in full.

## Read first

1. [Latest status and late source reviews](2026-10-15-late-findings.md)
2. [Current compact challenger proposal N3](dossier/rsna-compact-native-coatnet-challenger-2026-10-14.md)
3. [Final-week decision brief](dossier/rsna-research-ledger-20261014/final-week-decision.md)
4. [Reconciled experiment ledger](dossier/rsna-research-ledger-20261014/experiment-ledger-reconciled.md)
5. [Experiment-family coverage](dossier/rsna-research-ledger-20261014/experiment-family-coverage.md)
6. [Full file index](INDEX.md), [machine-readable manifest](MANIFEST.json), and [scope and exclusions](SCOPE.md)

## Status at publication preparation

Latest inspected repository commit: [70c3a6fab25f47e208afd4f0388296bf3b2e948a](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/commit/70c3a6fab25f47e208afd4f0388296bf3b2e948a). E8 completed at .863; E5/E6 remain the best recorded public score at .872. E10's portable public example was running in the 22:37:21 UTC receipt, with no competition submission created and no score. A successful packaging or synthetic check is not evidence of improved MRI accuracy.

Compact CoAtNet N3 remains an unexecuted proposal. Its current specification corrects normalization and hidden encoder dropout assumptions. Historical N1/N2 and prior alternatives are retained in their report history where available; obsolete recommendations are not silently erased. The late addendum takes precedence on current status and the newly screened methods. No result is promised.

## How this archive is organized

- `dossier/` preserves the existing delivered dossier's research content and historical README
- `additional/` contains reports, metadata inventories, synthetic results and scripts absent from that dossier, plus the full text of its Word memo
- The manifest maps original source names to published paths with byte counts and SHA256 digests. Identical copies are mapped to one publication file
- Reports retain their original dates, source pins, qualifications and historical statements. Read their status against the latest addendum rather than treating all snapshots as simultaneous

## Historical precedence

The historical dossier README mentions publication pending confirmation; that statement was superseded by the October 15 request to publish this collection. The original experiment ledger and initial strategy/adversarial reports are retained for completeness; use the reconciled ledger, final-week decision and current N3 proposal for decisions. E8 running/prepared snapshots are superseded by E8 .863. E10 prepared snapshots are superseded only by the running public-example receipt, not by any completed score. Slice-spacing, missingness, loss-weighting and pretraining screens did not promote new experiments beyond the bounded priorities described above.

## Reproducibility limits

Synthetic scripts and their existing outputs are preserved as research evidence. They have not been rerun for publication. Several scripts contain historical execution paths; geometry and sampling scripts expect source functions from the pinned repository snapshots. Repository code copies are not republished here. See the synthetic reports and source-copy manifest to reconstruct the exact inputs. Do not treat the scripts as a turnkey portable training package. Metadata classifiers characterize audit-directory names; they do not establish that all 1,090 entries were reviewed as completed experiments.

The competition final deadline was verified from indexed official sources as October 22, 2026 at 23:59 UTC. The late addendum records the limits of same-day rule-page coverage.
