# Multi-source supervision: source information before source cycling

The owner brief is now available and preserved verbatim in `owner-brief.md`.
The previous missing-file observation was correct when made and is superseded.
This experiment adds a working target constructor, not a new trainer or accuracy
claim. It leaves all existing readers, original releases and scoring owners intact.

The useful question is whether an additional annotation family corrects useful
training information. For ordinary BCE at fixed logits, the weighted average of
source losses equals BCE against the weighted target mean. Epoch cycling adds
optimizer/order effects; it does not add annotations. The initial experiment
therefore uses a deterministic mixture with the reference mask and denominator.

## What is actually used now

- The licensed-parent uniform/source-label comparison uses Steven v6 or the
  separately admitted S6 target array: 3,887 fitting studies, 25% source loss /
  75% detached parent, three epochs / 1,458 updates. These source arrays are
  alternatives in completed runs, not three independent annotator votes.
- The completed first spatial reader uses the original report-derived release,
  75% report / 25% parent on observed cells, parent on missing cells. Its weak-label
  internal result cannot select mixture weights or establish clinical truth.
- A separate whole-cohort Astra image-label export has not been located. The
  owner's description remains a provenance lead, not a disproven claim. Four
  original release files match pre-Devin Git and HF content identities; none
  was reverted. See the original `label-source-lineage-20261018-v1` audit.

## Working implementation

`multisource_overlay.py` retains the source tensor, explicit semantic states,
availability, family counts, normalized weights and descriptive disagreement.
Only meaningful, admitted, common-observed cells can change; accepted override
cells and all original masks remain identical. Explicit uncertainty/unmentioned
states abstain. A genuine probability of .5 remains meaningful when admitted as
such. It does not infer semantic states by numeric threshold.

Canonical sources must have disjoint documented transitive ancestry. Mirrored
files, a blend and its parent, or blends with a known shared ancestor cannot
be extra votes. The ancestry graph and per-target admission are supplied by an
external source review: the constructor cannot discover undocumented ancestry
or establish probability calibration. Caller-supplied metadata is not clinical
validation. It rejects exact fitting/evaluation identity or component overlap;
that check cannot detect unknown patient links.

Synthetic checks cover single-source identity, unchanged inputs/masks/overrides,
missing-sentinel handling, genuine .5 probabilities, anchored effective-batch
BCE loss/gradient equality and twelve refusal cases. No clinical rows, MRI,
Gold or confirmation were read by these tests. Source sampling is intentionally
not added; no extra MRI passes or deployment inputs are required.

## Admission decision and next concrete comparison

Existing source receipts quantify 46,644 differing Steven/S6 cells on the same
3,887 studies. That is a previous numerical comparison, **not** the count that
would change in a semantically admitted multi-source overlay. Steven's opaque
blend does not permit recovering which .5-like values originated as abstentions.
S6 has a declared blend/JEV rebuild, but its ancestors and source semantics must
be traced to primitive sources before claiming nonoverlapping votes against
Steven. Neither agreement nor provider names settle that question.

The current source table therefore does not yet admit a production multi-source
tensor. Actual changed-cell count is unmeasured; no new multi-source GPU fit was
launched. This is a specific pending source-construction task, not a need for
new image acquisition or another training framework. The existing completed
uniform control is the proposed reference only if masks, source-parent fraction,
initial state, MRI order/views, updates and inference remain exactly matched.
If semantic admission requires new masks, use a matched pair instead of that
historical control. Do not combine this contrast with dense/path changes.

Next: inspect the already-acquired S6 rebuild and primitive source metadata on
their owning node, then build one fitting-only safe-cell overlay and aggregate
change audit. Preserve legacy fallback cells explicitly. If no extra family can
be justified, retain this negative admission result and skip duplicate training.
No probability/source weights will be chosen from protected Gold or leaderboard
search. The retained source tensor supports later observation-process research,
but this implementation does not estimate annotator reliability or latent truth.

## Actual ancestry finding after the first constructor

CPU952283 read the already-acquired exact S6 rebuild, without any label rows.
Authenticated public metadata now identifies the nartaa blend as Gemini plus
Steven; JEV HYBRID is95%Gold-selected Steven/Pilkwang hard-swap plus5%JEV text
ensemble. Therefore Steven and S6 are overlapping families, not extra independent
votes. The two current source cards expose primitive Gemini probabilities and
a separate observation-mask table. Their declared unmentioned conventions must
be reconciled before using values. This changes the next action: acquire only
the missing primitive small files privately once, reuse original fitting arrays,
and test the baseline-versus-Gemini safe-cell contrast. Do not launch the naive
Steven/S6 mean. Current cards do not reproduce every historical numeric ancestor.
[Actual hashes, versions and source declarations](actual-overlapping-lineage.json).

## Actual primitive fitting-only audit, CPU952374

All three original files acquired by952331 were reused after schema-only
reconciliation952341; no repeated download. Gemini meaningful-source y matches
its probability file on every addressed cell. Recovered baseline y matches all
current training targets except two Synovitis cells. A source-mask intersection
and exact-value identity guard leaves those cells unchanged. Equal-family
probabilities would change35418/46644fitting cells, with no mask/coverage change.
Mean proposed changes are largest for Effusion(~.15955), PF OA(~.12959), lateral
and medial OA(~.10636/.10625) on eligible cells. These are target movements, not
ranking gains or clinically adjudicated corrections. This audit has NOT yet
materialized an overlay, integrated the trainer, admitted a new GPU run or read
protected gold target columns. Source-generation claims remain source-relative;
shared reports can induce correlated errors. [Actual all12target evidence](actual-primitive-source.json).
