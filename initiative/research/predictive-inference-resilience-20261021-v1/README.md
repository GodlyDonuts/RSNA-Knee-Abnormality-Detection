# Prospective MRI inference resilience

B57482618 and C57482739 actually failed with an unhandled hidden-notebook
exception and no score. Kaggle does not expose their hidden traceback. C0 was
still running at the last observation. This is **not a root-cause diagnosis**.

The distinct final-serving successor addresses concrete fragility in the
original worker while both original C100 fits continue unchanged:

- Keep the original decoder and exact successful tensors. If it rejects a
  study, recover usable original monochrome integer MRI, ignoring non-DICOM
  sidecars and unusable frames. Filesystem study/series identity is authoritative;
  absent ordering metadata uses source order. Small matrices incompatible with
  the existing 4-by-4/halo8 renderer are excluded. Repairs and rejections are
  counted. If no usable MRI remains, fail explicitly.
- Retry a numerically invalid FP16 prediction at FP32 with the same learned
  model and same contexts. There are no constant predictions or substitute
  models. A failed FP32 retry propagates.
- Keep strict memory/precision thresholds in actual qualification. Do not
  discard a completed finite competition CSV solely because its historical
  memory peak exceeded the stricter qualification threshold. Kaggle's physical
  memory and runtime limits still apply.
- Include a private traceback on failure and exact serving/helper hashes.

Actual selected-frame CPU experiment140041310v1 already proved complete
original-context equality on the three public studies and 27.94% lower warm
preparation wall time. It does not prove hidden scale, rare input support or
accuracy. Every new final notebook must pass its own actual two-T4 trained
qualification with complete original-versus-optimized context equality,
original decoder path and FP16-versus-FP32 comparison before code submission.

Focused synthetic checks cover normal tensor/alias/order equality, optional
metadata defects, unsupported frames, explicit all-unusable failure, numerical
retry and actual staged Newton/Colab compiler lineage. They are engineering
checks, not clinical results. Accepted labels, training recipe, learned graph,
weights and original source files remain unchanged. Recovery can alter slice
inclusion or order on previously rejected inputs; that limitation is explicit.

The independent private CPU-only header audit140048037v1 is now complete:
203,001 headers from1,087/4,407 accessible training studies passed every tested
assumption. All three public full decoders passed. This weakens the metadata
hypothesis but is partial coverage; no hidden data, full pixel-body scan or
model numerics were tested. It consumed no GPU or submission slot and does not
prove the cause. The numerical/resource corrections remain prospective.
