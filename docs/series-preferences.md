# Reviewed acquisition preferences

The reviewed sampling table preserves all 24,338 retained series and every existing
metadata column. It adds `Selection_Tier`: zero for ordinary candidates and one
for three reviewed alternatives. This affects ordering when the series budget is
limited; it does not remove an image or alter its pixels, features or labels.

- For 064:14, prefer coronal series 11 before 6, and sagittal series 12 before 5.
  Native/cache comparisons and the independent 71-slice cache review support the
  clearer presentation. Different geometry and coverage remain preserved.
- For 084:27, prefer the focused axial series 201 before wide-field series 101.
  Both visual readers saw a smaller main knee and a cropped peripheral second
  limb in the three sampled views of 101. This sampling preference does not claim
  a complete visual review of its 200 slices or justify an automatic crop.

The sampler retains plane coverage and randomizes within tiers during training.
Evaluation keeps its deterministic within-tier ordering. Alternatives remain
available when capacity permits. Tiers are never supplied as model features.
Without a tier column, behavior and random-number consumption are unchanged.
These are pairwise preferences, not required-series rules: a different ordinary
candidate in the same plane can still take the slot under the existing ordering.

Rebuild from the exact bound metadata and corrected cache:

```bash
python initiative/audits/reviewed-series-preferences-v1/export_preferences.py.txt \
  --repo . --cache ../cache256-v3-official-recovery-v1/cache \
  --policy initiative/audits/reviewed-series-preferences-v1/accepted-preferences.json \
  --expected-policy-sha 65e5516c1209c71659227c2358f5bf2f988f0a9de960954511e77c13de3229ca \
  --output data/series-sampling
```

The output directory must be new. The exporter verifies the accepted evidence,
source CSV, six referenced cache files, exact series identities and model features.
The output SHA-256 is
`14eb050d081abe9fce638f5dcfb2cedc688fdac5ace9a62bf9b1cf152421eada`.
The identical generated CSV is also preserved under the audit's `export-001/`.
Use the same table for training, resume, weak validation and gold inference;
existing checkpoint bindings enforce its bytes. The released test table remains
separate and unchanged.

Tests cover preference during random sampling, plane coverage, retained
alternatives, invalid tiers and unchanged default behavior. Verification on real
metadata preserves order and RNG behavior for all 4,405 unaffected studies and
checks 240 sampling configurations for the two reviewed studies. Eight real-cache
loader passes used no clinical or gold columns. A distinct code reviewer also
compared the previous loader against the new loader on all 4,407 studies,
verified 950 pair/plane configurations and 300 full-capacity retention checks,
and found no concrete code concerns. The independent review and replay helper are
preserved in `initiative/audits/series-preference-independent-code-review-v1/`.
These are input-selection checks;
final label release and actual final-data training remain unfinished.
