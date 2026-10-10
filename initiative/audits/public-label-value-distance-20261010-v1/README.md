# Public-source values differ materially, not just at floating precision

Actual fitting-only CPU897368 compares the13760 originally observed cells against the exact public-source probabilities. Mean absolute distance0.138868, median0.025,95th percentile0.91. At the fixed descriptive0.5 threshold,1624/13760 (11.8%) fall on opposite sides of the original binary label. This does not identify which source is correct and does not change any label, mask, threshold or model. Only fitting rows were read; no outer/protected targets or individual records were exported.

These quantities refine the supervision-confound interpretation: many changes are small, but the long tail is substantial. The already submitted mask-controlled experiment897333 tests the value change under identical original masks. Do not harden the probabilities or select target mixtures from this descriptive audit.
