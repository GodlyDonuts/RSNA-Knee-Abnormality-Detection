# Actual source-observation attribution: no blanket missingness explanation

CPU955379 read the unchanged3887fitting studies and all12targets for original
and family labels. No labels, masks, source weights, model, Dataset or job changed.
At hypothetical dense-student logits exactly equal to both stored teacher views,
both-source-declared-unmentioned cells account for only **4.64%** of the family
source's pooled per-cell squared logit-gradient energy (original labels4.22%).
That does **not support** a claim that these cells dominate overall adaptation.
It does not establish that masking them cannot help or that labels are correct.

There are3967such cells out of46644 (8.50%). Baker's is more
concentrated: both-unmentioned cells account for20.70%of that target's energy.
Synovitis is structurally different:1703baseline-only-observed cells account
for49.84%of its source energy. Fracture has1325Gemini-only-observed cells whose
values were not substituted by the safe-common mixture. These are provider
observation/definition disagreements, NOT confirmed label errors or MRI truth.

Source categories came from admitted explicit observation masks, never from a
probability threshold. All32actual-row canonical-loss/autograd checks agree
within1.63e-19, including a hypothetical silent-mask contraction with unchanged
N*12denominator. No hypothetical mask was written or admitted. No reports/Gold/
confirmation/evaluation membership was read.

Interpretation limits: this is an initial dense-logit counterfactual, not the
actual sampled/dropout student's trajectory, CNN parameter-gradient norm or
clinical/held-out/Kaggle accuracy. Energies sum individual cell derivatives before
batch cancellation, clipping and network Jacobians. The raw field named
`macro_cell_energy_fraction` is a POOLED cell-energy fraction (ratio of sums),
not an unweighted mean of12target fractions; all per-target fractions remain.

Decision: retain current family/finding fits unchanged. Do not launch a blanket
masking or alpha sweep from this audit. If further supervision work is warranted
by actual outcomes, prioritize fitting-only source-definition/report-evidence
review of the concentrated disagreements, retaining uncertainty and source
provenance. Two report-derived sources do not identify clinical truth or reliable
source confusion matrices. Whole94training's0.949tie also prevents treating
reduced pooling error alone as a demonstrated accuracy path.
