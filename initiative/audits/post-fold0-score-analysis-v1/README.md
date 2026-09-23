# Recomputed fold-0 ranking and rare-component sensitivity

`recompute.py` joins the immutable 1,156-row outer-fold prediction file to the
accepted `study_labels.csv` by study UID, applies each target's observed-cell
mask, and independently reproduces all twelve AUCs and the trainer's macro
**0.7783102649**. `metrics.json` records both source SHA-256 values, the exact
sklearn version, positive/negative study and conservative leakage-component
counts, and bounded leave-one-rare-component-out recomputations. Run
`python initiative/audits/post-fold0-score-analysis-v1/recompute.py` to check
the frozen JSON without replacing it.

The two observed MCL-positive outer-fold components yield AUC **0.5973**;
removing either one gives **0.4514–0.7431**. This measures sensitivity to those
two reports, not a confidence interval or a clinical truth estimate. The
lateral-meniscus target has 149 positive and 821 negative observed outer-fold
studies, with AUC **0.6472**. Its weaker ranking has substantially more support
than MCL and is the clearest controlled target for testing spatial patch
evidence under the unchanged DINOv3-L protocol. Fracture also has only two
positive outer-fold components, Baker's has three, and synovitis has five
negative components; their high AUCs remain fragile.

This is one weak report-derived development fold. No Kaggle score, hidden-test
runtime, new clinical image labels or second-fold result is inferred.
