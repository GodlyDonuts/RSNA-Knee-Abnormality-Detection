# DINOv3-L spatial patch-attention challenger

This is the first controlled model evolution after the fixed-final three-fold
global-CLS baseline. The frozen official DINOv3-L encoder returns its
normalized CLS token and the 14×14 grid of normalized 224-pixel-image patch
tokens; its four register tokens are excluded. A small trainable head adds
explicit two-dimensional patch coordinates, learns separate patch attention
for each of the twelve targets, and combines each target's patch summary with
the global CLS representation. The existing masked slice and series attention
then aggregates the same sampled MRI images. This changes the head's access to
spatial evidence without changing the accepted labels, folds, 256-pixel cache,
sampler, effective eight-study batch, optimizer, official weights, seed, or
ten-epoch fixed stopping schedule.

The predeclared primary hypothesis is better lateral-meniscus ranking than the
global-CLS baseline, whose pair-weighted three-fold AUC is 0.689. The full
baseline's twelve-target macro AUC is 0.7522657161782244 on 3,403 observed
outer-fold studies. MCL and fracture have too few positive components for a
single-fold improvement to be reliable; do not choose architecture or epoch
from those outer-fold results. The reserved confirmation and official gold
sets are never used for training or selection. A future Kaggle score is not
claimed while its hidden deployment still fails.

The [pilot configuration](pilot-config.json) uses fold 0, 100 training and 100
outer studies, two epochs, one H100 and two CPUs. It is diagnostic-only and
cannot produce a competition checkpoint. Gate production on a real allocated
CUDA context, a validated cache/release/header preflight, finite forward and
backward, at least one completed optimizer update, a terminal checkpoint and
no unexplained failure. The distinct [production configuration](production-config.json)
keeps the baseline's full fold-0 schedule and saves every completed epoch.
Only after a successful pilot and fresh Newton quota/Slurm review should a
single production job be submitted. Preserve each immutable completed epoch
privately, with full download/hash verification, under a new identity. A
meaningful architecture comparison ultimately requires all three outer folds
with the same fixed recipe.
