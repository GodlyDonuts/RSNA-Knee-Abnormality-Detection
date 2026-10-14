# Owning-node dense-suffix CPU admission review

PR59 head 1b83b9956f2009c9572298f7da8e61623964ebcb was reviewed and integrated locally. The runner preserves the original source public-input recipe, full 2,247/1,156 boundary, accepted label overlay, mounted official base and original input receipt. Changed loader modules require a fresh 16+16 tensor replay; receipt reuse alone does not accept them. It compares fresh head values, factory RNG and frozen official tensors across rank32 and dense suffix arms, and requires zero initial residuals.

The original configurations remain unchanged: frozen first12 blocks, suffix rank32 encoder7,077,888 learned parameters/LR1e-4 versus dense150,994,944/LR1e-5. This is a predeclared adaptation comparison, not a pure parameter-count causal experiment. It streams acquired triplets online and requires no full prefix cache. CPU input checks do not authorize a GPU or prove accuracy; GPU recovery and production storage remain separate obligations.

Focused QA:35 passed in5.16s. Ruff/format passed. This local integration has not been staged or dispatched; no original live trainer changed. Full project QA is running before publication. The stage manifest remains a proposed template, not an actual dispatch receipt.

The original full run completed3261passed/6failed/8skipped in1128.94s. Six failures were subprocess import environment issues: user-site scripts shadowing and an absent repository PYTHONPATH. With PYTHONNOUSERSITE=1 and the absolute repository PYTHONPATH, all six replayed successfully in16.30s, with no implementation change. The same-source result union covers3267passed/8skipped; this is not presented as a single green full invocation. See QA.json.
