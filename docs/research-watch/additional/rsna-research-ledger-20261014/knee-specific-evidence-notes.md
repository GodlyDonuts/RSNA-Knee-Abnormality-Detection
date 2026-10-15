# Knee-specific evidence: what to reuse and what to reject

Research only, 14 October 2026. No data acquisition, training or clinical inference.

## Two-image evidence is not cross-plane agreement

De Smet and Tuite's 174-patient arthroscopy-correlated study found that requiring abnormalities on at least two images increased positive predictive value, significantly for lateral meniscal tears (83% to 96%, p=.02); the medial improvement was not significant. This supports evaluating persistence of the relevant morphology, not a universal requirement that every plane agree. [Primary study](https://pubmed.ncbi.nlm.nih.gov/16985134/)

An earlier 200-patient study found that some tears were visible only sagittally. Therefore forcing sagittal/coronal prediction equality could suppress valid evidence. Two independently acquired images showing surface contact are not equivalent to repeating an image, interpolating one image, or seeing a high study-level logit in overlapping triplets. No new labels can be derived from this rule alone. [Primary study](https://pubmed.ncbi.nlm.nih.gov/8517286/)

**Implication:** if testing continuity, compare same-budget anatomy-aware interactions against shuffled/order-removed controls. Do not impose a hard two-window threshold on a teacher whose window scores are unvalidated. Existing neighbor-context and plane-memory experiments already cover parts of this hypothesis.

## KAMRNet: useful anatomy precedent, poor direct replacement

The 2025 multicentre study includes 13,419 patients and 14,962 examinations. Primary-abnormality AUC declines from .898 internally to .852 and .812 in two external tests. It requires all three PDwFS planes and excludes missing-plane cases, limiting its relevance to our missingness objective. These numbers are different endpoints/populations, not an RSNA ceiling. [Paper](https://pubmed.ncbi.nlm.nih.gov/41079026/) · [Publisher full text](https://www.thelancet.com/pdfs/journals/eclinm/PIIS2589-5370%2825%2900467-5.pdf)

The authors' model code uses three plane-specific ResNet branches, a global pass, attention-derived local crops, and a second local pass. The inference code iterates task-specific checkpoints. Thus “multi-task system” does not establish one shared deployed checkpoint with twelve outputs. No direct adoption is recommended under the current solo model objective. The transferable idea is a shared-encoder local/global comparison, already overlapping our patch-focus and adaptive-resolution candidates; this paper does not justify another unbounded architecture branch. [Author model](https://github.com/zqiuak/KAMRNet/blob/main/model_forallplane_MMAL.py) · [Author inference](https://github.com/zqiuak/KAMRNet/blob/main/inference.py)

Data access requires a research proposal and an agreement. No access was requested, and no new data eligibility is established.
