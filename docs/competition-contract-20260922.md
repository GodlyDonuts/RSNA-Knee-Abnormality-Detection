# Competition constraints checked on 2026-09-22

Root read the rendered public Kaggle pages using the Browser skill. The normal
web reader returned no page content for the rules and evaluation pages, so it
was not used as evidence of their contents. This check reads the rules; it does
not accept an agreement or enter the competition on the owner's behalf.

The main score is the arithmetic mean of 12 target ROC AUCs. Inference must run
in a Kaggle Notebook without internet, finish within nine hours on CPU or GPU,
and write `submission.csv` in the official column order. The entry and team
merger deadline is October 15, 2026; the final submission deadline is October 22,
2026. Deadlines are 23:59 UTC. The efficiency track also considers runtime, but
our primary objective remains the main score. The organizer lists 19 principal
data-contributing institutions and three additional sites. This confirms a
multicenter dataset; it does not identify an individual study's source center.
[Official overview, evaluation, timeline and acknowledgements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)

The rules allow five submissions per day and selection of two final submissions.
External data and models must meet the host's accessibility and cost conditions.
Competition data must remain restricted to eligible participants. Winners must
provide reproducible training and inference code, weights and environment details.
The page names CC-BY-NC 4.0 for the winner license, but also contains conflicting
boilerplate about commercial restrictions; this note does not resolve that
inconsistency or confer rights to an external dataset.
[Official rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

Scoring replaces the three example test studies with approximately 1,300 studies.
Reports are unavailable at test time. Fluid sensitivity and fat suppression are
distinct series attributes. Prevalence can differ between training, public and
private test sets. The page describes 86 retained DICOM metadata tags, varying
geometry/intensity and several compressed transfer syntaxes. Therefore the final
model must use images and permitted test metadata, and runtime estimates must
include decoding the hidden DICOMs.
[Official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

The current host clarification permits commercially hosted LLMs to extract labels
from reports, subject to the accessibility and cost rules. A straightforward
registration or click-through dataset agreement may be acceptable; institution
approval or lengthy credentialing can be a barrier. Competition prizes alone do
not make a noncommercial dataset automatically prohibited, but dataset-owner
terms still apply.
[Host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)

The KneeCoT discussion still contains the host's explicit prohibition. No reversal
appeared in that thread on this check. Exclude KneeCoT from the proposed training
pipeline and do not assume derived weights evade the restriction.
[KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109)

The exact severity rubric and independent image-label clarification remain pinned
in the [September 20 audit](competition-contract-20260920.md); those two discussion
pages were not re-read in this update. No newly authenticated byte comparison of
all original DICOMs is claimed by inspecting these public pages.
