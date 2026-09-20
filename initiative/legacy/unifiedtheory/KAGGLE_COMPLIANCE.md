# Kaggle competition compliance boundary

Audited against the official RSNA Knee Abnormality Detection rules,
evaluation, code-requirements, and competition-API metadata on 2026-08-10:

- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements

This pipeline is designed around the following hard constraints:

- hidden-test inference is image-only, internet-disabled, and time-budgeted;
- report text and the public NLI teacher are used only for training labels;
- no validation or test image is manually labeled or manually predicted;
- scanner groups never cross folds;
- weak labels never select their own checkpoints: selection uses available
  expert validation masks only;
- external pretrained assets are public and listed with licenses in
  `external_assets.json`;
- the metric is macro ROC AUC over twelve targets, so the final selection and
  submission blend operate on ranks; probability calibration is not used as a
  leaderboard optimization surrogate;
- the competition is notebook-only, permits five submissions per day, and
  permits at most two selected final submissions;
- the submitted notebook must have internet disabled, finish in at most nine
  GPU hours, and write exactly `submission.csv`;
- `kaggle_offline_infer.py` validates the exact sample-submission schema,
  study order, finite probabilities, and the nine-hour notebook boundary;
- `build_kaggle_patch_kernel.py` stages a public, internet-disabled GPU kernel
  with separate public code and checkpoint datasets, and fails unless all 15
  fixed patch checkpoints are attached; building it does not publish or
  submit anything;
- competition data and derived private artifacts must not be committed or
  shared outside a team whose members have accepted the competition rules.

The three public report-label sources are CC0 Kaggle datasets. RadImageNet is
an externally pretrained model. The attached Kaggle mirror declares
CC-BY-NC-SA-4.0; the competition-specific winner-license clause explicitly
allows input data or pretrained models with an incompatible license, but the
asset must remain disclosed and independently obtainable. The source and
license distinction is recorded in `external_assets.json`; it must not be
represented as code authored by this project.

The repository is public. Competition-specific code must not be pushed as a
private competitive advantage: if published, the applicable rules require it
to be made available to all competitors through the competition's associated
Kaggle forum or notebook under a permissive license. Publication and any
leaderboard submission remain explicit user actions.

This file is an engineering audit, not legal advice. The official rules remain
authoritative.
