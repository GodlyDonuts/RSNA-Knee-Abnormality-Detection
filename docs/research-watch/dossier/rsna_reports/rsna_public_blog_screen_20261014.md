# Additional public competition evidence screened

14 October 2026

[Author article](https://huggingface.co/blog/bishnoiyash/rsna-competetion), published 9 August, reports public .903 for submission 55361257. It describes a five-fold EfficientNet-B3 ensemble, so this is neither a .95 single-model result nor a current rank. Its reported Gold58 score fell from .9357 to .8568 after correcting in-sample evaluation; the same Gold58 also informed extraction and configuration selection. Cross-fitting the final image predictions alone does not erase upstream selection dependence.

The author reports a probability-to-integer truncation bug in label fusion and says the corrected fusion had not yet received a full retrain. Consequently .903 is not evidence of the repair’s benefit. No bound checkpoint, source repository, or independently readable scoring receipt was established in this screening.

Do not transfer the article’s proposed numerical label-quality ceiling to MRI AUC: agreement and ranking are different quantities. Do not treat study-ID grouping as verified patient-level separation. No new experiment follows from this source; it reinforces provenance and validation cautions already present in the dossier.
