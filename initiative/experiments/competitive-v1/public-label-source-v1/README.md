# Public versus accepted-v2 labels on the adapted MRI model

This isolates a supervision-source question. The control keeps the actual accepted
label values and masks; the public arm uses the pinned public soft-label table for
exactly the same2247 fitting studies. The1156-study evaluation frame, masks,
leakage-component folds, image inputs and accepted release stay unchanged. No gold
labels, gold-selected target blend/tier, EMA teacher, entropy term, laterality
transform, extra images, new backbone or ensemble is introduced.

The original public source is [Shio-Koube/knee3](https://huggingface.co/datasets/Shio-Koube/knee3),
revision373f4e1daf36984fa48bf3fc8fd9883fe17cb584. Its public identity/revision were
queried independently and the original630153-byte llm_labels_v4_blend.csv was
fully downloaded/SHA verified againsteb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4.
All2247 fitting studies are covered; no missing source row creates an implicit
negative. Only fitting-row probabilities enter the overlay. Original raw CSV is
kept in ignored artifacts; pinned source and fitting-only overlay are recorded.
This provenance does not assert clinical correctness or every OSS member's label
source. The HF repository has no dataset card/license metadata at this revision;
these are public competition-derived annotation probabilities, not external images
or source-trained model weights. Preserve source attribution and private milestones.

Both roles use the current plain target-specific attention DINOv3-L rank32 recipe:
all120 encoder projections/10,223,616 trainable adapters, native384/16 uniform slices/
max6 selected sequences, effective8, encoder/head LR1e-4/3e-4, warmup281/cosine,
FP16/scaler128/chunk8 and ten fixed epochs/2810 updates. The unused zero-weight
patch-mean block is absent in BOTH roles; this is a label-only contrast with common
model graph, initialization and inputs. The configurations differ only in the
fitting-supervision enabled flag. No best-epoch selection or gold tuning.

Primary: fixed-final all12 masked macro AUC on the unchanged weak-label development
frame, accompanied by all12 targets and paired whole-leakage-component uncertainty
with undefined draws retained. This evaluation favors its existing label semantics;
actual clean MRI-only Kaggle comparison is separately required before claiming
competition improvement. Do not select per-target label sources from exposed gold
summaries. Meaningful gains require another matched seed/fold.

The narrowly extracted PR4 label loader validates complete fitting coverage,
release/fold/roster/source hashes and disjoint leakage components. A new explicit
soft-target admission affects fitting only; default loader/evaluation binary
validation stays strict. Existing effusion overlay and old configurations retain
their behavior. Actual synthetic trainer full/stop1/resume state and CSV equality
are exercised in both source arms; software checks are not MRI/GPU or accuracy
acceptance.

Current status: local implementation, fitting overlay and paired configs prepared.
No new GPU job or clinical result. Keep live877134 and queued878853 immutable.
Actual CPU input/official initialization, separately funded checkpoint/publication
obligations, real allocated-GPU recovery and exact runtime/source admission remain
required before a distinct production dispatch. No bulk cache/weight transfers or
repeated original publications are authorized by this preparation.
