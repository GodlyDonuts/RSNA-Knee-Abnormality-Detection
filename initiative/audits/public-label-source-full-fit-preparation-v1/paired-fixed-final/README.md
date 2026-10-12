# Fixed-final public-label supervision result

Original job879172 completed0:0. Both ten-epoch arms finished2810 finite optimizer updates, using2247 fitting studies and the same1156 disjoint weak-label development studies. Only fitting supervision changed:26954 cells from the pinned public llm_labels_v4_blend source. Accepted release labels and evaluation masks were unchanged. All20 immutable checkpoints are privately re-downloaded/size/SHA verified exactly once.

Independent replay: macro AUC0.8786705965 →0.8997840648, delta+0.0211134683. Whole-component paired conditional95% interval[0.0029817801,0.0405243740],2836 defined and1164 undefined draws out of4000, all12 targets retained. This is promising internal evidence, not a Kaggle or clinical score. Rare MCL/fracture positives (two each) limit macro interpretation. Medial Meniscus regressed0.8737466522→0.8590684633, delta−0.0146781888; paired95%[−0.0295708757,−0.0000974954]. No blanket label-quality or all-target superiority claim is justified.

Next: matched replication and source provenance review before promotion; do not select mixtures on this endpoint or unblind protected cohorts. Existing experimental-supervision deployment guards remain. Contrast H100 complete recovery has separately passed, enabling funded production preparation rather than an accuracy claim.

The first local analyzer invocation lacked PYTHONPATH and failed before remote execution; rerun with the project import path succeeded. Neither the trainer nor scientific inputs changed. Original collector/analyzer and exact output hashes are retained here.
