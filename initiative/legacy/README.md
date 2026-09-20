# Historical artifacts: do not use as corrected training labels

The six `rsna_{pseudo,custom,manual}_labels{,_soft}.csv.gz` files are lossless
copies of the historical HF exports at source revision
`52f995f4568e36bfa5fbbaf69a89af41226b4f5d`. Before compression, each was compared
byte-for-byte with its pinned Git blob. `gzip -dc file.csv.gz` recovers it.
The `manual` filename is misleading: that export came from automated
rule/NLI labeling, not the current individual-reading/independent-review process.
Four targets have no non-gold positives in its hard export. Earlier calibrated
exports used official gold and cannot support independent gold evaluation.

`train_targets_*.csv.gz` and `report_nli_*.csv.gz` preserve available intermediate
outputs from the earlier labeling workspace. `score_nli_sentence.py` is the
available historical scoring script. Their original creation environments are
not the current package environment, and their quality is not certified.

`rsna_dataset.py`, `train_example.py`, and `hf-dataset-card-at-source.md` preserve
the former HF loader/example/documentation. The archived card contains
superseded claims; the current README and `docs/decisions.md` take precedence.
All four legacy image-row Parquet exports are also archived here, with hashes
verified against the pinned LFS objects. Their source references and sizes are
in the external artifact catalog.

`upload_missing_git.py` and `upload_git_watchdog.py` preserve the operative
historical transfer implementation. They assume the original Linux VM paths and
may change remote Git/LFS state when executed. They are reference artifacts,
not the recommended current data-download command. Runtime transfer checkpoints
and logs can contain signed URLs and are not copied verbatim.

`unifiedtheory/` preserves an existing local RSNA DINO/ALiBi research-code
snapshot, with its original README and requirements. It was already on the VM;
its upstream commit and claimed benchmark/result provenance have not been
verified in this initiative. Do not treat its README's historical scores or
smoke-test assertions as validation of the current dataset. It is retained so
future sessions can inspect the earlier report teacher, model, and inference
experiments rather than losing them with the VM.

These files are excluded from formatting/lint rewrites to preserve source bytes
and historical provenance. Active code and tests live in `rsna/` and `tests/`.
