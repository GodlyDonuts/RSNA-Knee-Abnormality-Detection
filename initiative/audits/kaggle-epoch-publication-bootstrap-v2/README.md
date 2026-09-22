# Default-Python coordinator bootstrap correction

This supplement supersedes the current coordinator/test source pins in
`kaggle-epoch-publication-v1`. The original staged source, mappings, initial and
final reviews, actual Kaggle structure proof, and root's earlier 937-test result
remain unchanged historical evidence. Root found the additional startup issue
before committing: a normal CLI import created a helper bytecode cache inside an
otherwise exact projection, then the membership gate rejected that new file.

The corrected coordinator disables bytecode writing before any local helper
execution and uses `runpy.run_path` on the explicit helper source. A pre-existing
cached helper is bypassed; unlisted caches are still rejected before loading the
workflow. Accepted `rsna` runtime code is unchanged.

`before-fix.json` records both clean-subprocess failures. The final tests use
fresh children without `-B` or ambient bytecode/path overrides and cover actual
CLI help, full projection loading, rejection of pre-existing cache files and
rejection of a valid-header cached helper without using it. The author's combined
Kaggle suite passed 85 tests; the independent supplement passed all 36 scoped
coordinator/publication tests. Root owns the later full 941-test run and final
archive refresh, commit and publication; those outcomes are not claimed here.

The source snapshots, focused receipt and independent supplement are preserved
with exact hashes. `closure.json` binds this supplement to the current integrated
files and maps the additional independent evidence to its original locations.
No remote writes, model actions, credential access by the reviewer or competition
submissions were involved in this correction.
