# Derivative endpoint-source reuse evidence

The existing 48,772 endpoint records remain source-compatible with corrected
revision `0b5e410288232f570c69ba4f44723356e6ae6ab7`. Every selected path, StudyUID,
SeriesUID, SOPUID, whole-source SHA256, byte size and storage mode is identical
in the complete baseline and corrected inventories. Both inventories contain
819,635 slices, 24,386 series and 4,410 studies; exactly two unselected source
records change. Neither corrected SOP was an endpoint selected by the survey.

The historical survey, manifest, summary and original code remain immutable and
bound to `52f995f4568e36bfa5fbbaf69a89af41226b4f5d`. Their hashes, all nine original
inventory part hashes, the accepted correction plan and private-publication
receipt, corrected inventory, and archived survey gzip content were verified.
The full lexical first/last selection was independently rebuilt from both full
inventories and matched to the recorded selection and all final endpoint rows.

`verification/reuse-provenance.json` binds every input and the exact result.
`verification/endpoint-source-equivalence.jsonl` provides one row per endpoint,
including its original line number, canonical record digest, unchanged source
row, lexical endpoint role and original prefix-verification scope.

This is derivative source-membership evidence, **not a new survey**. There were
zero network requests, zero new header parses and zero new pixel decodes. The
recorded prefixes are generally unavailable for rehashing here. Of the original
records, 48,770 retain partial-prefix/transport/UID scope and two retain their
recorded complete-small-file SHA256 scope. Middle-slice headers, pixel content,
anatomical correspondence and clinical labels are outside this check.

## Reproduce

From the recovered repository, use its environment and new output directories:

```bash
python ../annotation-staging/endpoint-source-reuse-v1/verify_reuse.py \
  --work-root .. \
  --output ../annotation-staging/endpoint-source-reuse-v1/verification-repeat

python ../annotation-staging/endpoint-source-reuse-v1/crosscheck_queue.py \
  --work-root .. \
  --output ../annotation-staging/endpoint-source-reuse-v1/queue-repeat
```

The scripts are standalone local evidence tools. They neither import nor call
the network survey or source recovery runtime. `verify_reuse.py` uses the
existing inventory validation/fingerprint helpers only. Both scripts reject an
existing output directory. The queue cross-check binds the original completed
`verification/reuse-provenance.json`; a repeat does not rewrite that evidence.

## Remaining technical queue

`queue/candidate-crosscheck.json` preserves all 129 original candidates and
cross-references completed technical audit hashes. `queue/candidate-summary.md`
lists the remaining priorities. There are 111 candidates without an initial
technical assessment: 31 stronger mixed-side/repeated-protocol cases and 80
other tag cases. Ten plane questions and four bilateral native confirmations
have completed evidence; four further cases have source/header assessments
with unresolved correspondence limits. Plane closure does not certify report
linkage or final duplicate selection.

The six remaining repeated-protocol cases are 001:18, 021:03, 048:12, 064:14,
075:14 and 086:13. Begin with original source and all-series headers. Their
metadata does not itself establish bilateral anatomy or justify automatic
pixel downloads, laterality assignment or label changes. No clinical
annotation files or report text were inspected for this cross-check.
