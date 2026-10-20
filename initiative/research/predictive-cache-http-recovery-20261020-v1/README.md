# Distinct missing-only HTTP recovery

**Actual progress at15:09:04UTC:** CPU981019 observed **1,081 raw/1,079 contexts of3,887**, jobs **980960/980969 RUNNING on evc33**, sole owner **91707**, source `d7326951eeb986f26eb21183d61c2b52476a7986`. Raw reused1,041 originals and added40 cases in433.34seconds; no completion-time estimate follows from this short interval. [Actual progress](actual-after-throttle-progress.json). Actual CPU 980959 admitted **1,041 raw/1,041 context studies**, 2,846 pending each, two retained buffers/one unfinished/zero orphans. [Actual admission](actual-mixed-prefix-admission.json). Current logs record four HTTP429 responses with seven-second retry delays and subsequent progress in both jobs; the historical failure status remains unknown. No full cache or predictive GPU fit exists yet. This owner was started once; do not repeat dispatch.

The previous raw job `980743` failed at its HTTP status assertion before opening the failed response body; context `980744` stopped after its producer failed. The old assertion did not retain the response status. A later successful header request does not establish the historical status.

This successor retains the original, alias-v3, and capacity-v1 generations. The read-only admission reconciles their sealed proofs, stat-bound caches, orphan outputs and retained source buffers. Complete source members are reused after the original byte/hash checks; missing members and partial suffixes alone may be fetched into the new owned source directory. Existing files are never overwritten.

`generation.py` derives the raw/context/admission programs from exact commit `370f2c598f4a25d92fc4764c899a1632ff50b675`. Numerical source stays identical to each predecessor snapshot. The only download change is a pinned `source_http.py`: one shared instance, at most eight concurrent responses, bounded retries before response-body admission, sanitized status/delay/attempt diagnostics, and no body or job mutation retries. `suffix.py` extends reconciliation with the exact failed predecessor jobs/intents and their unchanged ancestry.

The helper permits at most four pre-body attempts within a 180-second admission budget and uses finite per-request connect/read timeouts. This budget is checked around requests; it is not a hard interruption guarantee for DNS/redirect handling or an admitted body. Body reads preserve the original timeout and never restart after a partial read.

The new private roots are:

- `/tmp/rsna-predictive-full-raw-http-recovery-20261020-v1`
- `/tmp/rsna-predictive-full-context-http-recovery-20261020-v1`

Four study lanes retain the raw 320 GiB, context 96 GiB and combined source scratch 6 GiB limits. Every original generation counts toward paid physical bytes. The unchanged future reserve is 62 GiB + 594 MiB on node, and the combined shared obligation is 123 MiB / 376 files. No deletion credit is used.

## Admission and one owner

The launched controls and HTTP helper are frozen in source `d7326951`. Historical preparation called `owner.freeze(...)` with that exact commit, the original regression/prior-attempt/prefix receipts, the original capacity decision, and the accepted HTTP decision. Its generated `programs["admit"]` plus `payloads["raw"]` perform a separate read-only CPU admission. The resulting receipt must have status `actual_mixed_v4_http_prefix_read_only_admitted`, bind the generated suffix, HTTP helper and accepted decision hashes, and show both new roots absent. The original launch used CPU 980959’s actual accepted receipt; never repeat that admission or dispatch.

Historical owner interface, already used once; **do not rerun**:

```text
python3 owner.py --source-git <frozen-40-character-commit> --ssh-socket <existing-socket>
  --maximum-hours 30
  --regression-receipt <original-file> --regression-sha256 <sha>
  --prior-attempt-receipt <original-file> --prior-attempt-sha256 <sha>
  --prefix-admission-receipt <original-file> --prefix-admission-sha256 <sha>
  --capacity-decision <accepted-file> --capacity-decision-sha256 <sha>
  --http-decision <accepted-file> --http-decision-sha256 <sha>
  --mixed-prefix-admission <actual-new-file> --mixed-prefix-admission-sha256 <sha>
```

The journal is `artifacts/predictive-cache-http-recovery-20261020-v1`. Any existing owner/phase intent prevents a second dispatch. Raw uses 8 CPUs / 96 GiB and context 1 CPU / 12 GiB, each with a 20-hour limit and zero GPUs. Credentials enter only through the existing stdin transport. Successful actual collectors write `actual-owner-raw_collect.json` and `actual-owner-context_collect.json` here; they are not created by preparation.

Intent and terminal objects retain the original alias status family and all source/renderer/input fields. Their `continuation_chain` binds `980743` / `980744` and includes the preceding `979664` / `979665` as `ancestor_chain`. Both roles also bind `capacity_decision_sha256`, `mixed_prefix_admission_sha256`, `http_recovery_decision_sha256`, `source_http_sha256` and `scratch_cap_bytes=6442450944`.
