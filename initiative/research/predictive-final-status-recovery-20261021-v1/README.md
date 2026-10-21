# Final controller status recovery — October 21

The original Newton consumer3332 completed Dataset creation and all six-object
independent download/size/SHA checks, then successfully compiled the exact final
qualification source. Its inherited response parser expected the earlier generic
status instead of `C100_final_offline_source_compiled_CPU_only`, so it stopped
before any notebook push. This is a controller failure, not a model/GPU result.
Original transport00005 returned0. No original is deleted or retried.

The explicit continuation retains source1dc078f7f5b6764d04231e4f7e2f80b275bc5bca,
model, serving, Dataset, remote root, canonical claim, quota and runtime policy.
Newton resumes at the first push only after read-only checks of the existing
private Dataset12762386v1, original full-byte proof, exact compiled source
fe364a3c8a9b7ea2d0edd0ca6678302255aed8f83e709c3254af6172d0fe60c6,
canonical dataset row and absence of any prior push/submission. It does not
repeat staging, export, upload, verification download or qualification compile.

Colab's still-waiting local consumer has the same parser defect. A separately
recorded stop of that waiting consumer admits a distinct continuation reading
the same original browser handoff. The actual Colab training cell and queued
publisher remain untouched. Its role-specific compile receipt is admitted;
unknown responses still fail. Explicit no-mutation capacity waits retain their
original behavior. The original submit response (`state=submitted`) is unchanged.

Twelve focused checks passed: both actual role status schemas and modes,
strict malformed-response rejection, original submission/wait handling,
removal only of already-completed Newton phases and read-only boundary syntax.
This does not establish GPU qualification, hidden robustness or accuracy.
Actual dispatch/outcome receipts are separate from this accepted implementation.
