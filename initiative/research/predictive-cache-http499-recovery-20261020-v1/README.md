# Missing-only continuation after the actual HTTP499 failure

**Actual running continuation:** raw **983422**, context **983435**, sole cache owner **28954**, frozen source `986f32c176328f49d306ca93d3cd1948df50aff4`. CPU **983471 at 19:48:10 UTC** confirmed both jobs RUNNING on evc33 and **2,359 raw/2,356 contexts of 3,887**. The failed source has been passed; inputs remain incomplete and no full predictive model has trained. [Actual startup](actual-successor-startup.json).

Raw 981958 failed at 19:29:56 UTC and context 981960 at 19:30:30 UTC on October 20. The raw log records HTTP499 before admitting response bytes; the original bounded helper treated 499 as terminal. The local owner 10824 and both transports ended. [Actual terminal](actual-terminal.json); [accepted decision](decision.accepted.json).

Read-only CPU983321 found **2,349 completed raw and2,348 context studies of 3,887**, leaving1,538/1,539. The incomplete150-member source has57 complete-sized files and93 missing, with no partial or oversized files. All historical scratch occupies255,930,486 physical bytes. The collector checked metadata/stat proofs, not completed cache bodies or pixels; the subsequent exact prefix admission passed before dispatch. [Actual admission](actual-mixed-prefix-admission.json). [Inventory](actual-prefix-inventory.json).

This successor adds499 to the existing bounded prebody retry policy: at most4 attempts,180 seconds,8 shared requests. It preserves the original decoder, renderer, numerical modules, fitting roster and all five stopped generations. Both authoritative corrected studies1740/1850 must already be sealed with their exact original correction provenance. The fifth generation is verified against its effective two-file table; the first four retain the baseline table. No completed source is reacquired, no case is dropped, and there is no automatic restart after an uncertain operation.

New private roots are `/tmp/rsna-predictive-full-{raw,context}-http499-recovery-20261020-v1`; the distinct local journal is `artifacts/predictive-cache-http499-recovery-20261020-v1/`. The immediate parents are exact e9bdd124 raw981958/context 981960 intents. All older ancestry and correction envelopes remain bound. Both intents and final proofs add `http499_recovery_decision_sha256`, `previous_source_http_sha256` and the new `source_http_sha256` (`b15e3d56fb536fbb42ea9427ed57fb6c25db052bbcefd8e74139bd39c01c7fdb`).

Raw remains4 lanes/8CPUs/96GiB; context1CPU/12GiB; both CPU-only with20-hour limits. The finite local owner lasts at most30 hours. Existing320GiB raw,96GiB context,6GiB scratch and all other funding scopes remain unchanged, including123MiB/376 future shared files and the20GiB/5,000-file floors. No GPU or competition action is authorized by these controls.

## Preserved freeze and read-only admission interface

The actual frozen pin is `986f32c176328f49d306ca93d3cd1948df50aff4`. Admission and dispatch have already occurred; the interface below documents the original operation and must not be repeated. Any independently authorized future successor requires its own exact 40-character `PIN`. From this directory, the CPU admission payload is constructed without a token:

```python
from pathlib import Path
import hashlib
import owner

here = Path.cwd()
decision = (here / "decision.accepted.json").read_bytes()
failure = (here / "actual-terminal.json").read_bytes()
bundle = owner.freeze(
    PIN, decision_body=decision,
    decision_sha256=hashlib.sha256(decision).hexdigest(),
    failure_body=failure,
)
program = bundle["programs"]["admit"]
payload = bundle["payloads"]["raw"]
assert "token" not in payload
```

Run only that program through the original stdin transport on evc33 with1 CPU/12GiB/10 minutes, zero GPUs and `PYTHONDONTWRITEBYTECODE=1`. Preserve its original stdout/stderr and save the returned sanitized receipt as `actual-mixed-prefix-admission.json`. Required status: `actual_mixed_corrected_five_generation_HTTP499_prefix_read_only_admitted`. The earlier collector is not a replacement for this admission. Re-freeze with the exact receipt bytes and SHA as `mixed_body`/`mixed_sha256` before dispatch.

## Exact owner command

The following constructs the command; it does **not** execute it. `PIN` must be the same reviewed commit used for the successful admission. The six historical receipt arguments are retained verbatim from the original launcher; the frozen reader independently verifies their original hashes. The original command has run once and its journal exists; do not execute it again. Fitting 29171/deployment 29172/observer 29173 now consume its exact proofs through the independently bound 29128354 handoff.

```python
import hashlib
import json
import shlex
from pathlib import Path

repo = Path("/Users/sairamen/projects/RSNA-Knee-Abnormality-Detection")
here = repo / "initiative/research/predictive-cache-http499-recovery-20261020-v1"
old_command = json.loads((repo / "artifacts/predictive-cache-http-recovery-20261020-v1/launcher.json").read_text())["command"][2:]
old = dict(zip(old_command[::2], old_command[1::2]))
command = ["python3", str(here / "owner.py"), "--source-git", PIN,
           "--maximum-hours", "30"]
for name, digest_name in (("regression-receipt", "regression-sha256"),
                         ("prior-attempt-receipt", "prior-attempt-sha256"),
                         ("prefix-admission-receipt", "prefix-admission-sha256")):
    command += ["--" + name, old["--" + name],
                "--" + digest_name, old["--" + digest_name]]
for name, filename in (("http499-decision", "decision.accepted.json"),
                       ("failure-receipt", "actual-terminal.json"),
                       ("mixed-prefix-admission", "actual-mixed-prefix-admission.json")):
    path = here / filename
    command += ["--" + name, str(path), "--" + name + "-sha256",
                hashlib.sha256(path.read_bytes()).hexdigest()]
print(shlex.join(command))
```

An existing intent, ambiguous dispatch, changed original or failed proof stops the mutation. Observe and reconcile the original operation; do not rerun the owner. Five focused tests exercise frozen numerical parity, the old/new decision namespaces, corrected five-root prefix reuse, exact corrected raw/context preservation, and once-only transition behavior.
