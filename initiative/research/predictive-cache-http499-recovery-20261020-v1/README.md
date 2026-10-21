# Missing-only continuation after the actual HTTP499 failure

**Completed October21 around00:09UTC:** original raw **983422** and context **983435** both completed0:0 on evc33, in4:22:52 and4:22:54. All **3,887 original fitting studies** have exact accepted raw and FP32 context inputs. Original collection CPUs **988896/988907** checked all3,887 retained file/stat bindings without repeating pixel reads or bulk hashes. The raw cache totals232,498,651,306bytes; context67,005,012,716bytes. All historical generations and corrected source proofs remain. Sole cache owner **28954 finished normally** and must not be restarted. [Raw original collection](actual-owner-raw_collect.json); [context original collection](actual-owner-context_collect.json).

The existing all-evc27 fitting continuation33107 consumed these receipts. Source CPU988913 admitted all3,887 inputs and the exact original scientific source44cc559e; the single original evc33→evc27 transfer is underway through receiver988926/sender988929. This is input readiness, not a trained model or score. Training/deployment remain owned by33107/32378; no cache transfer or owner is duplicated.

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

The following constructs the command; it does **not** execute it. `PIN` must be the same reviewed commit used for the successful admission. The six historical receipt arguments are retained verbatim from the original launcher; the frozen reader independently verifies their original hashes. The original command has run once and its journal exists; do not execute it again. Historical fitting29171/deployment29172/observer29173 consumed this handoff and were subsequently stopped. Current fitting33107/deployment32378 use the separately accepted all-evc27 continuation; the completed cache owner must not be rerun.

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
