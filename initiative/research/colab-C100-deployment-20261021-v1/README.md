# Deploy the independent Colab final without another weight upload

Prepared October21 while the original A100 fit is running. This is executable,
locally tested integration; no deployment job, Dataset admission, notebook or
competition entry has been dispatched by it.

The final checkpoint will already be in the private epoch-006 v1 Dataset created
by the sole Colab writer. This route reuses that exact Dataset and keeps the
independent seed20261021 lineage. It never presents Colab as Newton job995351.
The original two-T4 decoder/model/CSV worker stays byte identical.

`run_once.py` exposes explicit `stage`, `prepare`, `adopt`, `compile`, `push`,
`status`, `submit`, and `outcome` phases. There is no polling owner or retry loop.
The source must be committed and match the supplied commit exactly. All phases
use one CPU/zero GPUs on evc27 and the existing stdin credential transport.

- `prepare` validates actual complete six-epoch/2,916-update evidence and prepares
  only small private metadata/source. It creates no checkpoint ZIP or new Dataset.
- `adopt` acquires the canonical Newton account claim and verifies the unchanged
  original Colab namespace lease. It checks all seven actual private v1 Dataset
  identities, readiness and exact file listings. It downloads the final model
  and small source/binding/terminal bytes once onto private deployment storage,
  comparing every download with the original Colab publication proof. The six
  already-verified optimizer archives are not downloaded again. Original whole-
  byte proofs remain explicitly attributed to Colab, not relabeled as fresh
  Newton downloads. The seven original publication receipts are retained in
  the canonical admission row.
- Subsequent notebook and submission phases retain the original account logic,
  daily allowance, immutable source/version checks and uncertainty handling.
  The additional3,360-second primary reservation retires only after the exact
  canonical B/C/C0 entries have been accepted. No paid credits are enabled.

The token-free configuration requires `owning_node: evc27`, a unique root
beginning `/tmp/rsna-predictive-final-deployment-20261020-v1-Colab-C100-`,
`Colab_evidence` in the [compiler's](../colab-C100-final-serving-20261021-v1/README.md)
format excluding `license_bytes`, and the existing private `license_path` and
`parent_source_path`. Original evidence must come from the actual completed run.
Fresh `funding` and `runtime_plan` follow the original deployment policy:
2GiB node-local,1MiB/eight shared files, all other liabilities retained, and
20GiB/5,000-file reserves. This implementation has not allocated that new scope.
Use the actual original two-T4 probe, fresh account allowance and this final
candidate's own trained qualification. Existing public example measurements or
another branch's qualification cannot substitute.

Eleven integration tests passed. They execute the staged compiler and CPU
preparation, exercise canonical existing-Dataset adoption without any create/
upload call, and reject changed metadata, a changed lease, an incomplete fit,
duplicate admission and a partial-download retry. The original submission and
budget code remains in the generated capsule. Ruff and CLI help passed.
The21 prior serving-contract tests also remain the compiler's existing coverage.

Remaining work is actual: collect the completed Colab terminal and all seven
publication proofs, fund/stage this exact route, run the trained two-T4 check,
then submit once if its runtime and fresh allowance permit. No new score or
accuracy gain is established by these software checks.
