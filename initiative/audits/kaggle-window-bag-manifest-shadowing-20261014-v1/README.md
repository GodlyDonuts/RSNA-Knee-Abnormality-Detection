# Original T4 bootstrap failure and correction

The authenticated original notebook Logs page was inspected read-only on October 14.
Notebook `godlydonuts/rsna-window-bag-two-t4-6bd780969358-v1`, version 1,
session 358243923, failed after 214 seconds on two T4s with zero output bytes.
Its source SHA is 6bd780969358895f2eff031f78522cf943085f5c4ed73489e1a5e0db397e1921.

At 205.7 seconds, the terminal traceback entered `run`, while constructing the
window-bag command, and raised `ValueError: Bundled encoder bytes/path differ
from the pinned source` in `encoder_arguments`. The subprocess command had not
launched. Earlier sitecustomize/wrapt warnings did not terminate bootstrap.

The injected window-bag branch assigned its numerical code manifest to the same
local `manifest` variable that held the mounted weight bundle's manifest. It then
passed those code records to the encoder file verifier. The correction names the
code manifest `science_manifest`, preserving the original bundle manifest for
encoder binding. Weight hashes, path containment, sizes and symlink checks remain
unchanged. No numerical trainer, inputs, encoder or scientific recipe changed.

The regression executes the actual bootstrap branch with a synthetic frozen-base
bundle and a distinct code manifest. It checks successful encoder runtime binding
and command construction, and reconstructs the original shadowing bug to require
the same exception before any subprocess call. Capsule extraction is mocked in
this test; it does not claim MRI, GPU or archive acceptance.

The original notebook remains terminal and immutable; it was not repushed or
submitted to the competition. A corrected distinct notebook still requires
fresh state/quota/ledger review and actual GPU execution. Newton jobs 899766 and
908454 were authoritatively RUNNING during this inspection. No training restarted.

Validation: the project QA environment ran the window-bag and delta-deployment
test files together: **75 passed, 3 skipped in 63.35 seconds**. Ruff and diff
whitespace checks passed. Initial system-Python attempts encountered unrelated
installed `scripts` namespace/dependency conflicts; the successful run explicitly
bound the repository's namespace and used `artifacts/qa-env/bin/python`.
