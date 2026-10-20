# Full-data generative-routing worker

Prepared October20; original matched pilot **986011 is now COMPLETED0:0**.
Its current G100 recipe is **rejected for full training before the deadline**
after substantially worse matched fitting loss and gradient behavior. See the
[actual result and decision](../predictive-matched-pilot-results-20261020-v1/README.md).
This adds the full 3,887-study/six-epoch G100 worker and its distinct epoch
publisher. **No full G100 admission, funding, dispatch or model exists.**

The mechanism is the same one under test in 986011: classification gradients
do not pass through the Bayes factor, while both diagnosis-conditioned density
graphs continue learning from the proper density objective. Its coefficient is
100. Forward loss values and inference probabilities remain unchanged for a
fixed model. This is a learning intervention, not a probability adjustment.

`derive.py` builds from the exact original full-data source `44cc559e` and the
previously prepared C100 source at `473a9a94`. It reuses the exact loss adapter
and routing module in pilot 986011, preserves all original numerical modules,
the complete MRI input roster, 2,916-update schedule, seed, initial state and
six immutable epoch snapshots. No live source is edited. The publisher uses a
distinct `predictive-G100-full3887-<source12>` prefix and includes G100 provenance.

`contract.py` requires the **actual original** pilot to finish 0:0, both
200-update histories through peak learning rate, common initialization and
complete G100 full/stop/resume equality. A separate decision must review those
diagnostics, bind the full-data recipe and protect the 3,360-second primary
Kaggle allocation. The four engineering predictions cannot select the recipe.

Full fitting would use the already admitted evc27 inputs without another MRI
copy. The distinct CPU admission/coordinator is now implemented. Actual source freeze,
fresh storage funding, the completed pilot and deadline review remain required
before execution. The
existing six-GiB **pilot** scope does not fund a twelve-GiB full run. Do not
reuse C100's admission, output, checkpoint prefix or reserved capacity by
assumption. Primary B/C/C0 and the completed contrast experiment are unchanged.

Local tests cover exact preservation of the original encoder/input/fit code,
the full worker's real positional loss interface and gradients, distinct
publication provenance, helper loading, and rejection of incomplete or stale
pilot evidence. These are engineering checks, not actual GPU acceptance or
accuracy evidence. The full input preparation and primary B/C/C0 comparison remain the
immediate execution priorities. This prepared route remains unexecuted.

## Single coordinator

`coordinator.py` exposes explicit `status`, `admit`, `dispatch`, `publish` and
`preserve` phases. `controls.py` derives operational code from immutable C100
controls at473a9a94, replacing only the G100 identities, actual matched-pilot
contract, explicit decision checks and separate funding. No existing remote
control or original source is edited. Both the original all-evc27 controller
and continuation33107 remain bound.

The new budget preserves the **entire C100 twelve-GiB full-run reservation plus
sixteen-MiB controls**, its six-GiB pilot, the G100 pilot, contrast, descriptors
and all three primary fits. No existing optional scope is reassigned. Actual
free space is checked again before admission/dispatch; no storage funding has
yet been accepted for G100. New outputs and logs remain node-local.

The optional job requests one H100/two CPUs/eight hours on ordinary two-GPU
evc27. An `after` dependency on all three actual primary jobs requires each to
have started before G100 becomes eligible. It does not restart or cancel them.
Dispatch ends October21 20:00UTC; scheduler finish deadline is October22
04:00UTC. Preserve every completed epoch once through the existing canonical
private publication lock, with independent download/size/SHA checks. Existing
intents or ambiguous submissions require read-only reconciliation.

Thirty focused tests passed, including the numerical routing tests, real
full-worker loss interface, strict actual-pilot prerequisites, exact source
reuse, separate resource arithmetic and all-three-primary dependencies. These
results do not constitute a GPU run, clinical gain or permission to substitute
a diagnostic checkpoint into a competition notebook.

Actual read-only Newton check at22:50UTC succeeded from source
`e6e81f052ecadd7c7089f6750464379843860bb4`. It confirmed original pilot986011
PENDING, complete portable inputs absent, and no full-G100 admission or
dispatch. The check wrote no remote files and submitted no GPU jobs.
[Exact status receipt](actual-status-read.json). It is operational connectivity
evidence, not model acceptance.

## Final inference path

[serving.py](serving.py) derives a standalone G100-specific compiler/contract
from pinned C100 serving code. It reuses the exact original MRI decoder,
architecture, complete learned-state loading, two-T4 worker and precision/CSV
checks. G100 changes training gradients only; no inference rewrite is needed.
The contract requires the actual complete600-update matched pilot, six verified
full-data epochs, exact gradient-routing provenance and its own trained T4
qualification. Pilot checkpoints and C100 acceptance cannot substitute.

Fourteen focused tests passed, covering original-worker and module preservation,
standalone runtime execution, buffer loading, partial/changed pilot rejection,
full-run identity and qualification mismatch. This is source readiness only;
no actual G100 final, production notebook, account allocation or entry exists.
