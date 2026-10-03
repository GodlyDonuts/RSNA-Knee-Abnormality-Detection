# Actual-MRI engineering admission for reviewed Effusion supervision

The paired experiment asks whether recovering individually reviewed, previously
masked positive Effusion cells helps the same solo MRI classifier. The original
release, all other targets and outer evaluation masks remain immutable. This
admission cannot establish clinical accuracy or choose a recipe.

The separately pinned two-epoch engineering configs use the production native384
rank32 recipe and first16 fitting plus first16 disjoint outer studies. The fitting
prefix contains one actually changed reviewed cell; the complete2247/1156 rosters
and31-cell private proposal are validated before subsetting. Both arms decode
identical original images, geometry and sampling. Input acceptance requires the
candidate's actual observation tensors to expose exactly one additional positive.

`scripts/newton/launch_effusion_engineering.py` materializes immutable source and
the previously accepted environment on node-local disk. It uses original whole-
byte/cache stat proofs rather than copying or hashing the complete cache again.
A oneCPU/zeroGPU gate on the owning evc31 must actually complete before a distinct
oneH100/twoCPU recovery allocation. GPU admission additionally retains the original
allocated CUDA/header checks. No device isolation or live recipe is changed.

`scripts/newton/probe_effusion_engineering.py` calls the canonical trainer in fresh
full/stop-after1/resume processes for each arm. Acceptance requires complete decoded
model/frozen-base/AdamW/LR/counter/RNG equality, all numerical update boundaries,
and identical16x12 MRI-only CSV bytes within each arm. Both arms bind the exact
source, complete admission and experimental supervision receipt. Twelve local
negative checks reject meaningful state, trajectory and prediction differences.
Competition inference continues to reject these diagnostic/experimental weights.

Remote controls use a distinct8MiB/40inode cap, conditional on actual original
native-all terminal/allten private preservation and fresh retirement of40 unused
future metadata obligations. Original artifacts and all other obligations remain;
no deletion credit. Node-local engineering checkpoints have3GiB cap, runtime7GiB,
and20GiB free reserve after obligations. This does not fund a production screen or
its future publication. Actual dispatch/result/funding receipts supersede this
prepared description; never retry an uncertain original dispatch.

A real CLI defect was caught before staging: the overlay binding hashed a fold
argument that production passes as a directory. It now hashes the verified sibling
fold.csv for every supported fold path. Canonical synthetic full/stop1/resume tests
exercise both enabled arms with directory and CSV forms.

Actual CPU gate869076 completed0:0 in178seconds on evc31. Its complete-roster
and actual MRI tensor admission passed:86/87 observed engineering cells expose
one reviewed positive with exactly identical image inputs. The original receipt
is retained as inputs-accepted.json. Fresh resource/source review then dispatched
distinct H100 recovery869077 once. Its terminal numerical acceptance is not yet
observed; never call this a clinical result or a production screen. The shared
funding retirement is now actually admitted, not merely a proposed plan.
