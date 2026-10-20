# Bounded real-MRI learning expansion

Original one-case GPU978218 passed exact complete recovery, but C showed an early
21.46 training exam-loss spike. This separate experiment lowers BOTH head learning
rates tenfold to3e-5, retains encoder3e-5, and gives three head-only updates before
full-encoder training. No change to the mathematical model, density coefficient,
accepted labels or masks. Context coverage grows from8 to24windows per mask while
all source descriptors/support priors remain complete. Twelve fitting studies
train, four disjoint fitting studies test engineering prediction reproducibility;
none is clinical validation or a competition candidate. Eight epochs/24updates,
effective4, both full/stop1/resume processes and exact complete-state/4x12CSV bytes.

The single finite coordinator waits for actual16-case cache acceptance before one
private CPU-to-CPU transfer; then dispatches at most one45minute one-H100/two-CPU
job on27. Existing pinned runtime/parent reused. New node-local4GiBcache/12GiB
engineering/16MiBcontrols are additional liabilities; all old liabilities and
20GiB reserve retained. TwoMiB/three new shared logs are separately admitted.
Original failed copy, completed recovery and private caches remain preserved.
Inspect actual receipts and intent before action; never retry uncertain mutation.
