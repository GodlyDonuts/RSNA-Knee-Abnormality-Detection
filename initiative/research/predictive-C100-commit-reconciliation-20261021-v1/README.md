# C100 original epoch4 commit reconciliation

CPU998781 failed after uploading the model blob and before returning a commit.
Original owner76338 and dependent86015 exited; GPU995351 remains healthy.
Read-only CPU998822 confirms the private repo is still at epoch3 parent
1ac9dfab79b1de09ba11fb184767293ad96f0ec2, epoch4 path absent, original intent
and manifest present, and no commit/verified receipt. This resolves the uncertain
commit boundary; it does not establish the reason for the original400 error.

Root accepts one explicit metadata-only completion under the original shared
publication lock, exact unchanged parent and freshly rechecked path absence.
It references the already-uploaded SHA-addressed model blob and sends only four
small original files. No model upload, deletion, billing/security change or
training restart. A rejected or uncertain request stops, preserving its original
HTTP response; there is no automatic retry. Success requires independent full
download/size/SHA of all five objects before original verified state is written.
Fresh516GiB other-node obligations plus20GiB reserve and shared123MiB/370files
plus20GiB/5000reserve are checked. New small node-only control fits the retained
publication envelope; no new shared files. Two focused tests passed. Actual
execution/acceptance must be recorded separately.
