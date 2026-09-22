# Scheduled submission workflow

The account is entered in the competition. Authenticated read-only checks found
five currently available submissions and no submissions on September 22 at the
recorded query time; this is not a standing guarantee of remaining allowance.
The workflow rechecks the server before each mutation, retains one account ledger
and exact checkpoint/notebook versions, and blocks uncertain duplicate actions.

The actual private environment probe used no competition data and was not a
competition submission. It confirmed Python 3.12, T4 GPUs and driver 580.159.04.
The preinstalled torch 2.10 environment does not itself validate the planned
torch 2.14 offline bundle. The separate real offline-wheelhouse T4 probe remains
required. Production notebook requests and checks an explicit T4 machine shape.

The read-only progress command links epoch, run, checkpoint, notebook version,
submission ID, status and public score. It optionally joins the last training
update loss at the exact epoch boundary; that is not an epoch-average loss.
Pending scores remain null. The user requested this trajectory while training
continues; it is not permission to tune labels or consume reserved/gold targets.
