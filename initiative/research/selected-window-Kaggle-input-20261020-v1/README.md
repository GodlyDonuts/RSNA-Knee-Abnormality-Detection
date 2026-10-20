# Private selected-window integrity input for Kaggle

The original full-cohort CPU pass has verified 3,887 fitting studies, both views
and all 94 original windows. Its 23,384,320-byte private SHA table permits the
selected-only reader to check every returned window without rendering the unused
ones. This is an input-speed opportunity, not an accuracy result.

This one-time transport sends only the existing private integrity array and its
aggregate provenance receipt to a dedicated **private** Kaggle Dataset. It does
not send pixels, reports, labels, UIDs or weights, or repeat the original pixel
transfer. The signatures remain private and are not copied to Git/HF/Mac. The
existing Dataset publisher is unchanged: an explicitly bounded ZIP, durable
canonical intent, one typed final CreateDataset RPC, and independent whole-byte
download/size/SHA verification of every member.

Fresh funding requires all previous shared liabilities plus 20GiB/5,000 files
and a new 32KiB ledger allowance. There are zero new permanent shared files;
the existing transient canonical-claim allowance is preserved. A separate32MiB
node-local control/ZIP cap preserves the conservative20+87+12+24GiB reserve.
No deletion or retired-liability credit is taken. Credentials travel only through
stdin to a one-CPU/zero-GPU owning-node allocation.

This is not a GPU diagnostic or full fit. A new immutable Kaggle wrapper must
copy the mounted manifest to an owner-only local file, bind the original full
cohort/record proof, and verify actual selected tensors/RNG in its runtime before
using the fast reader. Never change the healthy H100 or T4 numerical payloads.
