# First real original-source paired cache shard

The one-CPU, zero-GPU runner builds paired 256 and 384 caches directly from
pinned original DICOMs, then privately publishes and fully downloads/verifies
both archives and their four provenance objects in the same finite allocation.
It reclaims only its exact owned node-local scratch after successful private
preservation; a failure retains source/cache proofs and prevents automatic retry.
Credentials travel through stdin, never shared files or process arguments.

Newton staging checked all scientific Python sources against C9, the six accepted
input files against their immutable hashes, both runtime helper hashes, archive
presence/size, available quota and absence of an existing shard attempt. The
staged first shard contains 43 complete studies, 250 series and 8,484 slices:
5,272,478,310 original source bytes. All large outputs remain node-local; the
funded Lustre reserve after remaining obligations and staging is 21.945 GiB.
The accepted original reference archive will be checked for all selected members,
not freshly full-hashed as an 83 GB archive. Accepted labels, masks and folds stay
unchanged; no gold or confirmation labels are read.

Fifteen runner tests cover real paired synthetic-DICOM preservation, incomplete
or mismatched private receipts, local changes after preservation, missing remote
objects, unsafe cleanup and bound-checkout external imports. This is engineering
acceptance, not a model or clinical score. The initial payload is staged at
`/home/sa305415/rsna-knee/native-full-shard000-c9-v1`; no Slurm dispatch occurred
in this source/staging milestone. Full-cohort cache assembly and native production
training remain future work, conditional on actual source-shard acceptance.

The initial finite CPU dispatch became Slurm866372 and terminated FAILED2:0
before run-start, environment extraction, original-source reads or HF mutation.
Read-only replay identifies the path guard: `/home/sa305415/rsna-knee` resolves
to `/lustre/fs1/home/sa305415/rsna-knee`, so comparing a resolved control with
an unresolved base rejected the legitimate cluster path. Original intent and
failure evidence are retained; the job/control are never restarted or overwritten.
The corrected runner resolves both sides while refusing the base root, outside
paths and symlinked control leaves. Two regression cases prove that behavior.
A distinct v2 control is staged after exact terminal review, not dispatched yet.
