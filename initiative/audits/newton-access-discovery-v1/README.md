# Newton and Stokes access discovery

Measured on 2026-09-22 through the user's SSH account. Both servers authenticated
successfully; no VPN or MFA obstruction was encountered. Host keys were checked
with OpenSSH's normal known-hosts verification. Credentials are absent from this
audit. Subsequent authorized zero-GPU preparation and ongoing transfers are recorded below.

Newton's usable initial combination is account `skattel`, partition `normal`,
default QoS `normal`, and `gpu:nvidia_h100_pcie:2`. The scheduler lists 29 nodes of
that two-H100 type. One was idle in the snapshot, which is not a reservation.
The account's September report showed 0 of 2,000 GPU hours and 0 of 10,000 CPU
hours used. Initial requested resources remain two GPUs and four CPUs on one
node. A requested one-CPU probe received two hardware CPUs under site policy.
Cancellation has zero preemption grace and a 30-second TERM-to-KILL interval.

Both servers expose the same Lustre home storage. The actual user soft quota has
212.66 GiB remaining, but only 45,004 file slots. The accepted cache contains
48,772 files in 4,413 directories; it cannot be unpacked there. Group storage
also charges the user's quota. Slurm reports `TmpDisk=0` for all eligible H100
nodes: this is unconfigured metadata. Job 838365 measured 769,039,626,240 bytes free on evc22 node-local XFS /tmp; tiny write probes were removed. No official ARCC
node-local scratch policy was found in the checked documentation.

The authorized staging root `/home/sa305415/rsna-knee` and its `cache` and
`access-probe-v1` directories have effective permissions 0700 (Lustre may inherit setgid, yielding 02700). One existing public
4,414,765-byte release manifest was transferred, verified by SHA-256, and retained
at mode 0600. It took 0.669 seconds (6.30 MiB/s including setup), too small a
sample to establish sustained cache-transfer throughput. A first local attempt
failed before transfer because macOS openrsync rejected numeric chmod; the final
receipt records the corrected permission verification.

The ongoing full-cache stream target is
`/home/sa305415/rsna-knee/cache/cache256-v3-official-recovery-v1.tar`. Stream from
the existing local cache into a new private `.partial` file, verify inventory and
hashes, then publish atomically. Do not make an additional large local tar or
transfer the 569 GB raw-source collection for this experiment. The extraction helper checks node-local filesystem and capacity on every job. Archive the Linux environment as
well if its unpacked file count would exceed the remaining Lustre quota.

`module load python/python-3.11.4-gcc-12.2.0 && python3 --version` was verified to
report Python 3.11.4. Default Python is 3.6.8. CUDA modules are available, but no
allocated-node driver or GPU execution was inspected. A private local SSH
ControlMaster was established for coordinated transfers; its socket path is kept
only in ignored local artifacts, not in this portable audit.

See `summary.json`, the command-specific receipts, and `artifact-index.json`.
The latter hashes the JSON evidence. The cache remains archived on Lustre to avoid the measured file-quota constraint. Both full-cache and smaller pilot-cache streams are ongoing; their completion and hashes must come from their final receipts.

Approved DINOv3-L staging job **838408** completed successfully in 5m27s with zero GPUs. The exact 1,212,559,808-byte weight, config, and complete custom license were verified. The transient token file was deleted, and no partial remained. See `dinov3-stage-result.json`. GPU training has not been started by this agent.

Official references: [ARCC file-system policy](https://arcc.ist.ucf.edu/docs/data/files/)
and [ARCC Globus transfer guide](https://arcc.ist.ucf.edu/docs/data/globus/).
The documented Globus collection is **UCF ARCC**. The file-system policy states
that group-owned locations do not bypass a user's quota and that this storage
is not backed up.
