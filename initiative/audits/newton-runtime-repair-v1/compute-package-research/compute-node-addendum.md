# Compute-node package identity: correction to the candidate

Author: `/root/competition_contract`, 2026-09-22. This supplement preserves the original login-node research unchanged. It replaces only the package candidate for the measured compute nodes, not the repair procedure or accepted environment archive.

The dispatcher's read-only allocation 838977 measured both evc22 and intended GPU node evc37. Their `/usr/bin/python3.12` and actually mapped `/usr/lib64/libpython3.12.so.1.0` belong to Rocky `3.12.13-2.el8_10.x86_64`. The login-node receipt used `3.12.13-3.el8_10.x86_64`. Therefore the exact-build policy correctly rejected the first -3 candidate before extraction or compilation. This is observed package provenance drift; it does not prove that the -3 headers would intrinsically violate the ABI. Prefer the exact -2 package now that it is available.

The two original node receipts are `artifacts/newton-python-node-identity-v1/evc22.json` (SHA256 `ad1091d191555c89ed9b9c63b505ca8197a38bcecaf6ca08011c0cc92a92fd4f`) and `evc37.json` (SHA256 `7f931fcf728b5304c0afbea83aa686d94f2842553eae98a14d991823ce40edaf`). Both report the same configuration flags, SOABI `cpython-312-x86_64-linux-gnu`, 8-byte pointers and 16-byte PyObject. The mapped-library RPM owners provide stronger evidence than a version string alone. Binary content hashes remain a dispatcher execution binding.

The [official Rocky devel repository](https://download.rockylinux.org/pub/rocky/8.10/devel/x86_64/os/Packages/p/) retains this exact package:

- [python3.12-devel-3.12.13-2.el8_10.x86_64.rpm](https://download.rockylinux.org/pub/rocky/8.10/devel/x86_64/os/Packages/p/python3.12-devel-3.12.13-2.el8_10.x86_64.rpm)
- Size: **299,160 bytes**.
- SHA256: **`d822aa263492b730c391f19d7614ad05c9acdc33ba165964565548ffa54ea3e3`**.
- Exact dependencies in the signed-package candidate's repository record: Python and Python-libs **3.12.13-2.el8_10**.

This task verified the entire [official devel metadata](https://download.rockylinux.org/pub/rocky/8.10/devel/x86_64/os/repodata/repomd.xml) checksum chain, with exactly one matching package record:

| Object | Bytes | SHA256 |
|---|---:|---|
| repomd.xml | See catalog-head.json | `339cc3d01258d7c0bd6aba4c80619a94457ef5f0d929dc2d821ff83895b555ca` |
| primary.xml.gz | 103,702,027 | `81dff09b350250a1df040deaea829d469d5708ce5e67652a2ba4a5d290e1e3d6` |
| Expanded primary XML | 737,832,742 | `98bb008f92b49c4cd5e1b2fc0f0dcbcd3b2555f1751a003d7960103c78a75f0c` |

Only compact proof files were saved; the compressed metadata was read in memory and expanded through a streaming parser. The RPM itself was not downloaded in this research task. Repository checksum verification does not replace trusted-key RPM signature verification.

Proceed with the original repair plan's complete generated header extraction, private single-directory CPATH, allocated-node identity checks, fresh isolated Triton cache and exact installed builder/driver source binding. Change the pinned package URL, byte count, SHA256 and expected release to this -2 candidate. Keep the original -3 failure, research and accepted environment intact. A successful CPU extension and driver-helper compile/import still does not establish GPU backward or optimizer acceptance. The dispatcher owns execution; this research made no cluster changes or submissions.
