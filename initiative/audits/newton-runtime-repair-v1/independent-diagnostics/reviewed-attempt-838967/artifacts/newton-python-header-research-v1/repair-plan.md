# Private Python header supplement: research handoff

Author: `/root/competition_contract`. Scope: local evidence review and public primary-source research, 2026-09-22. No cluster commands, package installation, environment changes, or GPU execution were performed by this research task.

The preferred repair is a private extraction of the exact Rocky development package matching Newton's installed Python. Keep the accepted environment archive and model execution backend unchanged. The package identity and official catalog checksum are verified; its downloaded payload, signature, extracted headers, and successful compilation still require the dispatcher's separate execution proof.

## Observed failure and system identity

Pilot 838921 reached the first loss backward after loading the official model. Its eager `torch._native.ops.bmm_outer_product` path caused Triton to compile `driver.c`; GCC could not find `Python.h` through `/usr/include/python3.12`. This is not evidence that explicit `torch.compile` was enabled. Encoder freezing does not eliminate backward operations in the hierarchical head.

The preserved login-node receipt, `artifacts/newton-python-headers-v1/system-python-readonly.json`, identifies Rocky 8.10, `/usr/bin/python3.12`, Python and Python-libs `3.12.13-3.el8_10.x86_64`, GCC `8.5.0-28.el8_10`, SOABI `cpython-312-x86_64-linux-gnu`, `Py_DEBUG=0`, empty `ABIFLAGS`, `Py_ENABLE_SHARED=1`, and 8-byte pointers. Both Python.h and pyconfig.h are absent. This receipt is a login-node observation; repeat it inside the actual CPU allocation before accepting a supplement. Bind interpreter, resolved libpython, compiler, installed Triton build.py/driver.c, and sysconfig bytes in that execution receipt.

## Exact official candidate

| Field | Verified value |
|---|---|
| NEVRA | `python3.12-devel-0:3.12.13-3.el8_10.x86_64` |
| Package bytes | 299,268 |
| Package SHA256 | `0b2b188bf7c920e92ff0f3952ad43100dce29d565dc2c27adea7b4d0dda34dd2` |
| repomd.xml SHA256 | `b75b096e295f367be4dc4604147777c2e0c26d1e7a0d87540ad30614656f8caf` |
| Compressed primary XML SHA256 | `68a865f0e7f6eaefef33b6089f15784fc6b2c83f8345f8cad90bebf8154a4c88` |
| Expanded primary XML SHA256 | `7b4991eea6dc713ad2cb57e43c718d6c502279e82954636763ddadb11eaa7097` |

Download the [exact RPM from Rocky](https://download.rockylinux.org/pub/rocky/8.10/AppStream/x86_64/os/Packages/p/python3.12-devel-3.12.13-3.el8_10.x86_64.rpm). The [official repository metadata](https://download.rockylinux.org/pub/rocky/8.10/AppStream/x86_64/os/repodata/repomd.xml) binds the primary XML and its package record, which requires the exact installed Python and Python-libs release. `catalog-proof.json`, `exact-package-element.xml`, and `verify_catalog.py` preserve the independently verified full compressed and expanded metadata checksum chain. Metadata checksum verification is not an RPM signature check. The complete package has not been downloaded by this research task.

## Minimal preparation and acceptance

1. Download once into a private empty staging directory. Verify byte count, SHA256, exact NEVRA and dependencies; require `rpmkeys --checksig --verbose` success using the system's already trusted Rocky key and record the key identity. Do not install the RPM or execute its scripts. If signature verification or target-node identity fails, stop and retain the evidence.
2. Inspect the signed payload inventory, then extract the complete Python include tree privately. Preserve generated `pyconfig.h`, any architecture-specific pyconfig wrappers, nested headers and required relative relationships. Reject escaping paths or links. Record each extracted regular file's size and SHA256. Actual package inventory has not yet been inspected here; do not assume that copying Python.h alone is sufficient. [CPython's sysconfig documentation](https://docs.python.org/3.12/library/sysconfig.html) explains the build configuration and platform include information required by C extensions.
3. For the child compilation process, set `CC=/usr/bin/gcc` and `CPATH` to the one verified absolute private `usr/include/python3.12` directory. Do not append an empty path component or inherit unknown include overrides. The [GCC documentation](https://gcc.gnu.org/onlinedocs/gcc/Environment-Variables.html) specifies that CPATH supplements explicit include paths, while an empty component searches the current directory. Keep the existing interpreter, compiler, glibc and driver libraries; no new libpython or LD_LIBRARY_PATH override is indicated by the recorded compile command.
4. Under the exact unpacked accepted environment, compile and import a small C extension reporting compile-time `PY_VERSION`, runtime `Py_GetVersion()`, pointer size, PyObject size and debug flags. Compare with the target interpreter's sysconfig and runtime facts. Record compiler argv, return status, header manifest, source and resulting shared-object hashes.
5. Use a **fresh isolated TRITON_CACHE_DIR**. Compile and import the exact installed NVIDIA driver.c through the installed `triton.runtime.build.compile_module_from_file`, preserving the backend's actual include directories and libcuda.so.1 linkage. Record the source files, resolved driver library, compiler argv and compiled shared-object hash. Inspect the installed module initializer before importing and do not select an active device or invoke device APIs in this CPU proof. If this CPU node lacks the real driver library, report that limitation; do not substitute a stub and claim complete acceptance.
6. The dispatcher's later authorized GPU gate must execute the K=1 outer-product BMM path and actual hierarchical head/masked-loss backward, verify finite gradients and an optimizer update, then rerun the pilot/resume acceptance. A CPU extension import or model forward is not that GPU proof.

The [Triton v3.8.0 build source](https://raw.githubusercontent.com/triton-lang/triton/v3.8.0/python/triton/runtime/build.py) exposes the compile/import helper and cache key construction; environment header contents are not directly part of that key. Therefore a warm cache is insufficient evidence of a header repair. Its [NVIDIA backend](https://raw.githubusercontent.com/triton-lang/triton/v3.8.0/third_party/nvidia/backend/driver.py) supplies the driver.c includes and library arguments. These public tag sources explain the mechanism; the actual installed wheel source hashes remain authoritative for execution. The recorded wheel may differ in source line layout.

## Fallback assessed, not applied

The [PyTorch v2.14.0 source](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/_native/common_utils.py) implements `TORCH_DISABLE_NATIVE_JIT=1`; [override registration](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/_native/triton_utils.py) consults it. It must be set before importing Torch because checking and registration are cached. This is a source-supported internal backend switch, not a promise to disable every JIT subsystem or a stable public API. It avoids registering native DSL overrides, including the [outer-product BMM specialization](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/_native/ops/bmm_outer_product/triton_impl.py), while retaining the operator contract through fallback execution.

Changing that backend can change numerical rounding, performance and reproducibility. If the header repair is impossible, treat this as a new bound runtime configuration requiring gradient, optimizer, resume and inference acceptance. It is not applied or recommended ahead of the matching-header repair. No private dispatcher deregistration or model rewrite is proposed.

## Kaggle implications

The accepted T4 probe establishes its specific tested CUDA operations and small model forward only. It did not exercise this full head backward path. Newton's Rocky pyconfig must not be reused blindly on Kaggle: a matching Python version string alone does not prove an identical interpreter build, distro or ABI configuration. A future authorized Kaggle diagnostic should inspect its actual headers/compiler/sysconfig and exercise the relevant model operations with a fresh cache. Production inference also needs its own complete checkpoint acceptance; absence of backward does not prove that every forward operator avoids compilation. No Kaggle mutation or submission was made for this research.

## Handoff scope

The dispatcher is authorized separately to stage and test the supplement. This document supplies a candidate and acceptance criteria; it does not certify a completed repair. Preserve the failed pilot, accepted environment and package archives unchanged. Bind the supplement's eventual digest into each launcher receipt and job-local cache identity without rewriting those historical artifacts.
