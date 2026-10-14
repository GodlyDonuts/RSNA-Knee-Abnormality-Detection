# Optional node-private release admission cache (engineering only)

Issue #51. Engineering evidence, not scientific evidence. Not a GPU throughput,
accuracy, recipe, roster, label, fold or source change. The default trainer still
runs the full original `load_verified_release` in every fresh process.

## Measured (CPU only, no MRI, no GPU)

Devin VM, Intel Xeon Platinum 8559C, 8 vCPU, Python 3.10 / pandas 2.3.3, warm
page cache, canonical `initiative/releases/report-supervision-v1/studies.parquet`
(4407 x 131) with artifact root = repository checkout:

| Measurement | Seconds |
| --- | --- |
| Full original `load_verified_release` (direct) | 63.5 |
| Same, inside cache producer under the read trace | 61.5 (producer total 63.1 wall) |
| Same under cProfile (overhead inflated) | 123.8 |
| Cached `consume` (re-hash all 1209 members/287 MB, stat, listings, frame) | 1.1-1.2 |
| `import rsna.release_cache` cold | 0.21 |

The original replay opened 1209 repository members (286,836,482 bytes) and
listed 6 directories. cProfile attribution: `prepare_supervision` 75.8 s
cumulative, `inspect_semantic_supervision` 63.6 s, `_evidence` 58.2 s,
`_validate_table` 37.0 s, `_fresh_table` 36.1 s, `repair_weak_split_coverage`
26.6 s, repeated `Path.resolve`/`realpath` about 26 s. This is pure-Python CPU work;
the H100 job's two allocated cores, shared with overlapping workers, may be slower.

Encoder initialization and dataset/cache initialization were NOT measured here:
this VM has no original MRI cache or official encoder weights, by design. The
startup cost reported in the issue is therefore not yet attributed.

## Plan for root (actual MRI/state measurement is root's responsibility)

1. On the actual node, run the existing PR27 profiler
   (`scripts/newton/profile_training_throughput.py`) unchanged except that its
   receipt now carries `startup_seconds` = `release_admission`, `folds`, `series`,
   `dataset_initialization`, `encoder_initialization`, `model_optimizer`
   (`time.perf_counter`, no RNG use). Use one original fresh process, no new data.
2. Only if `release_admission` is a material share of startup across the
   repeated full/stop/resume processes, produce one cache privately on that node:
   `python -m rsna.release_cache produce --studies ... --artifact-root ... --cache <node-private dir>`
   then pass `--release-admission-cache <dir>` to the trainer (or profiler).
   Never stage, commit, copy to Mac or HF: `frame.parquet` holds reports, labels
   and case identifiers.

Caching is useless (do not use it) when: release replay is small relative to
encoder/dataset initialization or the update phase; each process is long enough
that about 60 s is negligible; the source tree, code (any top-level `rsna/*.py`)
or runtime changes between processes (the cache is then refused as stale); the
node cannot keep a private 0700 directory; or the consumer's own re-hash
(about 287 MB, about 1 s here; slower on cold network storage) approaches the
replay cost.

## Guarantees

* Producer calls the original `load_verified_release`, records every file it
  opened and directory it listed via a Python audit hook, refuses unbound reads
  outside the release/source/library trees, links inside the source tree, writes,
  members modified after admission start, and more than 4096 members, 2 GiB of
  members, 64 listings, 64 MiB frame or 16 MiB manifest.
* Cache directory 0700 and exactly `admission.json` + `frame.parquet` at 0600,
  owned by the user, outside the source, release and code trees.
* Consumer re-reads every member (no-follow), requires identical
  dev/inode/mode/size/mtime/ctime and SHA-256, identical directory listings
  (missing/extra members), identical top-level `rsna` code and runtime versions,
  identical frame bytes = bound studies table, identical
  shape/columns/dtypes/index/values/roster, and recomputes the binding from the
  bound receipt. Any difference raises `AdmissionCacheRefused`.
* Fallback to the full original admission happens only with
  `--release-admission-fallback`, and is announced on stderr; a failing original
  admission still fails.
