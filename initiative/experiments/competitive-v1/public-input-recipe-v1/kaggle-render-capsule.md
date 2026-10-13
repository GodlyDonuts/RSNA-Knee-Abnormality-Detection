# Private input capsule for the Kaggle CPU public renderer (prepared, NOT funded or dispatched)

Issue #33. Packaging and planning only. It adds no renderer and changes neither
`scripts/kaggle/render_public_source.py` nor any validator, source, mask, fold, release,
config or live/protected job (899766/908454 included). The tool makes no Kaggle, Hugging Face
or Slurm call and downloads nothing. It packages no pixels, cache, model weights,
credentials, Git history or unrelated datasets, and reads or analyzes no labels beyond what
the unchanged `load_verified_release`/`load_folds` already read. Synthetic tests are
engineering evidence only; nothing here admits runtime, the full cohort, parity or accuracy.
The eight-study probe's zero plane differences do not imply full-cohort equality.

Entry: `scripts/kaggle/package_render_capsule.py` (stdlib, except `trace`, which runs the
existing validators). Tests: `tests/test_package_render_capsule.py`.

## What the capsule holds

One uncompressed, deterministic PAX tar (`capsule.tar`: sorted names, mtime 0, uid/gid 0,
mode 0444, regular files only) and `capsule-manifest.json`. For every member the manifest
records the archive path, tree, source path, bytes, SHA256 and Git blob, plus code/artifact
commits, the closure SHA, bindings, root bounds, the archive SHA and the verifier's own SHA.
There are three trees:

| tree | content | Git binding |
|---|---|---|
| `code/` | renderer `CODE_FILES`, the wrapper, this verifier and their static `rsna`/`tools`/`scripts` import closure; `public-triplet-config.json`; `fold0-overlay.json`; the nine `initiative/source/inventory/part-00{0..8}.parquet`; `source_correction_plan.json` + `hf-source-correction-receipt.json`; `review.json`; the Pilkwang v15 `source.notebook.json` (155,668 B, `b32a9155…`); input receipt `actual-CPU-accepted.json`; `actual-plan-admitted.json` | exact blobs of `HEAD` (dirty or untracked files refuse) |
| `artifact/` | exactly the files the unchanged `load_verified_release` + `load_folds` read under the receipt's artifact root (`trace`), plus `studies.parquet`, `folds/fold.csv`, `folds/folds.json` and the **accepted** `series_sampling.csv` | exact blobs of a root-declared `--artifact-commit` |
| `inputs/` | `member-manifest.json` | SHA equal to `actual-plan-admitted.json` `original_member_manifest_sha256` (`496949a3…`) |

The renderer resolves inventory, config, overlay and correction paths relative to its own
location, so they live in `code/`. The release, fold and series inputs go in `artifact/` at
their original relative paths below the artifact root. The mounted competition
`train_series.csv` is **not** the reviewed release `series_sampling.csv`
(`14eb050d…` = release `series_sha256`). The capsule keeps `--accepted-series` separate. The
mount is still bound independently by the renderer at render time. `load_verified_release`
itself also hash-binds its own copy of the original `initiative/source/train.csv`/
`train_series.csv`, so the trace includes those two files.

Local dry run (this checkout's `HEAD`, code root = artifact root, a synthetic member
manifest, so it is not the authoritative capsule):

- trace replay: 4,407 release studies, 3,904 fold rows, binding `e2ecb283…`;
- 1,280 members, 410,144,684 B (391.1 MiB): `code/` 68 files, 122.8 MB, of which the
  inventory is about 116 MiB; `artifact/` 288.6 MB;
- the archive is 411,412,480 B, under the **proposed** 512 MiB cap;
- after `verify` extraction, a fresh process with `cwd=/tmp` and only `code/` on
  `sys.path` reran `load_verified_release`/`load_folds` from `artifact/` alone and
  reproduced binding `e2ecb283…`, 4,407 and 3,904. The renderer was not run, because
  torch/pydicom/cv2 are absent here.

## Root workflow

Use the authoritative retained artifact root and the paths in
`initiative/audits/public-label-source-inputs-v1/actual-CPU-accepted.json`
(`trainer_inputs`). `trace` refuses any other paths.

```
ART=/home/sa305415/rsna-knee/code-stopfix-1f486765
python scripts/kaggle/package_render_capsule.py trace --out <private>/closure.json \
  --artifact-root $ART \
  --studies $ART/initiative/releases/report-supervision-v1/studies.parquet \
  --folds $ART/initiative/experiments/competitive-v1/folds \
  --accepted-series $ART/initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv
```

Record the printed `closure_sha256`. Then pack from a clean checkout of the exact code
commit. `--artifact-commit` is the Git commit that the retained artifact root's files
belong to (`git -C $ART rev-parse HEAD`, if that root is a clean checkout). All three bounds
are required and are root-funded, not defaults:

```
python scripts/kaggle/package_render_capsule.py pack --artifact-root $ART \
  --artifact-commit <commit> --closure <private>/closure.json --closure-sha256 <sha> \
  --member-manifest <retained original member manifest> --out <new private dir> \
  --max-bytes 536870912 --max-files <n> --min-free-bytes <reserve>
```

`pack` refuses all of the following: an existing output, output inside a source, a file
that differs from its Git blob or trace, links, path escape, duplicates (including
case-folded ones), file/directory collisions, foreign trees, source bytes changing during
the write, and exceeded byte, file or free-space bounds. `reconcile` (same arguments, plus
`--manifest-sha256`, no bounds) is read-only. It recomputes the plan and requires the
existing manifest and archive to equal it.

## Kaggle mapping

Attach `capsule.tar` and `capsule-manifest.json` as **one private Dataset**, e.g.
`/kaggle/input/<capsule-slug>/`. The notebook is private, with `enable_gpu: false`,
`enable_internet: false` and `competition_sources: ["rsna-knee-abnormality-detection"]`,
and it makes no submission. Its first step runs the verifier straight from the archive, with
stdlib only and nothing extracted yet. Root passes the manifest SHA recorded at pack time:

```
C=/kaggle/input/<capsule-slug>; X=/kaggle/working/capsule   # X must not exist
tar -xOf $C/capsule.tar code/scripts/kaggle/package_render_capsule.py > /tmp/verify.py
python /tmp/verify.py verify --manifest $C/capsule-manifest.json --manifest-sha256 <sha> \
  --archive $C/capsule.tar --extract-to $X \
  --max-bytes <funded> --max-files <funded> --min-free-bytes <reserve>
```

`verify` checks the following:

- its own SHA against the manifest;
- the archive bytes and SHA;
- every header: regular files only, no links, safe relative names, exact name, size and
  order.

Next it extracts into the new directory within the bounds, writes members read-only and
re-hashes every member and the whole tree. Only then does it print the renderer arguments.
If Kaggle auto-extracts the archive, use `verify --tree <dir>` (read-only, same SHA checks)
in its place. A failure means no read and no render. The arguments are fixed relative to `X`:

| renderer argument | capsule path |
|---|---|
| script | `$X/code/scripts/kaggle/render_public_source.py` |
| `--source` | `$X/code/initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json` |
| `--member-manifest` | `$X/inputs/member-manifest.json` |
| `--studies` | `$X/artifact/initiative/releases/report-supervision-v1/studies.parquet` |
| `--artifact-root` | `$X/artifact` |
| `--folds` | `$X/artifact/initiative/experiments/competitive-v1/folds` |
| `--accepted-series` | `$X/artifact/initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv` |
| `--input-root` | `/kaggle/input` (competition mount, bound independently) |

The probe and shard commands, budget keys, acceptance and merge are unchanged. See
`kaggle-cpu-source-render.md`:

```
python $X/code/scripts/kaggle/render_public_source.py --mode probe --input-root /kaggle/input \
  --private-out /kaggle/working/private/probe --scratch /tmp/rsna-source-transport \
  --budget <budget.json> --budget-sha256 <sha> <the seven capsule paths above>
```

Runtime expectations come from the renderer: Kaggle CPU with no GPU, and the
`runtime_sha256` that root records after an **actual** probe (root is testing the exact
original Torch 2.14 / NumPy 2.2.1 runtime in kernel 138542169). The budget `code_sha256`
must equal `code_sha256()` of the extracted `code/` files.

Shards follow the same pattern. After actual probe acceptance, each CPU shard session
mounts the same Dataset, reruns `verify` into a new directory and runs
`--mode shard --cohort-manifest … --cohort-manifest-sha256 … --shard-index i`. The shard
size comes from the probe's `plan_shards` and fits the funded wall and output limits.
Each shard writes only its immutable private shard output (under the 20 GB working disk).
Root downloads the outputs privately and merges them with `merge_shards`. Root alone
verifies storage, creates the single private Dataset and dispatches sessions.

## Not claimed

- No capsule was built from the authoritative artifact root.
- No Dataset or notebook was created, and no probe or shard was run.
- No Kaggle runtime, parity, full-cohort equality or accuracy is claimed.
- The 512 MiB cap is proposed only.
