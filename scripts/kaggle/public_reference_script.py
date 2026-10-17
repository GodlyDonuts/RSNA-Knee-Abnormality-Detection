"""Preserve an admitted notebook's code cells in a batch-script launcher.

This changes execution packaging only. The original source/weights/credits
remain authoritative; compiled scripts and third-party source stay private.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rsna.schema import TARGETS

NOTEBOOK_SHA = "fac1723c32de2d27f59e10ebca7102327bb33346117a3fec3a58937ecf2d3a0e"
CELLS_SHA = "300b34417bc1bc91018a4bb50fa9440403e904ab7abcc7236e432e61932df56f"
R6_MANIFEST = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"
PUBLIC_CSV = "5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def code_cells(text, *, notebook_sha=NOTEBOOK_SHA, cells_sha=CELLS_SHA):
    if digest(text) != notebook_sha:
        raise ValueError("Exact admitted original scored notebook required")
    obj = json.loads(text)
    if obj.get("nbformat") != 4 or not isinstance(obj.get("cells"), list):
        raise ValueError("Complete original notebook structure required")
    cells, markdown = [], []
    for cell in obj["cells"]:
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(x, str) for x in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("Original source strings required")
        if cell.get("cell_type") == "code":
            cells.append(source)
        elif cell.get("cell_type") == "markdown":
            markdown.append(source)
        else:
            raise ValueError("Original code/attribution cells required")
    h = digest(json.dumps(cells, ensure_ascii=False, separators=(",", ":")))
    if len(cells) != 8 or h != cells_sha or not markdown:
        raise ValueError("All eight byte-identical original code cells and credits required")
    return cells, "\n\n".join(markdown)


def run_cells(cells, runner):
    """Stop on an actual cell error; never silently skip a failed stage."""
    completed = []
    for i, cell in enumerate(cells):
        result = runner(cell, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(f"Original reference cell {i} failed") from error
        completed.append(i)
    return completed


def build(text, *, notebook_sha=NOTEBOOK_SHA, cells_sha=CELLS_SHA):
    cells, credits = code_cells(text, notebook_sha=notebook_sha, cells_sha=cells_sha)
    # Embed the tested error/ordering contract without importing project code
    # into Kaggle. No cell text or model/preprocessing parameter is rewritten.
    import ast

    module = Path(__file__).read_text()
    node = next(
        n
        for n in ast.parse(module).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_cells"
    )
    runner_source = ast.get_source_segment(module, node)
    if runner_source is None:
        raise ValueError("Complete cell execution contract required")
    driver = (
        "# Private batch launcher for the exact public Helios v6 reference.\n"
        "# Displayed upstream AUC is not a score for this compiled candidate.\n"
        "import hashlib,json,os,time\nfrom pathlib import Path\n"
        "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
        f"ORIGINAL_CREDITS={credits!r}\n"
        f"ORIGINAL_NOTEBOOK_SHA={notebook_sha!r}\nORIGINAL_CELLS_SHA={cells_sha!r}\n"
        f"_CELLS={cells!r}\n"
        + runner_source
        + "\n\n"
        + RUNTIME.replace("R6_MANIFEST_VALUE", repr(R6_MANIFEST))
        .replace("TARGETS_VALUE", repr(list(TARGETS)))
        .replace("PUBLIC_CSV_VALUE", repr(PUBLIC_CSV))
    )
    compile(driver, "exact_reference_batch_script", "exec")
    return driver, {
        "original_notebook_sha256": notebook_sha,
        "all8code_cells_sha256": cells_sha,
        "original_cells_byte_unchanged": True,
        "original_attribution_markdown_retained": True,
        "script_sha256": digest(driver),
        "script_bytes": len(driver.encode()),
        "requires_original_R6_version": 1,
        "required_R6_manifest_sha256": R6_MANIFEST,
        "prior_interactive_public_CSV_sha256": PUBLIC_CSV,
        "no_model_pixel_label_or_probability_changes": True,
        "actual_batch_GPU_or_hidden_result": False,
    }


RUNTIME = r"""
def _original_manifest():
    candidates = [Path('/kaggle/input/rsna-knee-r6-frozen-fe055406/manifest.json'),
        Path('/kaggle/input/datasets/heliosli/rsna-knee-r6-frozen-fe055406/manifest.json')]
    found = [p for p in candidates if p.is_file()]
    assert found and all(p.samefile(found[0]) for p in found), 'One exact original R6 Dataset v1 is required'
    p = found[0]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == R6_MANIFEST_VALUE
    manifest = json.loads(p.read_text())
    assert manifest['bundle_id'] == 'fe055406fad6a12f'
    assert all((p.parent / n).is_file() for n in manifest['sha256'])
    return {'bundle_id': manifest['bundle_id'], 'manifest_sha256': R6_MANIFEST_VALUE,
            'required_dataset_version': 1}


def main():
    import numpy as np
    import pandas as pd
    import torch
    from IPython.core.interactiveshell import InteractiveShell
    assert hashlib.sha256(json.dumps(_CELLS, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == ORIGINAL_CELLS_SHA
    assert torch.cuda.device_count() == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    manifest = _original_manifest()
    start = time.monotonic()
    print(json.dumps({'exact_reference_batch_started': True, 'GPUs': [torch.cuda.get_device_name(i) for i in range(2)],
        'original_code_cells': len(_CELLS), 'cells_sha256': ORIGINAL_CELLS_SHA, **manifest}), flush=True)
    shell = InteractiveShell.instance()
    completed = run_cells(_CELLS, shell.run_cell)
    roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
             Path('/kaggle/input/rsna-knee-abnormality-detection')]
    roots = [p for p in roots if (p / 'test.csv').is_file()]
    assert len(roots) == 1
    ids = pd.read_csv(roots[0] / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    assert len(ids) == len(set(ids))
    labels = TARGETS_VALUE
    proofs = {}
    for name in ('submission.csv', '_public_stack.csv', '_own.csv', 'ours_sub.csv', 'raptor_sub.csv', 'raptor224_sub.csv'):
        p = Path('/kaggle/working') / name
        frame = pd.read_csv(p, dtype={'StudyInstanceUID': str})
        assert list(frame.columns) == ['StudyInstanceUID', *labels]
        assert frame.StudyInstanceUID.tolist() == ids
        values = frame[labels].to_numpy()
        assert values.shape == (len(ids), 12) and np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        proofs[name] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    r6 = json.loads(Path('/kaggle/working/submission_receipt.json').read_text())
    assert r6['status'] == 'PASS' and r6['rows'] == r6['scored'] == len(ids)
    assert r6['fallback_studies'] == 0 and r6['bundle_id'] == manifest['bundle_id']
    assert sorted(r6['gpus']) == ['Tesla T4', 'Tesla T4'] and r6['worker_exit_codes'] == {'0': 0, '1': 0}
    blend = json.loads(Path('/kaggle/working/blend_receipt.json').read_text())
    assert blend['rows'] == len(ids) and blend['mode'] == 'blend'
    assert all(blend[k] == 0 for k in ('raptor_failed', 'raptor384_failed', 'raptor224_failed', 'ours_failed', 'both_failed'))
    assert blend['weights'] == {'raptor': 0.5, 'ours': 0.5, 'raptor_per_class': {'Medial OA': 0.5,
        'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "Baker's": 0.3}, 'a224': 0.35}
    out = {'status': 'actual_exact8cell_reference_script_MRI_and_final_reader_accepted',
        'all8code_cells_sha256': ORIGINAL_CELLS_SHA, 'original_notebook_sha256': ORIGINAL_NOTEBOOK_SHA,
        'completed_original_cells': completed, 'rows': len(ids), 'columns': 13,
        'R6_fallbacks': 0, 'R6_bundle_id': manifest['bundle_id'], 'full_final_reader_output_exists': True,
        'small_output_proofs': proofs, 'elapsed_seconds': time.monotonic() - start,
        'original_public_example_exact_CSV_match': proofs['submission.csv']['sha256'] == PUBLIC_CSV_VALUE,
        'competition_entry': False, 'clinical_or_hidden_score_claim': False}
    Path('/kaggle/working/exact8cell-script-accepted.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
"""
