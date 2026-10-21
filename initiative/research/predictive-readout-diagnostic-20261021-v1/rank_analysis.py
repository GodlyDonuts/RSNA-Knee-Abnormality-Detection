"""Read existing private diagnostic arrays once; aggregate ranking movement only."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path('/tmp/rsna-predictive-readout-diagnostic-20261021-v1')
EXPECTED_RECEIPT = 'f4f891a0bf638aab4c2f7fde7d6b07aca480fbd6ed309e18a59c2c7b721d681f'

def rank(x):
    order = np.argsort(x, kind='stable')
    sorted_x = x[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    out = np.empty(len(x), dtype='float64')
    for lo, hi in zip(starts, ends):
        out[order[lo:hi]] = (lo + hi - 1) / 2
    return out

def summarize(context, full, bf):
    assert context.shape == full.shape == bf.shape == (128, 12)
    assert all(np.isfinite(a).all() for a in (context, full, bf))
    results = []
    for j in range(12):
        p, f, b = context[:, j], full[:, j], bf[:, j]
        rp, rf = rank(p), rank(f)
        spread = float(np.std(b))
        rho = float(np.corrcoef(rp, rf)[0, 1]) if np.std(rp) and np.std(rf) else None
        i, k = np.triu_indices(len(p), k=1)
        direction_p, direction_f = np.sign(p[i] - p[k]), np.sign(f[i] - f[k])
        non_tied = (direction_p != 0) & (direction_f != 0)
        results.append(dict(target_index=j, context_full_spearman=rho,
            comparable_pairs=int(non_tied.sum()),
            fraction_pair_orders_reversed=float(np.mean(direction_p[non_tied] != direction_f[non_tied])) if non_tied.any() else None,
            mean_logBF=float(np.mean(b)), std_logBF=spread,
            density_mean_shift_fraction=float(np.mean(b)**2 / np.mean(b**2)) if np.any(b) else None,
            full_below_context_fraction=float(np.mean(f < p))))
    return dict(targets=results, studies=128, fitting_only=True, independent_accuracy_claim=False)

def main():
    assert os.uname().nodename == 'evc27'
    raw = (ROOT/'complete.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RECEIPT
    receipt = json.loads(raw)
    assert receipt['GPU_job'] == '1002912'
    p = ROOT/'readout.private.npz'
    b = p.read_bytes()
    assert len(b) == receipt['private_arrays_bytes'] and hashlib.sha256(b).hexdigest() == receipt['private_arrays_sha256']
    import io
    with np.load(io.BytesIO(b), allow_pickle=False) as a:
        result = summarize(a['context'], a['full'], a['logBF'])
    print(json.dumps(dict(observer_job=os.environ['SLURM_JOB_ID'], original_GPU_job='1002912',
        original_receipt_sha256=EXPECTED_RECEIPT, result=result,
        original_model_unchanged=True, new_model_forwards=0, new_private_outputs=0), allow_nan=False))

if __name__ == '__main__': main()
