import numpy as np
import pytest

from scripts.newton.prepare_licensed_teacher_targets import write_targets


def inputs():
    return (
        ["synthetic1", "synthetic2"],
        np.full((2, 12), 0.5, np.float32),
        np.ones((2, 12), np.bool_),
        np.arange(48, dtype=np.float32).reshape(2, 2, 12),
    )


def test_compact_private_teacher_keeps_exact_views_rows_and_original_labels(tmp_path):
    uids, y, m, p = inputs()
    m[0, 0] = False
    y[0, 0] = np.nan
    output = tmp_path / "private.npz"
    receipt = write_targets(output, uids, y, m, p)
    assert (
        receipt["studies"] == 2 and receipt["bytes"] < 4096 and output.stat().st_mode & 0o077 == 0
    )
    with np.load(output, allow_pickle=False) as a:
        assert a["uids"].tolist() == uids
        assert np.array_equal(a["labels"], y, equal_nan=True) and np.array_equal(a["masks"], m)
        assert np.array_equal(a["parent_logits"], p)
    with pytest.raises(FileExistsError):
        write_targets(output, uids, y, m, p)


@pytest.mark.parametrize(
    "bad", ["duplicate", "nonfinite_teacher", "non_bool_mask", "unobserved_teacher_view"]
)
def test_incomplete_or_misaligned_source_teacher_bundle_is_rejected_before_write(tmp_path, bad):
    uids, y, m, p = inputs()
    if bad == "duplicate":
        uids[1] = uids[0]
    elif bad == "nonfinite_teacher":
        p[0, 1, 1] = np.nan
    elif bad == "non_bool_mask":
        m = m.astype(np.int8)
    else:
        p = p[:, :1]
    with pytest.raises(ValueError):
        write_targets(tmp_path / "private.npz", uids, y, m, p)
    assert not (tmp_path / "private.npz").exists()
