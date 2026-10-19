"""Keep the actual mount regression separate from immutable numerical code."""

from pathlib import Path

OLD = Path("initiative/research/S6-earlier-H100-execution-20261019-v1")
NEW = Path("initiative/research/S6-complete-target-execution-v2-20261019")


def test_all_control_sources_compile_and_do_not_reuse_failed_dispatch_stage():
    for p in NEW.glob("*.py.txt"):
        compile(p.read_text(), str(p), "exec")
    for name in ("gpu_entry.py.txt", "prepare_once.py.txt", "recovery_entry.py.txt"):
        text = (NEW / name).read_text()
        assert "recovery-v1" not in text and "recovery-v2" in text
    stage = (NEW / "recovery_prepare.py.txt").read_text()
    assert "new, meta = c.build(source, metadata)" in stage
    assert stage.index("new, meta = c.build(source, metadata)") < stage.index("G.mkdir")
    assert '"/kaggle/input": 9, "/kaggle/working": 5' in stage
    assert '"973776|FAILED|1:0"' in stage


def test_original_numerical_production_and_publication_modules_are_byte_identical():
    for name in (
        "compiler.py.txt",
        "epoch_verify.py.txt",
        "epoch_io.py.txt",
        "publish_epoch.py.txt",
        "target_compiler.py.txt",
    ):
        assert (NEW / name).read_bytes() == (OLD / name).read_bytes()
    for src, dst in [
        ("package.py.txt", "packaging_package.py.txt"),
        ("worker.py.txt", "packaging_worker.py.txt"),
    ]:
        assert (NEW / dst).read_bytes() == (
            Path("initiative/research/S6-fixed-final-packaging-20261019-v1") / src
        ).read_bytes()


def test_followers_use_actual_new_owner_handles_and_new_results_only():
    p = (NEW / "follow_production_once.py.txt").read_text()
    assert "S6-complete-target-execution-v2-20261019/recovery-owner-v2" in p
    assert "S6-earlier-H100-execution-20261019-v1/recovery-owner-v1" not in p
    for name in ("follow_packaging_once.py.txt", "follow_scoring_once.py.txt"):
        s = (NEW / name).read_text()
        assert "71311" not in s and "67269" not in s and "6500d1a8" not in s
        assert "int(sys.argv[1])" in s
        assert "S6-complete-target-execution-v2-20261019" in s
    scoring = (NEW / "follow_scoring_once.py.txt").read_text()
    assert 'D = ROOT / "initiative/research/S6-final-Kaggle-execution-20261019-v1"' in scoring
    assert "actual-packaging-owner.json" in scoring
