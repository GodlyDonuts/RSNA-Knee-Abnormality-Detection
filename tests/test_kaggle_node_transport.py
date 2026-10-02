from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kaggle import coordinator as c

SHA = "a" * 64


@pytest.fixture
def location(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "NODE_SCRATCH", tmp_path)
    root = tmp_path / ("rsna-kaggle-upload-" + SHA[:12])
    return SimpleNamespace(
        action="prepare-transport", node_local_upload_root=root
    ), tmp_path / "journal"


def test_owning_node_bound_transport_and_default_backward_compatibility(location):
    args, journal = location
    upload, binding = c.upload_location(args, journal, SHA)
    assert upload.parent == args.node_local_upload_root
    assert not journal.exists()
    args.action = "create-dataset"
    assert c.upload_location(args, journal, SHA, {"node_local_upload": binding}) == (
        upload,
        binding,
    )
    assert c.upload_location(SimpleNamespace(action="prepare-transport"), journal, SHA) == (
        journal / "upload",
        None,
    )


def test_wrong_node_replaced_directory_and_shared_fallback_rejected(location):
    args, journal = location
    _, binding = c.upload_location(args, journal, SHA)
    args.action = "create-dataset"
    with pytest.raises(ValueError, match="identity changed"):
        c.upload_location(args, journal, SHA, {"node_local_upload": binding | {"host": "other"}})
    with pytest.raises(ValueError, match="identity changed"):
        c.upload_location(args, journal, SHA, {"node_local_upload": binding | {"inode": -1}})
    with pytest.raises(ValueError, match="fall back"):
        c.upload_location(
            SimpleNamespace(action="submit"), journal, SHA, {"node_local_upload": binding}
        )


def test_permissions_existing_intent_and_symlink_rejected(location):
    args, journal = location
    c.upload_location(args, journal, SHA)
    with pytest.raises(FileExistsError):
        c.upload_location(args, journal, SHA)
    args.action = "create-dataset"
    args.node_local_upload_root.chmod(0o755)
    with pytest.raises(ValueError, match="permissions"):
        c.upload_location(args, journal, SHA, {})
    args.node_local_upload_root.rmdir()
    args.node_local_upload_root.symlink_to(journal)
    with pytest.raises(ValueError, match="exact private"):
        c.upload_location(args, journal, SHA, {})


def test_outside_scratch_or_different_checkpoint_rejected(location):
    args, journal = location
    with pytest.raises(ValueError, match="exact private"):
        c.upload_location(args, journal, "b" * 64)
    args.node_local_upload_root = Path("relative")
    with pytest.raises(ValueError, match="exact private"):
        c.upload_location(args, journal, SHA)
