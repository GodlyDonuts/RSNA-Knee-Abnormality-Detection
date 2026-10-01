"""Keep original-source scratch until private bytes are verified; never over-clean."""

import copy
import json
import os
import subprocess
import sys

import pytest
import test_native_cache_publisher as publisher_tests

from scripts.newton import run_native_cache_shard as runner


@pytest.fixture
def preserved(tmp_path):
    bundle = publisher_tests.bundle.__wrapped__(tmp_path)
    receipt = publisher_tests.publish(bundle, tmp_path / "journal", publisher_tests.Api())
    return bundle, receipt


def require(preserved):
    (root, shard, execution), receipt = preserved
    runner.require_private_preservation(
        root, shard["shard"], shard["selection_sha256"], receipt, execution
    )


def test_real_paired_payloads_require_complete_private_receipt(preserved):
    require(preserved)
    root = preserved[0][0]
    with (root / "cache-side-384.tar").open("ab") as stream:
        stream.write(b"changed after preservation")
    with pytest.raises(ValueError, match="local_preserved_payload_changed"):
        require(preserved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "commit_sent_but_not_verified"),
        ("repo_id", "someone/public"),
        ("shard", 99),
        ("execution_provenance", {"job_id": "different"}),
        ("revision", "main"),
        ("prefix", "experiments/some-other-selection/shard-000"),
    ],
)
def test_incomplete_or_different_preservation_keeps_cache(preserved, field, value):
    record = copy.deepcopy(preserved[1])
    record[field] = value
    with pytest.raises(ValueError, match="complete_private_preservation_required"):
        require((preserved[0], record))
    assert (preserved[0][0] / "cache-side-384.tar").is_file()


def test_missing_remote_object_and_unbound_generated_bytes_refused(preserved):
    record = copy.deepcopy(preserved[1])
    del record["files"]["attribution.json"]
    with pytest.raises(ValueError, match="complete_private_preservation_required"):
        require((preserved[0], record))
    record = copy.deepcopy(preserved[1])
    record["files"]["manifest.json"]["sha256"] = "unverified"
    with pytest.raises(ValueError, match="complete_remote_byte_bindings_required"):
        require((preserved[0], record))


def scratch(tmp_path):
    root = tmp_path / "rsna-native-shard-123-owned"
    root.mkdir()
    owner = runner.owned_marker("123", "a" * 64, os.getuid())
    (root / ".native-cache-owner.json").write_text(json.dumps(owner))
    (root / "keep-on-failure").write_text("important")
    return root, owner


@pytest.mark.parametrize(
    "change", ["shared_device", "wrong_marker", "linked_marker", "wrong_job", "linked_root"]
)
def test_cleanup_refuses_unowned_shared_or_linked_paths(tmp_path, change):
    root, owner = scratch(tmp_path)
    home_device = -1
    if change == "shared_device":
        home_device = root.stat().st_dev
    elif change == "wrong_marker":
        (root / ".native-cache-owner.json").write_text("{}")
    elif change == "linked_marker":
        (root / ".native-cache-owner.json").unlink()
        target = tmp_path / "outside-marker"
        target.write_text(json.dumps(owner))
        (root / ".native-cache-owner.json").symlink_to(target)
    elif change == "wrong_job":
        owner = {**owner, "job_id": "124"}
    elif change == "linked_root":
        target = root
        root = tmp_path / "rsna-native-shard-123-linked"
        root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="owned_node_local_scratch_required"):
        runner.remove_owned(root, owner, home_device)
    assert (root / "keep-on-failure").read_text() == "important"


def test_cleanup_removes_only_exact_owned_root(tmp_path):
    root, owner = scratch(tmp_path)
    neighbour = tmp_path / "unrelated-checkpoints"
    neighbour.mkdir()
    (neighbour / "checkpoint").write_text("retain")
    runner.remove_owned(root, owner, -1)
    assert not root.exists()
    assert (neighbour / "checkpoint").read_text() == "retain"


def test_external_entry_uses_bound_checkout_and_forwards_arguments(tmp_path):
    checkout = tmp_path / "science"
    checkout.mkdir()
    (checkout / "bound_science.py").write_text("VALUE = 'pinned-source'\n")
    entry = tmp_path / "external-entry.py"
    entry.write_text(
        "import bound_science,json,sys;print(json.dumps([bound_science.VALUE,sys.argv[1:]]))\n"
    )
    result = subprocess.check_output(
        runner.child_command(sys.executable, checkout, entry, ["--shard", 0]),
        cwd=checkout,
        text=True,
    )
    assert json.loads(result) == ["pinned-source", ["--shard", "0"]]
    assert not list(checkout.rglob("*.pyc"))


def test_private_control_allows_cluster_home_mount_mapping(tmp_path):
    physical = tmp_path / "lustre" / "home" / "rsna"
    physical.mkdir(parents=True)
    mapped = tmp_path / "home"
    mapped.symlink_to(physical, target_is_directory=True)
    control = mapped / "first-shard-v2"
    control.mkdir()
    runner.require_control(control, mapped)


def test_private_control_refuses_root_outside_and_linked_leaf(tmp_path):
    base = tmp_path / "private"
    base.mkdir()
    outside = tmp_path / "other-project"
    outside.mkdir()
    linked = base / "linked-control"
    linked.symlink_to(outside, target_is_directory=True)
    for control in (base, outside, linked):
        with pytest.raises(ValueError, match="private_cluster_control_required"):
            runner.require_control(control, base)
