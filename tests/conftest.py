"""Explicit test doubles for unrelated boundaries in narrow legacy unit tests.

Release/core tests exercise real evidence validators. These fixtures are opt-in;
production has no equivalent bypass or automatic fallback.
"""

from __future__ import annotations

import hashlib

import pytest


@pytest.fixture
def synthetic_release_binding():
    from rsna.semantic_supervision import digest

    def make(manifest, series, source_revision):
        binding = {
            "schema_version": 1,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
            "source_revision": source_revision,
            "release_receipt_sha256": "a" * 64,
            "semantic_registry_sha256": "b" * 64,
        }
        binding["binding_sha256"] = digest(binding)
        return binding

    return make


@pytest.fixture
def isolate_semantic_release(monkeypatch):
    """Isolate semantic approval when a CLI unit test targets another policy."""
    from rsna import release, semantic_supervision, weak_split

    audit = {
        "semantic_supervision": {"registry": {}, "ledger": {}, "n_cells": 0},
        "gold_uids": [],
        "technical_eligible_uids": [],
    }
    monkeypatch.setattr(release, "prepare_supervision", lambda *args: audit)
    monkeypatch.setattr(
        semantic_supervision, "apply_semantic_supervision", lambda studies, *args, **kw: studies
    )
    monkeypatch.setattr(release, "write_verified_release", lambda *args: {"unit_fixture": True})
    monkeypatch.setattr(weak_split, "repair_weak_split_coverage", lambda frame: frame)
    monkeypatch.setattr(weak_split, "split_repair_audit", lambda *args: {"unit_fixture": True})
