import hashlib

import pytest
from test_kaggle_shared_wheelhouse import proof

from scripts.kaggle import workflow


@pytest.mark.parametrize(
    "attachments", [["owner/model", "owner/wheels"], ["owner/wheels", "owner/model"]]
)
def test_exact_kernel_accepts_attachment_order_without_changing_identities(attachments):
    source = "exact code"
    operation = {
        "kernel_slug": "owner/notebook",
        "dataset_slug": "owner/model",
        "shared_wheelhouse": proof(),
        "runtime_image": "exact-runtime",
        "kernel_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }

    class Remote:
        def kernel(self, action, slug, version):
            assert slug == "owner/notebook" and version == 1
            if action == "GetKernelSessionStatus":
                return {"status": "RUNNING"}
            return {
                "metadata": {
                    "isPrivate": True,
                    "enableInternet": False,
                    "machineShape": "NvidiaTeslaT4",
                    "competitionDataSources": [workflow.COMPETITION],
                    "datasetDataSources": attachments,
                    "dockerImage": "exact-runtime",
                },
                "blob": {"source": source},
            }

    assert workflow.check_kernel(Remote(), {"operation": operation}, 1) == {"status": "RUNNING"}


@pytest.mark.parametrize(
    "attachments",
    [
        ["owner/model"],
        ["owner/model", "owner/other"],
        ["owner/model", "owner/wheels", "owner/wheels"],
    ],
)
def test_changed_or_duplicate_attachment_is_rejected(attachments):
    source = "exact code"
    operation = {
        "kernel_slug": "owner/notebook",
        "dataset_slug": "owner/model",
        "shared_wheelhouse": proof(),
        "kernel_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }

    class Remote:
        def kernel(self, action, slug, version):
            return {
                "metadata": {
                    "isPrivate": True,
                    "enableInternet": False,
                    "machineShape": "NvidiaTeslaT4",
                    "competitionDataSources": [workflow.COMPETITION],
                    "datasetDataSources": attachments,
                },
                "blob": {"source": source},
            }

    with pytest.raises(ValueError, match="exact prepared"):
        workflow.check_kernel(Remote(), {"operation": operation}, 1)
