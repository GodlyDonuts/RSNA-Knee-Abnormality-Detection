"""Explicit Kaggle backup provenance in the original C100 final contract."""

import hashlib

HELPER = """
def storage_identity(receipt):
    import re
    if receipt.get("storage_provider") is None:
        return re.fullmatch("[0-9a-f]{40}",receipt.get("revision","")) is not None
    b=receipt.get("private_backup",{})
    epoch=receipt.get("epoch")
    files=receipt.get("files",{})
    return (receipt.get("storage_provider")=="Kaggle" and epoch in (4,5,6)
        and b.get("private") is True and b.get("version")==1
        and type(b.get("id")) is int and b["id"]>0
        and b.get("slug","").startswith("godlydonuts/rsna-c100-backup456-")
        and b.get("downloaded_all_bytes") is True
        and b.get("original_members_verified_before_upload") is True
        and 0 < b.get("archive",{}).get("bytes",0) < 2*2**30
        and re.fullmatch("[0-9a-f]{64}",b.get("archive",{}).get("sha256","")) is not None
        and receipt.get("revision")==f"Kaggle:{b['id']}:v1:epoch-{epoch:03d}"
        and b.get("members")=={f"epoch-{epoch:03d}/"+n:p for n,p in files.items()}
        and receipt.get("source_git")=="8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
        and receipt.get("GPU_job")=="995351")
"""
exec(HELPER)


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Original storage contract anchor differs")
    return source.replace(old, new)


def extend(source):
    if (
        hashlib.sha256(source.encode()).hexdigest()
        != "881a88f511e5bbdaf134d955a75508ff0db9636eee00c66254f8a2d455fea0c9"
    ):
        raise ValueError("Exact original C100 model contract required")
    addition = f"\nSTORAGE_HELPER_SOURCE={HELPER!r}\nexec(STORAGE_HELPER_SOURCE)\n"
    source = replace(source, "class Contract:", addition + "\nclass Contract:")
    source = replace(
        source,
        '        self._base = types.ModuleType("isolated_C100_final_contract")',
        """        changed = changed.replace('re.fullmatch("[0-9a-f]{40}", publication["revision"]) is not None', 'storage_identity(publication)')
        changed = STORAGE_HELPER_SOURCE + changed
        self._base = types.ModuleType("isolated_C100_final_contract")""",
    )
    source = replace(
        source, 'and hexadecimal(receipt.get("revision"), 40),', "and storage_identity(receipt),"
    )
    return source
