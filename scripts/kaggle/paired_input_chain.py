"""Two private offline CPU inputs for an existing immutable small ZIP.

No network, credentials or mutation at compilation. Chaining avoids both an
uncertain Dataset retry and a source larger than900kB. Payloads stay private.
"""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
import zlib

LIMIT = 2 * 1024**2
SOURCE_LIMIT = 900_000
NAMES = {
    "corrections.private.npz",
    "construction-receipt.json",
    "source-permission.txt",
    "provenance.json",
    "bundle-manifest.json",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(archive, files):
    if not isinstance(archive, bytes) or not 0 < len(archive) < LIMIT or set(files) != NAMES:
        raise ValueError("One bounded original five-file correction archive required")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        if z.namelist() != sorted(files):
            raise ValueError("Sorted unique original archive members required")
        if sum(i.file_size for i in z.infolist()) >= LIMIT:
            raise ValueError("Uncompressed original files exceed cap")
        for name, item in files.items():
            info = z.getinfo(name)
            if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("Original unencrypted stored archive required")
            data = z.read(name)
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError("Original file bytes differ")
    compressed = zlib.compress(archive, 9)
    cut = (len(compressed) + 1) // 2
    parts = (compressed[:cut], compressed[cut:])
    if not all(parts):
        raise ValueError("Two nonempty original transport pieces required")
    binding = {
        "archive_bytes": len(archive),
        "archive_sha256": digest(archive),
        "compressed_bytes": len(compressed),
        "compressed_sha256": digest(compressed),
        "files": files,
        "part0_bytes": len(parts[0]),
        "part0_sha256": digest(parts[0]),
        "part1_bytes": len(parts[1]),
        "part1_sha256": digest(parts[1]),
        "no_model_GPU_training_or_competition_entry": True,
    }
    header = "import base64,hashlib,io,json,zipfile,zlib\nfrom pathlib import Path\n"
    header += "B=" + repr(binding) + "\n"
    first = header + "P=" + repr(base64.b85encode(parts[0]).decode()) + "\n"
    first += r"""
raw=base64.b85decode(P);assert len(raw)==B['part0_bytes'] and hashlib.sha256(raw).hexdigest()==B['part0_sha256']
root=Path('/kaggle/working');assert not (root/'paired-part-000.bin').exists()
with (root/'paired-part-000.bin').open('xb') as f:f.write(raw)
receipt={'status':'private_original_correction_transport_part0_complete','binding':B,'part0_bytes':len(raw),'part0_sha256':hashlib.sha256(raw).hexdigest(),'GPU_model_training_or_competition_entry':False}
(root/'cpu-part-receipt.json').write_text(json.dumps(receipt,sort_keys=True));print(json.dumps(receipt))
"""
    second = header + "P=" + repr(base64.b85encode(parts[1]).decode()) + "\n"
    second += r"""
paths=[p for p in Path('/kaggle/input').rglob('paired-part-000.bin') if p.is_file() and not p.is_symlink()]
assert len(paths)==1 and paths[0].stat().st_size==B['part0_bytes']
part0=paths[0].read_bytes();assert hashlib.sha256(part0).hexdigest()==B['part0_sha256']
part1=base64.b85decode(P);assert len(part1)==B['part1_bytes'] and hashlib.sha256(part1).hexdigest()==B['part1_sha256']
packed=part0+part1;assert len(packed)==B['compressed_bytes'] and hashlib.sha256(packed).hexdigest()==B['compressed_sha256']
d=zlib.decompressobj();raw=d.decompress(packed,B['archive_bytes']+1)
assert len(raw)==B['archive_bytes'] and d.eof and not d.unused_data and not d.unconsumed_tail
assert hashlib.sha256(raw).hexdigest()==B['archive_sha256']
root=Path('/kaggle/working');target=root/'paired-correction-input';assert not target.exists()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert z.namelist()==sorted(B['files']) and sum(i.file_size for i in z.infolist())<2*1024**2
 # Validate ALL members before writing any accepted input.
 for name,item in B['files'].items():
  assert Path(name).name==name and name not in ('.','..')
  info=z.getinfo(name);assert info.compress_type==zipfile.ZIP_STORED and not info.flag_bits&1
  value=z.read(name);assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
 target.mkdir(mode=0o700)
 for name in B['files']:
  with (target/name).open('xb') as f:f.write(z.read(name))
receipt={'status':'private_original_correction_five_files_reconstructed_CPU_only','binding':B,'output_files':B['files'],'GPU_model_training_or_competition_entry':False,'original_Dataset_retry':False}
(root/'cpu-input-receipt.json').write_text(json.dumps(receipt,sort_keys=True));print(json.dumps(receipt))
"""
    sources = (first, second)
    if any(len(s.encode()) >= SOURCE_LIMIT for s in sources):
        raise ValueError("Two-source original transport exceeds predeclared source cap")
    for source in sources:
        compile(source, "private_CPU_only_input", "exec")
    return sources, {
        "binding": binding,
        "source_sha256": [digest(s.encode()) for s in sources],
        "source_bytes": [len(s.encode()) for s in sources],
        "source_limit": SOURCE_LIMIT,
        "maximum_new_CPU_notebooks": 2,
        "new_GPU_jobs_or_competition_entries": 0,
        "original_Dataset_retry": False,
    }
