import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("CPU_preparation_tests", HERE / "benchmark.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_complete_tensor_and_support_bytes_affect_hash():
    from rsna.competitive.predictive.density import Supports

    value = {
        "images": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        "supports": Supports(torch.tensor([0, 2]), torch.tensor([0, 1]), torch.zeros(12, 1), 2),
    }
    original = b.context_hash(b.pack(value))
    altered = {**value, "supports": dataclasses.replace(value["supports"], region_count=3)}
    assert b.context_hash(b.pack(altered)) != original
    value["images"][0, 0, 0] = -0.0
    assert b.context_hash(b.pack(value)) != original
    with pytest.raises(TypeError):
        b.pack(object())


def test_spawned_original_DICOM_preparation_matches_threads(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

    tasks = []
    for case in range(3):
        uid = f"1.2.826.0.1.3680043.8.498.90{case}"
        series = uid + ".1"
        for i in range(2):
            sop = series + f".{i + 1}"
            path = tmp_path / "test_series" / uid / series / (sop + ".dcm")
            path.parent.mkdir(parents=True, exist_ok=True)
            fm = FileMetaDataset()
            fm.TransferSyntaxUID = ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID = MRImageStorage
            fm.MediaStorageSOPInstanceUID = sop
            ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
            ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = uid, series, sop
            ds.SOPClassUID = MRImageStorage
            ds.InstanceNumber = i + 1
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, i]
            ds.Rows, ds.Columns = 96, 112
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
            ds.PixelData = (
                (np.arange(96 * 112, dtype=np.uint16) + case * 17 + i * 53) % 4096
            ).tobytes()
            pydicom.dcmwrite(path, ds, enforce_file_format=True)
        tasks.append([case, str(tmp_path), uid])
    (tmp_path / "_predictive_descriptor_cache.py").write_text(
        (HERE.parent / "predictive-final-serving-20261020-v1/descriptor_cache.py").read_text()
    )
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    script = tmp_path / "check.py"
    script.write_text("""import benchmark,json,time
from pathlib import Path
if __name__=='__main__':
 benchmark.initialize()
 tasks=json.loads(Path(__file__).with_name('tasks.json').read_text())
 a=benchmark.trial('thread',2,tasks,lambda:90)
 b=benchmark.trial('process',2,tasks,lambda:90)
 assert [x['context_sha256'] for x in a['studies']]==[x['context_sha256'] for x in b['studies']]
 assert [x['stats'] for x in a['studies']]==[x['stats'] for x in b['studies']]
 assert len(set(x['worker_pid'] for x in b['studies']))==2
 print('complete_contexts_and_original_DICOM_stats_equal')
""")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            map(str, (HERE, tmp_path, ROOT, Path(pydicom.__file__).parents[1]))
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "complete_contexts_and_original_DICOM_stats_equal" in result.stdout
