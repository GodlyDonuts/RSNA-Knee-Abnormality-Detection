"""Fresh-process synthetic CPU recovery of the real shared optimizer loop."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from check_source_chains import fixture
from path_reader import PathCachedData, PathReader, PathTrainer

from rsna.competitive.spatial_train import SpatialRecipe

D = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class SyntheticOriginal:
    def __init__(self, role):
        self.binding = {"synthetic_only": True, "role": role}
        self.records, self.samples = [], []
        for j in range(16):
            graph, windows, plan = fixture(shear=(j % 3) * 0.5)
            uid = f"synthetic-{role}-{j:02}"
            generator = torch.Generator().manual_seed(842 + j + (100 if role == "outer" else 0))
            maps = torch.randn(7, 8, 4, 4, generator=generator)
            base = torch.randn(12, generator=generator)
            labels = torch.tensor([(j + target) % 2 for target in range(12)], dtype=torch.float32)
            mask = torch.tensor([(j + target) % 3 != 0 for target in range(12)])
            if role == "outer":
                labels.zero_()
                mask.zero_()
            self.records.append({"study_uid": uid, "source_graphs": [graph]})
            self.samples.append(
                {
                    "study_uid": uid,
                    "maps": maps,
                    "base_logits": base,
                    "plan": plan,
                    "labels": labels,
                    "label_mask": mask,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def identical(first, second):
    if isinstance(first, torch.Tensor):
        assert (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and torch.equal(first, second)
        )
    elif isinstance(first, np.ndarray):
        assert (
            isinstance(second, np.ndarray)
            and first.dtype == second.dtype
            and np.array_equal(first, second)
        )
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            identical(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            identical(a, b)
    else:
        assert type(first) is type(second) and first == second


def digest_model(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def child(root, arm, phase):
    mechanism = {
        name: hashlib.sha256((D / name).read_bytes()).hexdigest()
        for name in ("path_reader.py", "variable_paths.py", "variable_source_chains.py")
    }
    data = {
        role: PathCachedData(SyntheticOriginal(role), mechanism) for role in ("fitting", "outer")
    }
    recipe = SpatialRecipe(epochs=2, channels=8, width=4)
    engine = PathTrainer(
        data["fitting"],
        recipe,
        arm=arm,
        device=torch.device("cpu"),
        binding={
            "diagnostic_only": True,
            "path_mechanism_source": mechanism,
            "actual_MRI": False,
            "production_eligible": False,
        },
    )
    initial = digest_model(engine.model)
    output = root / (arm + ("-full" if phase == "full" else "-replay"))
    result = engine.fit(output, resume=phase == "resume", stop_after=1 if phase == "stop" else None)
    if phase != "resume":
        (output / "initial.json").write_text(json.dumps({"model": initial}) + "\n")
    if phase != "stop":
        assert result["completed"] == 4 and len(engine.history) == 4
        assert engine.history[0]["projection_gradient_norm"] == 0
        assert engine.history[-1]["projection_gradient_norm"] > 0
        frame = engine.predictions(data["outer"])
        assert frame.shape == (16, 13)
        frame.to_csv(output / "synthetic-predictions.csv", index=False)


def main():
    if len(sys.argv) > 1:
        assert len(sys.argv) == 4
        child(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
        return
    with tempfile.TemporaryDirectory(prefix="rsna-synthetic-witness-recovery-") as temp:
        root = Path(temp)
        histories = {}
        for arm in ("aligned", "null"):
            for phase in ("full", "stop", "resume"):
                x = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), str(root), arm, phase],
                    capture_output=True,
                    text=True,
                )
                assert x.returncode == 0, x.stderr
            full, replay = root / (arm + "-full"), root / (arm + "-replay")
            a, b = [
                torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=True)
                for p in (full, replay)
            ]
            identical(a, b)
            for epoch in (1, 2):
                identical(
                    *[
                        torch.load(
                            p / "epoch-checkpoints" / f"epoch-{epoch:03}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        for p in (full, replay)
                    ]
                )
            assert (full / "synthetic-predictions.csv").read_bytes() == (
                replay / "synthetic-predictions.csv"
            ).read_bytes()
            histories[arm] = a["history"]
        assert (root / "aligned-full/initial.json").read_bytes() == (
            root / "null-full/initial.json"
        ).read_bytes()
        result = {
            "status": "synthetic_CPU_fresh_process_path_reader_shared_optimizer_full_stop1_resume_exact",
            "source_sha256": hashlib.sha256((D / "path_reader.py").read_bytes()).hexdigest(),
            "synthetic_training_and_prediction_only_cases": [16, 16],
            "epochs": 2,
            "updates_per_full": 4,
            "complete_model_AdamW_LR_cursor_order_allRNG_history_and_each_epoch_state_equal": True,
            "exact16x12_prediction_CSV_bytes_equal": True,
            "both_arms_common_initial_model": True,
            "actual_MRI_GPU_or_accuracy_acceptance": False,
            "full1024_width64_reader_parameters": sum(p.numel() for p in PathReader().parameters()),
            "synthetic_fixture_channels_width": [8, 4],
            "synthetic_update_records": histories,
        }
        (D / "synthetic-reader-recovery.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "synthetic_update_records"}))


if __name__ == "__main__":
    main()
