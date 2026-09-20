"""
Minimal example: stream RSNA training data from Hugging Face and run a few batches.
"""

import numpy as np
import torch

from rsna_dataset import TARGETS, RSNAStreamingDataset, collate_fn


def _manual_transform(img):
    """Fallback transform that resizes to 512x512 and returns a CHW tensor."""
    img = img.resize((512, 512))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def main():
    try:
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.Resize((512, 512)),
                T.ToTensor(),
            ]
        )
    except ImportError:
        transform = _manual_transform

    ds = RSNAStreamingDataset(split="train", transform=transform, label_mode="hard")
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        num_workers=0,
        collate_fn=collate_fn,
    )

    for i, (images, labels) in enumerate(loader):
        print(f"batch {i}: images shape={images.shape}, labels shape={labels.shape}")
        print({t: labels[0, j].item() for j, t in enumerate(TARGETS)})
        if i >= 2:
            break


if __name__ == "__main__":
    main()
