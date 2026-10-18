"""Version2: preserve DeploymentBagClassifier's explicitly qualified FP32 path.

The first streamed GPU probe failed original-parent equality because the new
wrapper copied the training model's BF16 context. Do not replace old source or
claim the original frozen encoder was wrong. This is a distinct input wrapper.
"""

import torch
from online_inputs import StreamedPathData, rechunk

from rsna.competitive.compact_infer import DeploymentBagClassifier


class StreamedFP32PathData(StreamedPathData):
    def __init__(self, dataset, parent, binding, *, sampling, seed=20261017, chunk_size=12):
        if not isinstance(parent, DeploymentBagClassifier) or parent.precision != "float32":
            raise ValueError("Exact explicit FP32 deployment parent required")
        super().__init__(
            dataset, parent, binding, sampling=sampling, seed=seed, chunk_size=chunk_size
        )
        self.binding.update(encoder_precision="float32", stream_version="deployment_FP32_stream_v2")

    def encode(self, entries, sampling, *, keep_maps):
        parent = self.parent
        if (
            parent.training
            or parent.precision != "float32"
            or any(p.requires_grad for p in parent.parameters())
            or any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in parent.state_dict().values()
            )
        ):
            raise ValueError("Original complete FP32 frozen deployment parent must stay unchanged")
        device = next(parent.parameters()).device
        captured, embeddings = [], []
        handle = None
        if keep_maps:
            handle = parent.encoder.head.global_pool.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().float().contiguous())
            )
        try:
            with torch.no_grad():
                for images in rechunk(self.images(entries, sampling), self.chunk_size):
                    # Exactly DeploymentBagClassifier.forward's context. A CUDA
                    # device alone never licenses enabling training-time BF16.
                    with torch.autocast(device.type, dtype=torch.float16, enabled=False):
                        embeddings.append(parent.encoder(images.mul(2).sub(1).to(device)))
                if not embeddings:
                    raise ValueError("No complete original three-slice observation")
                with torch.autocast(device.type, enabled=False):
                    logits = parent.pool(torch.cat(embeddings).float())
        finally:
            if handle is not None:
                handle.remove()
        if keep_maps and len(captured) != len(embeddings):
            raise ValueError("One original full-map batch per encoder call required")
        return torch.cat(captured) if keep_maps else None, logits.detach()
