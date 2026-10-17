"""Stream full maps while retaining the SAME original sparse parent reference.

Sparse/dense changes evidence coverage and RGB context, not the teacher. Actual
GPU encoder equality/throughput and complete trainer recovery are still needed.
No full-cohort feature cache, encoder fitting or cross-series registration.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import torch
from dense_inputs import VERSION, source_plan, tensor_chunks
from path_reader import prepare_plan

from rsna.competitive.spatial_data import OnlineSpatialData, PlanCache, _supervision
from rsna.data import select_series
from rsna.schema import TARGETS


def rechunk(parts, size):
    """Keep canonical whole-study encoder batches across series boundaries."""
    if type(size) is not int or size < 1:
        raise ValueError("Positive fixed encoder chunk size required")
    remaining = None
    for part in parts:
        if part.ndim != 4 or part.shape[1] != 3 or len(part) > size:
            raise ValueError("Bounded three-channel source image chunks required")
        values = part if remaining is None else torch.cat((remaining, part))
        while len(values) >= size:
            yield values[:size].contiguous()
            values = values[size:]
        remaining = values
    if remaining is not None and len(remaining):
        yield remaining.contiguous()


class StreamedPathData:
    def __init__(self, dataset, parent, binding, *, sampling, seed=20261017, chunk_size=12):
        # Reuse the original fixed384/16/six-series, no augmentation/contrast
        # and frozen-eval-parent admission. Do not mutate that accepted object.
        self.original = OnlineSpatialData(
            dataset, parent, binding, seed=seed, chunk_size=chunk_size
        )
        if sampling not in {"sparse16", "dense"}:
            raise ValueError("Explicit source sampling required")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Positive fixed encoder chunk size required")
        self.dataset, self.parent = dataset, parent
        self.sampling, self.chunk_size = sampling, chunk_size
        self.plans = PlanCache(seed)
        self.binding = {
            "original": binding,
            "input_policy": VERSION,
            "sampling": sampling,
            "teacher": "unchanged_original_sparse16_parent_for_both_samplings",
            "encoder_chunk_size": chunk_size,
            "encoder_fitted": False,
        }

    def __len__(self):
        return len(self.dataset)

    def metadata(self, uid):
        values = []
        for slot, entry in enumerate(select_series(self.dataset.series[uid], 6, False)):
            sid = entry["SeriesInstanceUID"]
            expected, meta = self.dataset._metadata[(uid, sid)]
            path = self.dataset.cache / self.dataset.raw_split / uid / (sid + ".npz")
            if hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest() != expected:
                raise ValueError("Original source metadata bytes changed")
            if (meta["study_uid"], meta["series_uid"]) != (uid, sid):
                raise ValueError("Original source metadata membership differs")
            values.append((slot, path, expected, meta))
        return values

    def images(self, entries, sampling):
        for _, path, _, meta in entries:
            raw = (
                path.read_bytes()
                if self.dataset.volume_provider is None
                else self.dataset.volume_provider(meta)
            )
            if (
                len(raw) != meta["cache_bytes"]
                or hashlib.sha256(raw).hexdigest() != meta["cache_sha256"]
            ):
                raise ValueError("Original native volume bytes changed")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                volume = archive["volume"]
            del raw
            for _, _, images in tensor_chunks(
                volume, meta, sampling=sampling, side=384, chunk_size=self.chunk_size
            ):
                yield images

    def encode(self, entries, sampling, *, keep_maps):
        parent = self.parent
        if parent.training or any(p.requires_grad for p in parent.parameters()):
            raise ValueError("Original parent must remain frozen and in evaluation mode")
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
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        # Original window_tensor performs this affine AFTER
                        # native grayscale normalization, on CPU, before CUDA.
                        # Omitting it preserves pixel order but changes every
                        # encoder input; native [0,1] equality alone is weaker.
                        embeddings.append(parent.encoder(images.mul(2).sub(1).to(device)))
                if not embeddings:
                    raise ValueError("Original study has no complete three-slice observation")
                with torch.autocast(device.type, enabled=False):
                    logits = parent.pool(torch.cat(embeddings))
        finally:
            if handle is not None:
                handle.remove()
        if keep_maps and len(captured) != len(embeddings):
            raise ValueError("Exactly one original pre-pooling map batch per encoder call required")
        return torch.cat(captured) if keep_maps else None, logits.detach()

    def __getitem__(self, index):
        row = self.dataset.studies.iloc[index]
        uid = row.study_uid
        entries = self.metadata(uid)
        # Dense evidence must not quietly replace the common sparse teacher.
        maps, base = self.encode(entries, "sparse16", keep_maps=self.sampling == "sparse16")
        if self.sampling == "dense":
            maps, _ = self.encode(entries, "dense", keep_maps=True)
        graphs, windows = [], []
        for slot, _, digest, meta in entries:
            graph, members = source_plan(
                meta, sampling=self.sampling, slot=slot, metadata_sha256=digest
            )
            graphs.append(graph)
            windows.append(members)
        windows = np.concatenate(windows)
        if len(maps) != len(windows):
            raise ValueError("Original image windows, maps and source support differ")
        original_plan = self.plans.get(uid, graphs, windows, tuple(maps.shape[1:]))
        plan = prepare_plan(graphs, windows, original_plan)
        labels, masks = _supervision(
            [row.get(t, 0.0) for t in TARGETS], [row.get(t + "__mask", 0.0) for t in TARGETS]
        )
        return {
            "study_uid": uid,
            "maps": maps,
            "base_logits": base,
            "plan": plan,
            "labels": labels,
            "label_mask": masks,
        }
