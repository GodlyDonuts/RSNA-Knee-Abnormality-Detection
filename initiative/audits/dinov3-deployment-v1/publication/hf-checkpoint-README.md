# DINOv3 preparation checkpoint

Git `06816e6b698901bd5148d47a3cdf73252c5a3a89` supplies the verified training runtime, Newton launch and two-epoch Kaggle submission workflow. The accepted labels and image processing remain unchanged. CPU and two-process Gloo recovery passed; actual GPU and competition score acceptance remain separate.

This is an overlay on the pinned complete accepted checkpoint named in manifest.json, whose own base dependencies must be restored first. Apply the overlay only inside a new restoration directory, replacing old files without following symlinks, and verify the entire target file manifest. It contains no credentials, pretrained large weights, or training image cache. Those artifacts have separate pinned sources.
