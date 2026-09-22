# DINOv3 preparation checkpoint

Git `018a43f4a8cf4061a2cbae493a19a171d15916e9` supplies the verified training runtime, Newton launch and two-epoch Kaggle submission workflow. The accepted labels and image processing remain unchanged. CPU and two-process Gloo recovery passed; actual GPU and competition score acceptance remain separate.

This is an overlay on the pinned complete accepted checkpoint named in manifest.json, whose own base dependencies must be restored first. Apply the overlay only inside a new restoration directory, replacing old files without following symlinks, and verify the entire target file manifest. It contains no credentials, pretrained large weights, or training image cache. Those artifacts have separate pinned sources.
