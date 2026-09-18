# DVD provider patch

This directory contains only the narrow compatibility delta required by the
Dream.exe DVD runtime. It does not contain DVD provider source, model
configuration, checkpoints, tokenizers, or other model assets.

The patch is pinned to
`EnVision-Research/DVD@5501c8bbf5983554f5bc8d3747a26e0d0c49d4ee`.
Its SHA-256 and sole allowed target are recorded in `manifest.json`. Apply it
only to a detached checkout of that exact commit, following
`docs/DVD_PROVIDER.md`.
