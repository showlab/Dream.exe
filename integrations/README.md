# External provider integration reference

This directory is the implementation surface behind Dream.exe's explicit
provider setup. New users should follow [INSTALL.md](../INSTALL.md); they do not
need to assemble commands from this reference.

Package installation, import, `doctor`, and benchmark runs never clone a
repository, apply a patch, install a provider, or download a checkpoint.

| Path | Responsibility |
|---|---|
| `sources.json` | Pins provider repositories and source revisions |
| `setup.py` | Prepares or verifies selected provider checkouts |
| `setup_dependencies.sh` | Installs one dependency profile and delegates source preparation |
| `download_checkpoints.sh` | Acquires and verifies the pinned public core checkpoint manifest |
| `patches/` | Stores reviewed compatibility patches and their SHA-256 manifests |
| `requirements/` | Stores pinned provider-specific requirement inputs |

## Setup profiles

| Profile | Lifecycle | Providers | Required for bundled case |
|---|---|---|---|
| `core` | `exec` | CoTracker, DVD, GroundingDINO, SAM 2, RoboCasa/RoboSuite | Yes |
| `optional` | `exec` | Video Depth Anything | No |
| `generation` | `video_gen` | Wan2.2 | No; use a separate environment |

Hosted or local VLMs belong to `eval`; no setup profile makes a VLM part of
trajectory extraction or execution. CoTracker is always imported from the
pinned `external/CoTracker` checkout, never from a second pip Git install.

The canonical acquisition paths are:

- required environment, core providers, and bundled-case model assets:
  [INSTALL.md](../INSTALL.md);
- complete benchmark data and Dream.exe model family:
  [docs/BENCHMARK.md](../docs/BENCHMARK.md);
- optional VDA, FoundationPose, and Wan2.2 assets:
  [docs/MODEL_ASSETS.md](../docs/MODEL_ASSETS.md).

License-gated public checkpoints require the matching `--accept-license` value
or the documented non-commercial convenience flag before any network request.
The downloader stages files outside the destination, verifies exact size and
SHA-256, and publishes them atomically.
