# Model assets and model choices

This reference explains which model files Dream.exe uses, where they live, and
which models are required or optional. The end-to-end commands for acquiring
the assets required by the bundled case have one owner:
**[INSTALL.md](../INSTALL.md#4-download-the-required-model-assets)**.

Dream.exe never downloads a checkpoint during installation, import, `doctor`,
or a benchmark run. Acquisition is always an explicit user command. With the
released workspace, model files live below repository-root `checkpoints/`.

## Required for the bundled case

| Component | Role | Source or destination | Why it is required |
|---|---|---|---|
| GroundingDINO Swin-T | Region detection | `checkpoints/grounding_dino/` | Detects visual regions from text prompts |
| BERT base uncased | GroundingDINO encoder | `checkpoints/bert-base-uncased/` | Text encoder used by GroundingDINO |
| SAM 2.1 Hiera Large | Region segmentation | `checkpoints/sam2/` | Produces masks from detected regions |
| CoTracker 3 offline | Point tracking | `checkpoints/cotracker/` | Tracks selected image points through the video |
| Upstream DVD base | DVD foundation model | `checkpoints/DVD/official/` | Base weights used by the Dream.exe fine-tuned DVD runtime |
| Wan2.1 DVD runtime files | DVD dependency | `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/` | Diffusion, VAE, and text assets consumed by DVD |
| Dream.exe per-case DVD | Default depth checkpoint for the bundled UID | `checkpoints/DVD/lora/specific/rc_cheesybread_ep000001/` | Produces the depth used by the shipped generated-video experiment |

The public core files are pinned in
`dream_exe/model_assets/configs/core.json`. The Dream.exe DVD files are pinned
in `dream_exe/model_assets/configs/dvd.json`. Those manifests record complete
revisions or digests, byte counts, license notices, and expected paths.

The upstream DVD and CoTracker checkpoints declare CC BY-NC 4.0 terms. Review
[THIRD_PARTY.md](../THIRD_PARTY.md) before using the install guide's explicit
license-acceptance flag.

## Dream.exe DVD release layout

Dream.exe checkpoints are published separately at
[kaimingyang/DVD_for_Dream.exe](https://huggingface.co/kaimingyang/DVD_for_Dream.exe).
The local runtime uses the following canonical tree:

```text
checkpoints/
└── DVD/
    ├── official/
    │   ├── model.safetensors
    │   └── model_config.yaml
    └── lora/
        ├── shared/
        │   ├── model.safetensors
        │   └── model_config.yaml
        └── specific/
            └── <uid>/
                ├── model.safetensors
                └── model_config.yaml
```

We fine-tune DVD with LoRA on Dream.exe benchmark data derived from RoboCasa.
These checkpoints support the project's video-to-trajectory depth estimation.
We acknowledge the official DVD model as the base model and RoboCasa as the
source of the benchmark data. The official base is downloaded separately;
the project model repository distributes our fine-tuned checkpoints.

| HF repository path | Local destination | Runtime selection |
|---|---|---|
| `DVD/lora/shared/` | `checkpoints/DVD/lora/shared/` | `dvd_lora_shared` |
| `DVD/lora/specific/<uid>/` | `checkpoints/DVD/lora/specific/<uid>/` | `dvd_lora_specific` |

Each directory contains `model.safetensors` and `model_config.yaml`.
The packaged `dream_exe/model_assets/configs/dvd.json` records their identities.
A case-specific checkpoint must match the selected UID; it is not a general
unseen-task depth model.

Use the single-case download in [INSTALL.md](../INSTALL.md#4-download-the-required-model-assets)
for quickstart, or the complete-family download in
[BENCHMARK.md](BENCHMARK.md#2-download-the-complete-dreamexe-dvd-family).
Both preserve the `DVD/lora/` prefix below the local `checkpoints/` root.
Training outputs and machine-specific training metadata are not runtime assets.

## Where depth model selection lives

Checkpoint paths and model selection are separate concerns:

- `workspace.json` binds the `checkpoints/` root; it does not choose a depth
  model.
- A formal benchmark route chooses its built-in depth preset through
  `video2traj.depth.model` in the benchmark-owned protocol. Reproduction users
  do not edit that protocol; changing it defines a new experiment.
- A standalone `python -m dream_exe.video2traj` call chooses the preset in its
  caller-owned pipeline JSON.
- A caller-provided depth implementation is registered in `models.json` and
  selected in `runtime.json`, as described in
  [CUSTOM_MODELS.md](CUSTOM_MODELS.md#exec-full-pipeline-and-standalone-video2traj).

The current built-in preset status is:

| Selection | Current status | Assets |
|---|---|---|
| `dvd_official` | Supported official DVD route; `dvd_base` is an alias | Official `FayeHongfeiZhang/DVD` checkpoint plus Wan2.1 runtime files |
| `dvd_lora_specific` | Supported; default for the bundled generated-video route | Upstream DVD base plus one UID-specific Dream.exe checkpoint |
| `dvd_lora_shared` | Supported for routes that explicitly select it | Upstream DVD base plus the shared Dream.exe checkpoint |
| `vda_metric` / `vda_non_metric` | Supported optional presets | Optional VDA source and checkpoint |

For a new candidate experiment, copy the case into an experiment-owned
benchmark root and select one canonical preset in that case's candidate route:

```json
{
  "routes": {
    "candidate": {
      "video2traj": {
        "depth": {"model": "dvd_official"}
      }
    }
  }
}
```

In a standalone video2traj pipeline JSON, the equivalent selection is:

```json
{
  "depth": {"model": "dvd_official"}
}
```

Replace `dvd_official` with `dvd_lora_shared` or `dvd_lora_specific` to select
the corresponding Dream.exe LoRA. Plain `dvd` remains the low-level backend
name for callers supplying a complete explicit config; it is intentionally not
an ambiguous checkpoint preset. Frozen released benchmark protocols must not be
edited in place; a changed preset is a new experiment.

All four DVD selections use the same lazy runtime adapter, but record distinct
model-family and checkpoint provenance. The official preset resolves
`checkpoints/DVD/official/model.safetensors`; LoRA presets resolve the
corresponding `DVD/lora/` subtree and the default remains
`dvd_lora_specific`.

With the shipped workspace, `bindings.checkpoints.dvd_assets` resolves to the
repository `checkpoints/` root, so the default LoRA locations are
`checkpoints/DVD/lora/shared/` and
`checkpoints/DVD/lora/specific/<uid>/`. If checkpoints are stored elsewhere,
make a caller-owned workspace copy and update both `roots.checkpoints` and the
corresponding `bindings.checkpoints.*.path` values (at minimum
`bindings.checkpoints.dvd_assets.path`); the packaged DVD presets and
`dvd.json` identities remain unchanged.

## Additional assets for the full benchmark

The full 101-case benchmark needs the complete `DVD/lora/shared/` and
`DVD/lora/specific/<uid>/` trees. The exact additional download command belongs
to the [benchmark installation](BENCHMARK.md#2-download-the-complete-dreamexe-dvd-family),
because these files are not required for the bundled case.

It also needs the untracked RoboCasa kitchen asset namespace used by the
simulator. The pinned RoboCasa source revision is
`9a3a78680443734786c9784ab661413edb87067b`; source setup does not fetch its
assets. After reviewing RoboCasa's CC BY 4.0 asset notice, acquire them with
the upstream script from the repository root:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets --type all
```

The command writes to `external/RoboCasa/robocasa/models/assets/` and is
interactive (approximately 10 GB). Re-run `python integrations/setup.py
robocasa --workspace configs/workspace.json` afterwards to refresh the
detached runtime copy. This namespace is not included in the tracked-source
revision or the DVD model manifest; its asset version, content manifest, and
license evidence must therefore be recorded separately before claiming a
clean-download benchmark reproduction.

## Optional assets

The following models are never selected merely because their files exist. An
experiment must explicitly select the corresponding provider or backend.

### exec: VDA depth model

VDA is an alternative depth estimator and is not required for the bundled case
or the released default route. Install its optional dependencies and pinned
source checkout in the main environment:

```bash
conda activate dream-exe
bash integrations/setup_dependencies.sh --profile optional
```

Then let the reviewed upstream script populate its own checkpoint directory:

```bash
(cd external/Video-Depth-Anything && bash get_weights.sh)
```

Select `vda_metric` or `vda_non_metric` only in a new experiment. The default
VDA configuration uses the Large (`vitl`) checkpoint, whose model card declares
non-commercial terms.

### exec: FoundationPose pose model

FoundationPose is an optional **pose** backend, not a depth model. The released
default pose route is point-cloud Kabsch and needs no FoundationPose files.
After reviewing NVIDIA's upstream non-commercial terms, a caller-managed
FoundationPose integration may acquire the upstream weights with:

```bash
python -m pip install -e ".[assets]"
gdown --folder \
  "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i" \
  -O checkpoints/foundationpose/weights
```

Downloading these files does not change the default route; the experiment must
also select a FoundationPose backend explicitly.

### video-gen: Wan2.2 generator

Wan2.2 generates candidate videos and belongs to `video_gen`; it is unrelated
to the Wan2.1 files consumed internally by DVD. Its dependencies conflict with
the main execution environment, so prepare it separately:

```bash
conda create -n dream-exe-generation python=3.10 -y
conda activate dream-exe-generation
python -m pip install --upgrade pip
bash integrations/setup_dependencies.sh --profile generation
```

Acquire the pinned checkpoint revision from the main Dream.exe repository root:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B
```

The source checkout is independently pinned to
`Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc`.
Continue with the
[Wan2.2 one-case path](VIDEO_MODELS.md#start-from-zero-with-the-included-wan22).

### video-gen: Dream.exe Wan2.2 I2V A14B LoRA

The [paper-release 2K and 7K LoRAs](https://huggingface.co/kaimingyang/VideoModel_as_RoboPolicy_for_Dream.exe)
require the separate Wan2.2 I2V A14B base checkpoint and pinned
DiffSynth-Studio loader. The complete download, environment, model selection,
and generation commands are in the
[video model guide](VIDEO_MODELS.md#dream-exe-wan22-image-to-video-lora-2k-and-7k).

### eval: hosted VLMs

Hosted VLM evaluation consumes saved outputs after execution. It does not need
another execution checkpoint and never supplies depth, tracking, or pose.
Prepare only local credentials as described in the
[VLM example](../examples/quickstart/VLM.md).
