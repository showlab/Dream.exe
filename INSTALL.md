# Install once: everything required for the bundled case

This guide prepares one Linux machine to run the bundled Dream.exe example from
video to trajectory, action, simulation, and evaluation. Follow it once from
top to bottom, including the run command in step 5.

The install is intentionally split by what is being acquired:

| Category | Required for the bundled case | Installed or downloaded here |
|---|---|---|
| Environment | Yes | Python 3.10, the Dream.exe CLI, CUDA-enabled execution dependencies, and FFmpeg-facing Python packages |
| Provider source | Yes | Pinned CoTracker, DVD, GroundingDINO, SAM 2, and RoboCasa/RoboSuite-compatible runtime |
| Simulator asset data | Yes | RoboCasa kitchen assets, downloaded explicitly after the provider checkout (about 10 GB for `--type all`) |
| Model assets | Yes | GroundingDINO, BERT, SAM 2, CoTracker, upstream DVD/Wan2.1 assets, and the Dream.exe DVD checkpoint for the bundled UID |
| Example data | Yes | Already tracked under `examples/quickstart/data/` through Git LFS; no benchmark download is needed |
| Full benchmark data | No | Downloaded later from the [benchmark guide](docs/BENCHMARK.md) |
| Alternative models and hosted evaluation | No | VDA, FoundationPose, Wan2.2 generation, VLM credentials, and custom backends are optional |

Dream.exe never installs, clones, patches, or downloads any of these resources
during import or a benchmark run. Every acquisition step below is explicit.

## 1. Check the system

The validated platform is Linux with an NVIDIA GPU, CUDA 12, Git, Git LFS,
FFmpeg, a C/C++ compiler, and Conda. CPU-only installs can inspect metadata but
cannot run the end-to-end example.

Install Conda and the NVIDIA driver/CUDA toolkit before continuing. On Ubuntu,
the remaining system dependencies can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg build-essential libegl1 libgl1
```

The CUDA toolkit must include `nvcc`; the PyTorch wheel alone does not provide
the compiler required by GroundingDINO. Set `CUDA_HOME` to your CUDA toolkit
directory if it is not detected automatically. EGL is needed for headless
simulation rendering.

```bash
git lfs install
ffmpeg -version
nvidia-smi
nvcc --version
```

## 2. Clone the code and bundled data

```bash
git clone https://github.com/showlab/Dream.exe.git
cd Dream.exe
git lfs pull
```

`git lfs pull` downloads the one-case data and released Kling 3.0 video already
declared by this repository. It does not download the 101-case benchmark.

## 3. Create the environment and install required providers

```bash
conda create -n dream-exe python=3.10 -y
conda activate dream-exe
python -m pip install --upgrade pip 'setuptools>=77' wheel ninja
python -m pip install -e \
  ".[assets,video,sim,robocasa-runtime,dvd-runtime,tracking-runtime,region-runtime]"
SAM2_BUILD_CUDA=0 python -m pip install \
  --no-deps --no-build-isolation \
  -r integrations/requirements/sam2.txt
python integrations/setup.py core \
  --repository-root . \
  --workspace configs/workspace.json
```

These are exactly the dependency groups required by the bundled execution
pipeline. They do not install VDA, FoundationPose, Wan2.2 generation, or the VLM
client. The final command prepares the pinned provider checkouts under
`external/`, applies the reviewed patches, and builds the GroundingDINO CUDA
extension.

These commands assume that the Python 3.10 environment is activated, so
`python` resolves to that interpreter. If the shell has no `python` alias, use
`DREAM_EXE_PYTHON=/path/to/python3.10` or pass `--python /path/to/python3.10`
to the integration scripts; they reject Python versions below 3.10.

It applies only the reviewed compatibility patches declared by:

- `integrations/patches/dvd/manifest.json`;
- `integrations/patches/grounding_dino/manifest.json`;
- `integrations/patches/robocasa/manifest.json`.

The provider setup does **not** download RoboCasa's separate kitchen-asset
archive. After the setup succeeds, acquire that data explicitly in the pinned
RoboCasa checkout (review its terms and the download prompt first):

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets --type all
python integrations/setup.py robocasa \
  --repository-root . \
  --workspace configs/workspace.json
```

The `--type all` archive is roughly 10 GB and is written into the RoboCasa
asset locations expected by the simulator; it is not part of the Dream.exe
GitHub code package. A clean install must complete this step before the
bundled execution case or benchmark simulator routes can be claimed as
available. Run the read-only provider check and `dream-exe doctor` afterwards.

Do not run `git apply` manually. The setup command is idempotent and its
read-only verification mode checks the environment, pinned sources, patches,
and build products:

```bash
python -m pip check
python integrations/setup.py core \
  --repository-root . \
  --workspace configs/workspace.json \
  --check
```

## 4. Download the required model assets

First review [THIRD_PARTY.md](THIRD_PARTY.md). CoTracker and the upstream DVD
checkpoint are non-commercial assets whose terms must be accepted explicitly.
Previewing the plan performs no network access or writes:

```bash
bash integrations/download_checkpoints.sh --dry-run
```

After accepting the listed terms, acquire the pinned public assets:

```bash
bash integrations/download_checkpoints.sh \
  --accept-noncommercial-licenses
```

This downloads GroundingDINO, BERT, SAM 2, CoTracker, the official DVD model,
and the Wan2.1 files consumed by DVD into `checkpoints/`. The official model is
both the base used by the Dream.exe LoRA routes and a directly selectable
`dvd_official` depth preset.

Download only the Dream.exe DVD LoRA checkpoint for the bundled case. Our
default depth route uses DVD fine-tuned with LoRA on Dream.exe benchmark data,
based on RoboCasa; the upstream DVD model is acknowledged as the base model.

> Release synchronization is in progress; the commands require the complete
> canonical `DVD/lora/` files on Hugging Face.

```bash
hf download kaimingyang/DVD_for_Dream.exe \
  --include "DVD/lora/specific/rc_cheesybread_ep000001/*" \
  --local-dir checkpoints
```

Run from the repository root. This fetches only the case's
`model.safetensors` and `model_config.yaml` into
`checkpoints/DVD/lora/specific/rc_cheesybread_ep000001/`.
The public assets downloaded above are still required. Download the full
checkpoint family only when moving to the full benchmark.

Keep `--local-dir checkpoints`: Hugging Face preserves the `DVD/lora/`
repository prefix. For another storage location, update `roots.checkpoints`
and the checkpoint bindings in a caller-owned workspace copy.

Verify the public asset bundle without downloading or modifying anything:

```bash
bash integrations/download_checkpoints.sh --check
```

The `dream-exe run` command also validates the Dream.exe DVD files against the
packaged release manifest before GPU inference. Model sources, exact revisions,
destinations, licenses, and the full checkpoint tree are recorded in the
[model-assets reference](docs/MODEL_ASSETS.md).

## 5. Run the bundled case

From the repository root, run:

```bash
dream-exe doctor --workspace examples/quickstart/workspace.json --case rc_cheesybread_ep000001
dream-exe run --workspace examples/quickstart/workspace.json --spec examples/quickstart/run.json
```

The shipped
`examples/quickstart/workspace.json` already binds:

- the bundled case data;
- repository-root `external/` provider checkouts;
- repository-root `checkpoints/` model assets;
- the isolated `.dream-exe/quickstart/` work and output roots.

Do not edit that workspace and do not run `dream-exe configure` for the bundled
case. If the run reports a missing asset, use the README's `doctor` command and
the [bundled-case troubleshooting section](examples/quickstart/README.md#troubleshooting).

## Additional installs are optional or workflow-specific

Nothing below is required to run the bundled case.

| Additional goal | What must be added | Guide |
|---|---|---|
| Run the 101-case benchmark | Full benchmark data and the complete Dream.exe DVD checkpoint family | [Benchmark and reproduction](docs/BENCHMARK.md) |
| Try Video Depth Anything | Optional VDA dependency/source profile and a selected VDA checkpoint | [Optional model assets: VDA](docs/MODEL_ASSETS.md#exec-vda-depth-model) |
| Try FoundationPose | Caller-managed FoundationPose runtime and weights; the default pose route remains point-cloud Kabsch | [Optional model assets: FoundationPose](docs/MODEL_ASSETS.md#exec-foundationpose-pose-model) |
| Generate videos with Wan2.2 | A separate generation environment, source checkout, and checkpoint | [Optional model assets: Wan2.2](docs/MODEL_ASSETS.md#video-gen-wan22-generator) |
| Run VLM visual evaluation | Install `.[vlm]` and add local endpoint credentials; no additional execution model | [VLM example](examples/quickstart/VLM.md) |
| Use another provider or model | Only that provider's dependencies/assets plus an explicit model catalog/runtime selection | [Custom model backends](docs/CUSTOM_MODELS.md) |

Providers, checkpoints, or writable outputs may live outside the repository.
Create a caller-owned workspace only in that case; the
[configuration reference](docs/CONFIGURATION.md) explains the path keys. Full
benchmark data location is configured only in the benchmark workflow.
