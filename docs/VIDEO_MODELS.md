# Evaluate your own videos and video generation models

This guide covers only the video being evaluated: an externally generated MP4,
a video produced by a world action model (WAM), a hosted generation API, or a
locally deployed video generator. A WAM output is a normal candidate video; it
does not have to be a policy rollout. Video generators belong to catalog
category `video_gen` and kind `video_generation`. Detector, tracking, depth,
and pose belong to `exec`; VLM judges belong to `eval`. Those replaceable
pipeline components are documented separately in
[CUSTOM_MODELS.md](CUSTOM_MODELS.md).

## Choose the shortest route

| What you already have | Route | Code required? |
|---|---|---|
| Nothing beyond the installed repo | Use the included Wan2.2 adapter on the bundled case | No |
| The released Dream.exe Wan2.2 I2V LoRA and its upstream base model | Select the 2K or 7K catalog entry | No |
| One MP4 or a directory of MP4s | Import the finished video | No |
| A hosted asynchronous generation API | Implement `PollingImageToVideoBackend` | Three transport methods |
| A local model or synchronous service | Implement `BaseImageToVideoBackend` | One generation method |

For benchmark-comparable results, every model must receive the exact case
first frame and either the `standard` or `enhanced` prompt from:

```text
<bench>/cases/<uid>/generation/first_frame.png
<bench>/cases/<uid>/generation/input.json
```

The output must be a non-empty MP4.

## Every candidate video is preprocessed

All videos must pass the same preprocessing before video2traj, including a
Wan2.2 result, an imported WAM video, an API result, and a local-model result.
The `dream-exe generate` and `dream-exe bench import-video[s]` commands do this
automatically; do not run a separate FFmpeg command.

Dream.exe stores both files under
`outputs/videos/<uid>/<model>/<variant>/`:

```text
video.mp4           # exact model/user output, preserved for provenance
preprocessed.mp4    # validated input consumed by video2traj
video.json          # hashes, producer identity, and preprocessing contract
```

The current benchmark contract resizes the video to 512×512, keeps the source
frame rate, encodes H.264/yuv420p, and removes audio. The command validates the
derived video with FFprobe before publishing it. `--preprocessed` remains an
advanced import override for an existing derivative; it must already satisfy
the same contract and is normally omitted.

## Start from zero with the included Wan2.2

Dream.exe includes a pinned Wan2.2 TI2V-5B adapter. The bundled benchmark case
already provides its frozen initialization, first frame, standard/enhanced
prompts, protocol, and reference data, so the full 101-case download is not
needed for this first generated-video run.

Wan2.2 uses a separate environment because its dependencies conflict with the
main execution pipeline. First complete [INSTALL.md](../INSTALL.md), then
prepare the generation environment and weights in
[MODEL_ASSETS.md](MODEL_ASSETS.md#video-gen-wan22-generator).

Generate and preprocess one video in that environment:

```bash
conda activate dream-exe-generation
dream-exe generate \
  --workspace examples/quickstart/workspace.json \
  --case rc_cheesybread_ep000001 \
  --model Wan2.2 \
  --variant standard \
  --run-id wan22-one-case
```

This reads the bundled first frame and prompt, invokes the implemented Wan2.2
backend, preserves `video.mp4`, creates `preprocessed.mp4`, registers both, and
writes the next-step run spec. Then switch to the main environment:

```bash
conda activate dream-exe
dream-exe run \
  --workspace examples/quickstart/workspace.json \
  --spec .dream-exe/quickstart/work/imports/wan22-one-case/run.json
```

Together the two commands cover the complete case flow:
`frozen initialization/first frame → Wan2.2 video generation → preprocessing →
video2traj → action → simulation execution → evaluation`. The environment
switch is the only split; no manifest or configuration file needs editing.

## Dream.exe Wan2.2 image-to-video LoRA: 2K and 7K

Our [paper-release model repository](https://huggingface.co/kaimingyang/VideoModel_as_RoboPolicy_for_Dream.exe)
contains two Wan2.2 I2V A14B LoRA checkpoints. Each checkpoint needs its
matching **high-noise and low-noise** adapter files plus the separate
[Wan2.2 I2V A14B base model](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B).
The 2K and 7K entries in
[`configs/models.wan22_lora.example.json`](../configs/models.wan22_lora.example.json)
run those pairs through DiffSynth-Studio. This is a different base model and
loader from the built-in Wan2.2 TI2V-5B adapter above.

Use a Python 3.10 generation environment with Dream.exe's `video` extra and
the [pinned DiffSynth-Studio source](https://github.com/modelscope/DiffSynth-Studio/tree/7686e54d41d25c0e8ed5f1318acc23b6bb832654):

```bash
conda create -n dream-exe-diffsynth python=3.10 -y
conda activate dream-exe-diffsynth
python -m pip install -e '.[video,assets]'
git clone https://github.com/modelscope/DiffSynth-Studio.git external/DiffSynth-Studio
git -C external/DiffSynth-Studio checkout 7686e54d41d25c0e8ed5f1318acc23b6bb832654
python -m pip install -e external/DiffSynth-Studio
```

From the Dream.exe repository root, download the base and either LoRA pair.
The commands below download both pairs so either catalog entry can be used:

```bash
hf download Wan-AI/Wan2.2-I2V-A14B \
  --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3 \
  --local-dir checkpoints/Wan-AI/Wan2.2-I2V-A14B
hf download kaimingyang/VideoModel_as_RoboPolicy_for_Dream.exe \
  --revision f90d07b69c6c434902617b6305ad83731fe926ff \
  --include 'Wan2.2_I2V_A14B_lora_2k/*' 'Wan2.2_I2V_A14B_lora_7k/*' 'inference_config.json' 'weights_manifest.json' \
  --local-dir checkpoints/Dream.exe/Wan2.2-I2V-A14B-LoRA
```

Then generate the bundled case with one released checkpoint:

```bash
dream-exe models check \
  --models-config configs/models.wan22_lora.example.json \
  --model Wan2.2-LoRA-7K --category video_gen --kind video_generation
dream-exe generate \
  --workspace examples/quickstart/workspace.json \
  --case rc_cheesybread_ep000001 \
  --models-config configs/models.wan22_lora.example.json \
  --model Wan2.2-LoRA-7K --variant standard \
  --run-id wan22-lora-7k-one-case
```

Select `Wan2.2-LoRA-2K` to use the 2K pair. The adapter reads the released
`inference_config.json` for 480 × 480, 81 frames, 16 FPS, 40 steps, CFG 3.5,
the negative prompt, and other defaults. `--size`, `--frame-num`,
`--sample-steps`, and `--guidance-scale` can override the corresponding values.
The command preprocesses the result and records the model and base revisions.
Run `dream-exe run` in the main environment using the printed `run_spec` path
to continue through execution and evaluation.

## Route A: import finished videos

### One video

The embedded case is the fastest test and does not require the full benchmark:

```bash
dream-exe bench import-video \
  --workspace examples/quickstart/workspace.json \
  --case rc_cheesybread_ep000001 \
  --model MyVideoModel \
  --variant standard \
  --video /absolute/path/generated.mp4 \
  --producer-revision my-checkpoint-or-api-version \
  --run
```

The command copies the original and automatically creates the required
`preprocessed.mp4`. `--move` intentionally removes the original after an
atomic same-filesystem transfer, so do not use it for the first test.

`producer-kind` describes how the MP4 was obtained, not the model family. Keep
the default `generator` for a WAM that predicts or generates a video. Use
`policy_rollout` only when the MP4 records a policy actually executing in an
environment or on a robot. `--producer-name`, `--producer-revision`, and
`--seed` are optional but strongly recommended for reproducibility.

`--variant standard` and `--variant enhanced` mean that the corresponding
frozen benchmark prompt was used. Use `--variant custom --prompt "..."` only
for a one-case exploratory import; that result is not directly comparable to
the released prompt protocol.

### A complete benchmark batch

Put exactly one video per benchmark UID in one directory:

```text
my-videos/
├── rc_breadandcheese_ep000001.mp4
├── rc_breadandcheese_ep000017.mp4
└── <uid>.mp4
```

Then import the closed collection. Omit `--run` to validate and register the
batch first; add it only when ready to launch the full pipeline.

```bash
dream-exe bench import-videos \
  --workspace configs/workspace.json \
  --model MyVideoModel \
  --variant standard \
  --video-dir /absolute/path/my-videos \
  --producer-revision my-checkpoint-or-api-version
```

The default filename pattern is `{uid}.mp4`. Change it with `--pattern` only
when every file follows another UID-based pattern.

## Route B: connect a hosted generation API

Copy the two polling examples rather than starting from an empty file:

```text
examples/custom_models/video_generation_backend.py
examples/custom_models/models.example.json
```

Subclass `dream_exe.generation.PollingImageToVideoBackend` and implement:

- `submit(...)`: send the first frame, prompt, seed, and parameters; return a
  provider job handle;
- `poll(job)`: return the current provider status;
- `download(job, status, output_path)`: write the completed MP4 to
  `output_path` and return credential-free provenance.

The base class supplies bounded retries, a monotonic timeout, terminal-failure
handling, temporary downloads, atomic publication, and cleanup. Keep the API
key in an environment variable; never put the key in JSON.

Register the backend in a trusted `dream-exe.models` catalog:

```json
{
  "format": "dream-exe.models",
  "categories": {
    "video_gen": {
      "my_video_api": {
        "kind": "video_generation",
        "backend": "factory",
        "factory": "./video_generation_backend.py:MyPollingBackend",
        "kwargs": {
          "api_key_env": "MY_VIDEO_API_KEY",
          "timeout_seconds": 900,
          "poll_interval_seconds": 2,
          "max_transport_attempts": 3
        },
        "options": {},
        "identity": {
          "provider_kind": "external",
          "backend_id": "my_video_api",
          "contract_version": "image_to_video_backend"
        }
      }
    }
  }
}
```

Required user choices are `factory`, `api_key_env`, and both identity IDs.
The timeout, polling interval, retries, and generation `options` above are safe
starting defaults; change them only when the provider requires different
values.

## Route C: connect a local model

Subclass `dream_exe.generation.BaseImageToVideoBackend` and implement one
method:

```python
def generate(
    self, *, image_path, prompt, output_path, seed, parameters
):
    # Run local inference and write a complete MP4 to output_path.
    return {"checkpoint_revision": "my-model-revision"}
```

Start from `ExampleLocalVideoBackend` in
`examples/custom_models/video_generation_backend.py`. Register it with the
same catalog structure as the API route, but point `factory` at the local class
and remove API-only constructor keys. Put model defaults such as size, frame
count, sampling steps, and guidance under `options`; the provided example uses
the released Wan-compatible defaults.

## Check, generate, and optionally run

Check trusted factory code before allocating a model or simulator:

```bash
dream-exe models check \
  --models-config /absolute/path/models.json \
  --model my_video_api \
  --category video_gen --kind video_generation

dream-exe generate \
  --workspace examples/quickstart/workspace.json \
  --case rc_cheesybread_ep000001 \
  --variant standard \
  --models-config /absolute/path/models.json \
  --model my_video_api \
  --dry-run
```

Remove `--dry-run` to generate, preprocess, and register the video. Add `--run`
to execute the complete evaluation pipeline too when the generator and
execution dependencies coexist in one environment. For a local model, use its
catalog ID in place of `my_video_api`.

Catalog generation records the catalog digest, exact model ID, source digest,
backend identity, effective parameters, and video provenance. A changed model
definition therefore invalidates exact resume instead of silently reusing an
older output.

## Scope limit for unrelated videos

An arbitrary MP4 from a different scene does not have the frozen simulator
state, camera calibration, objects, task protocol, or reference trajectory
needed for a benchmark execution. It can still be passed to the standalone
video2traj interface, but a full comparable Dream.exe case requires a matching
benchmark case package.
