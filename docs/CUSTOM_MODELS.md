# Replace exec and eval models

Dream.exe separates replaceable models by lifecycle responsibility:

| Category | Responsibility | Kinds in this guide |
|---|---|---|
| `exec` | Convert a candidate video into trajectory/action inputs used for execution | `region_detector`, `region_segmenter`, `tracking`, `depth`, `pose` |
| `eval` | Consume saved videos, trajectories, traces, and metrics | `vlm` |

VLM is an `eval` model and never drives execution. Depth and the other
video2traj components are `exec` models. `video_gen` is a third, separate
category documented in [VIDEO_MODELS.md](VIDEO_MODELS.md).

Dream.exe separates each component's Python input/output contract from its
transport. A backend may call a hosted API, a service on `localhost`, a
subprocess, or an in-process PyTorch model. The pipeline sees the same typed
Python boundary in every case.

There are two integration levels:

| Goal | Integration |
|---|---|
| Use an OpenAI-compatible VLM endpoint | Change `base_url`, upstream `model`, and `api_key_env` in a model catalog. |
| Use a new model or provider protocol | Implement one small backend class and register its factory. |

“Any model” means any implementation that satisfies the public input/output
contract below. Dream.exe does not guess arbitrary provider JSON fields and
does not silently adapt incompatible coordinates or frame rates.

## Catalog and runtime files

The complete catalog example lives in
[`examples/custom_models/models.example.json`](../examples/custom_models/models.example.json).
It has root format `dream-exe.models`, exact case-sensitive model IDs, and three
top-level category objects:

```text
categories.video_gen  -> video_generation
categories.exec       -> region_detector, region_segmenter, tracking, depth, pose
categories.eval       -> vlm
```

The loader rejects a kind placed in the wrong category. The old flat
`"models": {...}` representation remains readable for compatibility, but new
catalogs should use `"categories"`.

A factory uses one of these forms:

```json
"factory": "package.module:BackendClass"
```

```json
"factory": "./tracking_backend.py:ExampleTrackingBackend"
```

The second form is relative to `models.json`, must remain inside that
directory, and must name a regular non-symlink `.py` file. The catalog cannot
replace an exact built-in ID. Unknown IDs and kind mismatches fail with the
available same-kind IDs.

The optional
[`examples/custom_models/runtime.example.json`](../examples/custom_models/runtime.example.json)
has format `dream-exe.runtime`. It selects region, tracking, depth, and pose
models for the formal pipeline and declares `"category": "exec"`. It cannot
select a VLM or video generator. A partial file overrides only its named
stages; other stages retain the current benchmark defaults. Precedence is:

1. current built-in runtime defaults;
2. the selected catalog definition;
3. selector-level `options` in `runtime.json`.

For a factory model, catalog `kwargs` are constructor defaults and runtime
selector `options` override those constructor values for one experiment. The
existing complete backend runtime JSON remains accepted when it omits
`format`; model selectors require `dream-exe.runtime`.

List and statically check models before a run:

```bash
dream-exe models list \
  --models-config examples/custom_models/models.example.json \
  --category exec

dream-exe models check \
  --models-config examples/custom_models/models.example.json \
  --model example_tracker \
  --category exec --kind tracking
```

The static check loads trusted code and validates the factory, constructor,
method signature, declared identity, optional dependencies, and explicitly
configured asset paths. It never downloads a weight, clones a repository,
installs a dependency, or patches third-party source. The example classes pass
the static check but intentionally raise `NotImplementedError` during inference
until their marked hook is implemented.

An optional real smoke check uses a caller-owned JSON file with a factory that
builds Python call arguments:

```json
{
  "factory": "./smoke_inputs.py:build_tracking_smoke",
  "kwargs": {}
}
```

`build_tracking_smoke()` must return
`{"args": [...], "kwargs": {...}}`. This file and its factory are trusted
executable input, just like `models.json`.

## Public contracts

| Category | Kind | Base class | Method and normalized result |
|---|---|---|---|
| `exec` | `region_detector` | `dream_exe.video2traj.region.BaseRegionDetector` | `detect(image_rgb[H,W,3], prompt)`; return one pixel `xyxy` proposal. |
| `exec` | `region_segmenter` | `dream_exe.video2traj.region.BaseRegionSegmenter` | `segment_from_bbox(image_rgb, bbox_xyxy, prompt)`; return a frame-aligned `[H,W]` mask. |
| `exec` | `tracking` | `dream_exe.video2traj.tracking.BaseTrackingPredictionBackend` | `predict(...)`; return tracks `[T,N,2]`, visibility `[T,N]`, queries `[N,2]`, and effective mode. Coordinates are pixel `xy`. |
| `exec` | `depth` | `dream_exe.video2traj.depth.BaseDepthBackend` | `infer(...)`; output must normalize to a frame-aligned depth stack `[T,H,W]`. Declare units/scale in provenance when not metric. |
| `exec` | `pose` | `dream_exe.video2traj.pose.BasePoseBackend` | `infer(PoseBackendRequest) -> PosePrediction`; candidates align to video frames and use the request's camera-frame conventions. |
| `eval` | `vlm` | `dream_exe.evaluation.vlm.BaseVLMBackend` | `infer(prompt, media_path, generation_options) -> str`; return the raw provider text. |

The six minimal Python template files are in
[`examples/custom_models/`](../examples/custom_models/). Region detection and
segmentation share one template file because they compose into one region
stage.

Every factory entry must declare credential-free identity. Use
`provider_kind: "external"`, a stable lowercase `backend_id`, and the exact
contract version shown by the example catalog. Region providers also declare a
stable `algorithm_id`. The object created by the factory must expose matching
values; a mismatch fails before simulator startup.

## eval: VLM judges

For an OpenAI-compatible service, no Python class is needed:

```json
{
  "format": "dream-exe.models",
  "categories": {
    "eval": {
      "my_vlm": {
        "kind": "vlm",
        "backend": "openai-compatible",
        "options": {
          "model": "provider-model-name",
          "base_url": "https://provider.example/v1",
          "api_key_env": "MY_VLM_API_KEY"
        }
      }
    }
  }
}
```

Set the key only in the environment, then select the catalog ID:

```bash
export MY_VLM_API_KEY=...

dream-exe eval-vlm \
  --models-config /trusted/path/models.json \
  --model my_vlm \
  --mode video_only \
  --media-dir /path/to/grids \
  --prompts-json /path/to/prompts.json \
  --rubric subject_stability \
  --output-csv /path/to/results.csv
```

Catalog mode cannot be mixed with direct `--base-url`, `--api-key-env`,
credential-profile, or backend-label flags. The existing direct endpoint mode
continues to work when `--models-config` is absent. Raw response artifacts,
parsing, prompts, retry records, and cache behavior continue to use the current
evaluation contracts.

Use `BaseVLMBackend` for a non-compatible API or an in-process VLM. Its
`inference_identity()` must describe behavior-affecting settings without URLs
containing credentials, headers, API keys, or tokens.

## exec: full pipeline and standalone video2traj

Compose custom core models into the formal single-UID workflow:

```bash
dream-exe run \
  --workspace configs/workspace.json \
  --spec /path/to/run.json \
  --models-config /trusted/path/models.json \
  --runtime-config /path/to/runtime.json
```

The same resolver is available without the simulator:

```bash
python -m dream_exe.video2traj \
  --video /path/to/video.mp4 \
  --simulator-config /path/to/simulator.json \
  --pipeline-config /path/to/pipeline.json \
  --runtime-config /path/to/runtime.json \
  --models-config /trusted/path/models.json
```

The formal command still enters the existing `init → video2traj → action →
simulation execution → evaluation` workflow. A successful custom-backend run
proves that extension plumbing executed; it does not by itself establish the
scientific validity of the new model.

## Provenance, resume, and trust

Dream.exe fingerprints the catalog and runtime files, selected model identity,
effective options, and implementation evidence. A single-file factory records
its SHA-256 and size. A module factory records its import spec, distribution
and version when available, and source-file digest when inspectable. These
values enter the resolved runtime provenance and exact-resume request, so a
change invalidates exact reuse.

Do not put credentials in either JSON file. Only environment-variable names
are accepted for secrets. Catalog and runtime files are caller-owned executable
configuration: load only trusted files and code. Benchmark cases and downloaded
benchmark data cannot select or execute a factory. Existing artifact paths and
schemas are unchanged.
