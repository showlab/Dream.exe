# Full benchmark installation and reproduction

This guide begins **after** the bundled one-case example succeeds. It owns the
additional downloads, path setup, and commands needed for the complete 101-case
benchmark. None of these steps is required for the README quickstart. Benchmark
inputs and released result inputs are separate artifacts.

## Reproduction contract: frozen video versus regeneration

This release distinguishes two different workflows. They must not be mixed in
one result table.

### A. Exact reproduction from a frozen generated video

Use this workflow when checking the released Dream.exe result. The generated
MP4 is treated as an immutable input; Dream.exe runs the configured
`video2traj → action → execution → evaluation` stages against it. The run must
also use the matching preprocessed-video identity, published run spec, bench
case, checkpoint manifest, and provider versions. No video-generation service
is called, and changing the trajectory/depth/evaluation configuration creates a
new experiment rather than a reproduction. The required generated-video and
run-spec bundle is the separate `data/results/` asset described in section 5.

If a generated MP4 is unavailable, a GT video, execution video, opposite
model/variant, or newly generated replacement cannot be silently promoted to
the frozen input. Record that task as unavailable until an independently
verified source binds the exact UID, generator model, variant, bytes, and
media decode.

### B. Regeneration from benchmark prompts

Use this workflow only when intentionally producing a new video from the
frozen benchmark first frame, prompt, and protocol. Select and record the
video-generation backend, model revision, provider/API version, seed or other
sampling settings, output media hash, and the new experiment identity. Store
the regenerated video and downstream artifacts under a caller-owned experiment
root; it is not interchangeable with the released `data/results/` input and
must not be merged into the exact-reproduction table without an explicit new
release decision.

The current package promises inference-time reproduction from supplied frozen
videos and released checkpoints. It does not promise training reproduction:
training code, training data provenance, optimizer state, and complete training
checkpoints are not part of this benchmark release. A training rerun is a
separate research experiment with its own manifest.

## What the full benchmark adds

| Additional input | Destination with the default workspace | Purpose |
|---|---|---|
| 101 frozen benchmark cases | `data/bench/` | Scenes, simulator state, cameras, prompts, protocols, and ground-truth references |
| Complete Dream.exe DVD family | `checkpoints/DVD/lora/shared/` and `checkpoints/DVD/lora/specific/<uid>/` | Depth checkpoints selected by the released routes |
| RoboCasa kitchen assets | `external/RoboCasa/robocasa/models/assets/` (then the detached runtime copy) | Untracked scenes, fixtures, textures, and objects loaded by the simulator |

Local recomputed trajectories, actions, execution videos, and metrics continue
to go under `.dream-exe/`; the downloaded publication is never overwritten.

## 1. Complete the base installation

Finish [INSTALL.md](../INSTALL.md) and successfully run the bundled case first.
That establishes the environment, providers, public core model assets, one
Dream.exe DVD checkpoint, and the real end-to-end pipeline before any large
benchmark download.

## 2. Download the complete Dream.exe DVD family

The minimal install fetched only the bundled UID. Download the shared and
case-specific Dream.exe LoRA checkpoints for the full benchmark:

```bash
hf download kaimingyang/DVD_for_Dream.exe \
  --include "DVD/lora/shared/*" "DVD/lora/specific/*" \
  --local-dir checkpoints
```

Run from the repository root. The resulting directories are
`checkpoints/DVD/lora/shared/` and `checkpoints/DVD/lora/specific/<uid>/`,
each containing `model.safetensors` and `model_config.yaml`.
The upstream DVD base and its Wan2.1 runtime assets were installed in the base
installation. The default generated-video route uses our case-specific DVD
LoRA fine-tuned on the benchmark data.

Benchmark membership is defined by the dataset's `benchmark` collection.
`dream-exe doctor` validates the selected
checkpoint against the packaged manifest before execution.

If checkpoints live elsewhere, update `roots.checkpoints` and the checkpoint
bindings in a caller-owned workspace copy. Keep the `DVD/lora/` relative
layout shown above.

## 3. Download the benchmark data

From the repository root, download the public
[kaimingyang/Dream.exe dataset](https://huggingface.co/datasets/kaimingyang/Dream.exe)
directly to `data/`:

```bash
hf download kaimingyang/Dream.exe \
  --repo-type dataset \
  --include 'bench/**' 'results/runs/**' 'results/videos/**' 'results/experiments/**' \
  --local-dir data
```

This downloads the benchmark, frozen video inputs, run descriptors, and
published results into the default repository-local paths. Publication is in
progress; use these commands after the complete release is available.
For reproducibility, add `--revision <dataset-commit>` to pin the download.

The dataset contains the immutable benchmark input tree:

```text
data/
└── bench/
    ├── collections.json
    ├── protocol/
    └── cases/<uid>/
        ├── env/                 # frozen scene, state, and camera
        ├── generation/          # first frame and prompts
        ├── protocol.json        # rare per-case protocol values
        └── references/          # GT video, action, and depth
```

The `bench/` subtree contains immutable benchmark inputs. Generated videos,
run descriptors, and published experiment outputs live separately under
`data/results/`, as described in section 5.
Do not edit benchmark files for reproduction. A changed case or protocol is a
new experiment, not the released benchmark.

### 3.1 Acquire the RoboCasa kitchen asset namespace

The pinned RoboCasa source checkout and its untracked kitchen assets are
separate inputs. `integrations/setup.py` verifies the source revision but does
not download simulator assets; the Git source manifest intentionally excludes
`robocasa/models/assets`. The released integration is pinned to
`9a3a78680443734786c9784ab661413edb87067b` and the upstream asset script
currently downloads the asset bundles from the RoboCasa links. Review the
asset terms (the RoboCasa repository declares CC BY 4.0 for assets and
datasets) before downloading.

From the repository root, after the core provider setup in [INSTALL.md](../INSTALL.md):

```bash
python integrations/setup.py robocasa \
  --repository-root . \
  --workspace configs/workspace.json
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets --type all
```

The upstream script is interactive and downloads roughly 10 GB. Use one or
more explicit `--type` values when a smaller, reviewed asset set is intended;
the available names are `tex`, `tex_generative`, `fixtures_lw`,
`objs_objaverse`, `objs_aigen`, and `objs_lw`. It writes only below
`external/RoboCasa/robocasa/models/assets/`. Re-run the non-`--check` provider
setup after the download so the detached runtime copy is refreshed, then run
the zero-write checks:

```bash
python integrations/setup.py robocasa \
  --repository-root . \
  --workspace configs/workspace.json
python integrations/setup.py robocasa \
  --repository-root . \
  --workspace configs/workspace.json \
  --check
dream-exe doctor --workspace configs/workspace.json --bench-only
```

Do not treat a successful source check with an empty or partial asset
namespace as a full-benchmark acceptance. The simulator may select fixture or
object XMLs dynamically during restore and execution, so the final asset
bundle must be validated by a representative end-to-end run and recorded with
its source, size, content hashes, and license evidence. This acquisition step
was not performed automatically by the migration work.

## 4. Bind and validate the full benchmark

The shipped `configs/workspace.json` already resolves to repository-root
`data/bench`, `data/results`, `external`, `checkpoints`, and `.dream-exe`. With
the default download locations above, do not edit it:

```bash
dream-exe configure --workspace configs/workspace.json
dream-exe doctor --workspace configs/workspace.json --bench-only
dream-exe doctor \
  --workspace configs/workspace.json \
  --case rc_cheesybread_ep000001
```

`configure` creates only the writable `work`, `outputs`, and `archive` roots. It
does not create or modify benchmark inputs or published results.

If the downloaded data lives elsewhere, use a caller-owned copy of the
workspace and change only `roots.bench` and `roots.published_results`:

```json
{
  "roots": {
    "bench": "/absolute/location/bench",
    "published_results": "/absolute/location/results"
  }
}
```

Keep the remaining keys from the shipped workspace unchanged, then pass the
caller-owned file to every command. The exact meaning of all path and binding
keys is in the [configuration reference](CONFIGURATION.md#workspacejson-where-files-live).

## 5. Understand the downloaded result bundles

Released video inputs, run descriptors, and output bundles are stored under
`data/results/`:

```text
data/results/
├── videos/<uid>/<model>/<variant>/
├── runs/<model>/<variant>/run.json
└── experiments/<uid>/<model>/<variant>/
```

The download in section 3 includes these trees. `reproduce` reads the run
descriptors and frozen video bundles, including each `video.json` and any
declared preprocessed video. It recomputes trajectories and execution outputs
under `.dream-exe/`; previously published `experiments/` are not reused as
new results. To inspect or aggregate published results, preserve each complete
experiment bundle: `result.json`, `request.json`, the bound
`resolved_config.json`, and every artifact listed in `result.json`.

## 6. Reproduce one released run first

Preview the immutable run matrix without starting GPU work:

```bash
dream-exe reproduce \
  --workspace configs/workspace.json \
  --dry-run
```

The dry run must list the expected runs. It only lists descriptors; it does
not verify videos, providers, or checkpoints, and an empty list is not a
successful release check. Use an ID printed by the dry run to execute one run:

```bash
dream-exe reproduce \
  --workspace configs/workspace.json \
  --run-id <ID_FROM_DRY_RUN> \
  --stop-on-failure
```

Repeat `--run-id` to select several released runs. This progression catches an
environment or asset problem before committing to the full matrix.

## 7. Reproduce the full matrix

After one released run succeeds:

```bash
dream-exe reproduce --workspace configs/workspace.json
```

Exact resume is enabled inside the local output root. An interrupted run can be
continued only when its run spec, resolved configuration, providers, model
identities, and protected inputs still match.

Videos unavailable because of generation-model compliance restrictions are
recorded as `input_missing`; affected runs return `partial`. This availability
status is separate from execution success.

## 8. Continue with your own model or analysis

- To evaluate externally generated MP4s or connect a video/WAM backend, follow
  [VIDEO_MODELS.md](VIDEO_MODELS.md).
- To calculate deterministic or VLM metrics from saved outputs, follow
  [EVALUATION.md](EVALUATION.md).
- To change depth, tracking, region, pose, or VLM implementations, follow
  [CUSTOM_MODELS.md](CUSTOM_MODELS.md). Such a run is a new experiment rather
  than a reproduction of the frozen release.
