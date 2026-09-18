# Configuration ownership

Dream.exe separates machine paths, experiment selection, secrets, optional model
registration, and benchmark protocol. Each value has one owner; later layers do
not silently override the same field.

## The five user-facing files

| File | Owner | Controls | Does not control |
|---|---|---|---|
| `workspace.json` | Machine/workspace owner | Data, provider, checkpoint, work, and output locations | Cases, stages, model choice, or secrets |
| `run.json` | Experiment owner | Cases, input videos, stage chain, seed, resume, and destination | Machine paths, model implementation, or benchmark protocol |
| `credentials.local.json` | Local user | Private VLM endpoints and API keys | Execution models or reproducible protocol |
| `models.json` | Model integrator | Available caller-provided video, exec, and eval implementations | Which exec implementation is active in a run |
| `runtime.json` | Experiment owner | Optional `exec` model composition for region, tracking, depth, and pose | Video-generator or VLM selection |

The bundled example uses `examples/quickstart/workspace.json` and
`examples/quickstart/run.json` unchanged. A standard full-benchmark download
uses `configs/workspace.json` unchanged.

## `workspace.json`: where files live

Every relative path resolves from the directory containing the workspace file,
not from the current shell directory. The released `configs/workspace.json`
therefore resolves to:

```text
Dream.exe/
├── configs/workspace.json
├── data/{bench,results}/
├── external/
├── checkpoints/
└── .dream-exe/{work,outputs,archive}/
```

### Root keys

| Key | Default resolved location | Edit when |
|---|---|---|
| `roots.bench` | `Dream.exe/data/bench` | Full benchmark inputs live elsewhere |
| `roots.published_results` | `Dream.exe/data/results` | Released input videos/results live elsewhere |
| `roots.outputs` | `Dream.exe/.dream-exe/outputs` | Durable generated results should use another writable disk |
| `roots.work` | `Dream.exe/.dream-exe/work` | Temporary materialization should use another writable disk |
| `roots.archive` | `Dream.exe/.dream-exe/archive` | Archived run state should use another writable disk |
| `roots.external` | `Dream.exe/external` | Pinned provider checkouts live elsewhere |
| `roots.checkpoints` | `Dream.exe/checkpoints` | Model assets live elsewhere |

All seven roots must be pairwise disjoint. `dream-exe configure` validates the
workspace and creates only writable local roots; it never writes benchmark
inputs or published results.

For the default paths, use the shipped file. If only full benchmark storage is
external, make a caller-owned copy and change only `roots.bench` and
`roots.published_results`, as shown in the
[benchmark guide](BENCHMARK.md#4-bind-and-validate-the-full-benchmark).

The default DVD LoRA checkpoint root is the workspace binding
`bindings.checkpoints.dvd_assets.path`, which resolves to `checkpoints/`; the
runtime then looks below `DVD/lora/shared/` or
`DVD/lora/specific/<uid>/` according to the selected preset. If model assets
live on another disk, update `roots.checkpoints` and the relevant
`bindings.checkpoints.*.path` entries in a caller-owned workspace copy. Moving
only the files without changing those bindings leaves the default resolver
pointing at the original `checkpoints/` directory.

### Source and checkpoint bindings

`bindings.sources.<name>` binds the runtime checkout, the setup checkout, and
its provenance manifest. The default execution providers are `robocasa`,
`grounding_dino`, `cotracker`, and `dvd`; `vda` and `wan22` remain unused unless
an experiment selects them.

`bindings.checkpoints.<name>` binds concrete asset files or roots:

| Binding | Used by | Required for bundled case |
|---|---|---|
| `grounding_dino` | Text-guided detector | Yes |
| `bert_base_uncased` | GroundingDINO encoder | Yes |
| `sam2` | Segmentation | Yes |
| `cotracker` | Point tracking | Yes |
| `dvd_assets` | Upstream DVD plus Dream.exe DVD families | Yes |
| `foundationpose` | Optional pose backend | No |
| `wan22` | Optional video generator | No |

Bindings declare locations; they never select a model merely because files are
present.

## `run.json`: what experiment runs

A run spec selects cases, candidate/reference inputs, the canonical stage chain,
seed, resume/retry policy, artifact level, and destination run ID. Runnable
examples live under `examples/`; released full-benchmark specs are downloaded
under `data/results/runs/`.

The run spec intentionally contains no local filesystem roots, API keys, Python
factories, or algorithm internals. The README's bundled command uses
`examples/quickstart/run.json`; `run_full.json` is an optional three-input
comparison described by the bundled-case guide.

## `credentials.local.json`: VLM secrets

Copy the template only when visual VLM evaluation is needed:

```bash
cp configs/credentials.template.json configs/credentials.local.json
chmod 600 configs/credentials.local.json
```

Fill only the selected endpoint and API-key fields, then pass the file with
`--credentials`. It is ignored by Git and excluded from distributions. VLM
evaluation consumes saved artifacts and does not affect trajectory extraction,
depth estimation, or simulator execution.

## `models.json` and `runtime.json`: optional model implementations

Start from `examples/custom_models/models.example.json` and
`examples/custom_models/runtime.example.json` only when integrating a new
backend.

`models.json` is a categorized catalog:

| Category | Allowed kinds | Selected by |
|---|---|---|
| `video_gen` | `video_generation` | `dream-exe generate` |
| `exec` | `region_detector`, `region_segmenter`, `tracking`, `depth`, `pose` | `runtime.json` during `run` or standalone video2traj |
| `eval` | `vlm` | Evaluation commands after artifacts exist |

`runtime.json` has `category: "exec"` and selects only region, tracking, depth,
and pose implementations. A partial runtime file replaces only the named
components. Neither file may contain credentials. Both are trusted
caller-provided executable configuration and are passed explicitly with
`--models-config` and `--runtime-config`.

See [VIDEO_MODELS.md](VIDEO_MODELS.md) for video generation and
[CUSTOM_MODELS.md](CUSTOM_MODELS.md) for exec/eval contracts.

## Where built-in depth selection lives

For formal benchmark runs, the built-in depth preset is a protocol value at
`video2traj.depth.model`. It is compiled from benchmark-wide and per-case
protocol files; reproduction users must not edit it. A different preset is a
new experiment. For standalone video2traj, the same key belongs in the
caller-owned pipeline JSON.

The current built-in choices for the official DVD model, Dream.exe LoRAs, and
VDA are listed in
[MODEL_ASSETS.md](MODEL_ASSETS.md#where-depth-model-selection-lives). A custom
depth implementation instead uses `models.json` plus `runtime.json`.

## Benchmark-owned values are immutable inputs

The downloaded benchmark owns the frozen scene, camera, generation prompt,
per-case routing, algorithm protocol, depth-preset choice, execution behavior,
and evaluation rubrics:

```text
data/bench/cases/<uid>/case.json
data/bench/cases/<uid>/protocol.json
data/bench/protocol/{generation,video2traj,action,execution,evaluation}.json
data/bench/collections.json
```

Users do not edit these files for reproduction. A changed protocol must be
stored and reported as a new experiment.

## Packaged implementation defaults

Stable implementation defaults live beside their consumer:

```text
dream_exe/video2traj/configs/
dream_exe/model_assets/configs/
```

They are shipped in wheels and source distributions and are intentionally
absent from root `configs/`. Python files named `config.py` are loaders,
validators, or typed adapters; they are code rather than user configuration.

Other repository JSON, YAML, and TOML files have infrastructure ownership:
`integrations/sources.json` and patch manifests pin external provenance,
`pyproject.toml` defines the package, and `.github/workflows/public-ci.yml`
defines release checks. No committed runtime JSON may contain a machine-local
absolute path.
