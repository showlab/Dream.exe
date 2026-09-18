# Bundled one-case guide

This directory is a complete one-case benchmark closure for
`rc_cheesybread_ep000001 / Kling3.0 / standard`. It lets a new user run the real
Dream.exe pipeline before downloading the 101-case benchmark.

The root [README](../../README.md#get-started) owns the shortest entry path:
install once, then run one command. This guide begins after that command and
explains what was included, where the result went, and what to try next.

## What is included in Git LFS

- the frozen scene, simulator state, calibrated camera, and exact simulator
  asset closure needed to restore the environment;
- the generation first frame and standard/enhanced prompts;
- ground-truth video, action, and metric depth references;
- the released Kling 3.0 standard generated video;
- the generation, video2traj, action, execution, and evaluation protocols.

The data is under `examples/quickstart/data/`. It is an immutable benchmark
input. The full 101-case data is not read by this example.

Provider source and checkpoints are intentionally not embedded. The one-time
[installation guide](../../INSTALL.md) prepares the required CoTracker, DVD,
GroundingDINO, SAM 2, RoboCasa/RoboSuite, public checkpoints, and the bundled
UID's Dream.exe DVD checkpoint. VDA, FoundationPose, Wan2.2 generation, and a
VLM key are not required.

## What the first run does

The shipped `workspace.json` points to the data in this directory, the
repository-root `external/` and `checkpoints/` directories, and isolated
`.dream-exe/quickstart/` writable roots. Do not edit it and do not run
`dream-exe configure` before the first run.

The shipped `run.json` selects one generated input and the canonical
`video2traj → action → execution → evaluation` stage chain. It runs DVD
with the per-UID Dream.exe checkpoint selected by the benchmark protocol.

## Inspect the result

A successful command prints a report whose top-level `status` is `completed`.
The durable result bundle is written under:

```text
.dream-exe/quickstart/outputs/
├── runs/Kling3.0/standard/run.json
└── experiments/
    └── rc_cheesybread_ep000001/Kling3.0/standard/
        ├── request.json
        ├── resolved_config.json
        ├── result.json
        ├── trajectory/
        ├── action/
        ├── execution/
        └── evaluation/
```

`work/` contains materialization and in-progress state; `outputs/` contains the
completed immutable result. The source data under this directory remains
unchanged.

## Compare three input settings

`run_full.json` evaluates the same task with three inputs:

| Input | Meaning |
|---|---|
| `reference/w_gt_depth` | Ground-truth video plus ground-truth metric depth: oracle input and pipeline upper bound |
| `reference/wo_gt_depth` | Ground-truth video plus DVD depth: isolates the depth-model bottleneck |
| `Kling3.0/standard` | Generated video plus DVD depth: includes generation and depth errors |

Run it in a fresh quickstart output root:

```bash
dream-exe run \
  --workspace examples/quickstart/workspace.json \
  --spec examples/quickstart/run_full.json
```

Do not run `run.json` and `run_full.json` into conflicting existing destinations.
Exact resume accepts only a byte-compatible request and resolved configuration.

## Evaluate saved outputs

Deterministic trajectory, executability, and task metrics can be recomputed from
the saved result without a hosted model:

```bash
dream-exe evaluate trajectory \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 --prompt-variant standard

dream-exe evaluate executability \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 --prompt-variant standard

dream-exe evaluate task \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 --prompt-variant standard
```

Visual VLM scoring is an optional post-run consumer. Continue with
[`VLM.md`](VLM.md) only if those metrics are needed.

## Test one of your videos

Generate an MP4 from
`data/bench/cases/rc_cheesybread_ep000001/generation/first_frame.png` using the
matching prompt in `generation/input.json`, then import and run it:

```bash
dream-exe bench import-video \
  --workspace examples/quickstart/workspace.json \
  --case rc_cheesybread_ep000001 \
  --model MyVideoModel \
  --variant standard \
  --video /absolute/path/generated.mp4 \
  --producer-revision my-model-or-api-version \
  --run
```

The original MP4 is copied into the workspace output boundary. Dream.exe also
creates and validates the 512×512 `preprocessed.mp4` consumed by video2traj;
every user-supplied video goes through this step automatically. Batch import,
the included Wan2.2 adapter, hosted generation APIs, local generators, custom
prompts, and WAM outputs are documented in
[`docs/VIDEO_MODELS.md`](../../docs/VIDEO_MODELS.md).

## Troubleshooting

If the root README command stops during preflight, run:

```bash
dream-exe doctor \
  --workspace examples/quickstart/workspace.json \
  --case rc_cheesybread_ep000001
```

The report distinguishes missing Git LFS data, provider source, model assets,
or runtime imports. Re-run only the corresponding installation check:

- `git lfs pull` for bundled case data;
- `python -m pip check` and `python integrations/setup.py core --repository-root
  . --workspace configs/workspace.json --check` for the environment and
  providers;
- `bash integrations/download_checkpoints.sh --check` for public core model
  assets;
- the Dream.exe DVD download step in `INSTALL.md` for the UID-specific files.

Do not download the full benchmark to repair this one-case path.
