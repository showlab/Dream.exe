# Code structure and reading paths

This guide is for users who have already run the bundled case and now want to
understand, debug, extend, or modify Dream.exe. It maps public commands to their
implementation and gives a short reading path for each common goal.

## One `dream-exe run` request

```text
dream-exe CLI
  └── dream_exe/cli/{router.py,parser.py,bench.py}
      └── dream_exe/bench/runtime/run.py::run_benchmark
          ├── load workspace + validate run spec
          ├── read immutable case/protocol/video inputs
          ├── materialize isolated work bundle
          └── dream_exe/pipeline/runner/
              ├── video2traj + action
              │   └── dream_exe/video2traj/
              ├── simulator execution
              │   └── dream_exe/sim/
              ├── deterministic / optional VLM evaluation
              │   └── dream_exe/evaluation/
              └── validate and publish immutable result
                  ├── dream_exe/artifacts/
                  └── dream_exe/bench/outputs/
```

The CLI is deliberately thin. It parses explicit files and delegates to public
Python functions; pipeline and model code do not discover a benchmark directory
or download a dependency implicitly.

## Package ownership

| Package | Owns | Does not own |
|---|---|---|
| `dream_exe.cli` | Command parsing and dispatch | Scientific algorithms or storage policy |
| `dream_exe.bench` | Workspace, schemas, immutable case access, materialization, run supervision, result publication | Tracking/depth/pose algorithms or simulator internals |
| `dream_exe.pipeline` | Planning and sequencing the canonical stages, resume records, stage-boundary validation | Provider installation or benchmark construction |
| `dream_exe.video2traj` | Simulator-independent video-to-trajectory, gripper, and action computation | Benchmark paths or simulator lifecycle |
| `dream_exe.sim` | Frozen environment restore, controllers, action execution, traces, and task success | Video perception or VLM judging |
| `dream_exe.evaluation` | Trajectory, executability, task, and optional VLM metrics over saved evidence | Generating trajectories or controlling the simulator |
| `dream_exe.generation` | Candidate-video generation interfaces and included providers | Execution or metric definitions |
| `dream_exe.models` | Caller-owned model catalogs and runtime composition | Downloading or silently selecting models |
| `dream_exe.model_assets` | Pinned model manifests, digests, and DVD identity | Network acquisition during runtime |
| `dream_exe.artifacts` | Stable artifact paths and serialization helpers | Experiment selection |

## Read by task

### Follow the public CLI

1. `dream_exe/cli/router.py` calls the root parser and selected handler.
2. `dream_exe/cli/parser.py` shows every public top-level command.
3. `dream_exe/cli/bench.py` implements `configure`, `doctor`, `run`,
   `reproduce`, `generate`, and `bench ...` routing.
4. `dream_exe/bench/runtime/run.py::run_benchmark` binds the workspace, run
   spec, benchmark repository, runtime configuration, and output repository.

### Follow the pipeline

1. `dream_exe/pipeline/runner/single_case.py` owns the formal per-case facade.
2. `dream_exe/pipeline/runner/workflow.py` connects stage inputs and outputs.
3. `dream_exe/pipeline/runner/sequence.py` defines stage order, reuse, failure,
   and stop behavior.
4. `dream_exe/pipeline/stages/` contains the video2traj, simulator,
   task-success, and evaluation adapters.
5. `dream_exe/pipeline/validation/artifacts.py` checks cross-stage artifact
   compatibility before a result is accepted.

### Understand video2traj

Start at `dream_exe/video2traj/runtime/standalone.py`, then follow only the
component relevant to the change:

| Concern | Directory |
|---|---|
| Video loading | `dream_exe/video2traj/media/` |
| Detection, segmentation, sampling | `dream_exe/video2traj/region/` |
| Point tracking | `dream_exe/video2traj/tracking/` |
| DVD, VDA, calibration, and depth provenance | `dream_exe/video2traj/depth/` |
| Camera transforms and 3D lifting | `dream_exe/video2traj/geometry/` |
| Pose estimation | `dream_exe/video2traj/pose/` |
| Trajectory composition | `dream_exe/video2traj/trajectory/` |
| Gripper inference | `dream_exe/video2traj/gripper/` |
| Executable action construction | `dream_exe/video2traj/action/` |

`dream_exe.video2traj` must remain importable without loading benchmark,
RoboCasa, MuJoCo, or VLM runtimes.

### Understand simulator execution

Start at `dream_exe/pipeline/stages/sim.py`, then read:

- `dream_exe/sim/frozen/` for saved-environment restore;
- `dream_exe/sim/robocasa/` for RoboCasa task/runtime integration;
- `dream_exe/sim/execution/` for action input, execution loop, videos, and
  traces;
- `dream_exe/sim/runtime/` for cameras, controllers, contacts, and normalized
  actions;
- `dream_exe/sim/task_success/` for replay-based task evaluation.

### Understand outputs and evaluation

- `dream_exe/artifacts/layout.py` names stable artifact locations.
- `dream_exe/artifacts/action_bundle.py` defines the action bundle boundary.
- `dream_exe/bench/outputs/lifecycle.py` validates and atomically publishes a
  completed result.
- `dream_exe/bench/outputs/results.py` reads and aggregates immutable results.
- `dream_exe/evaluation/trajectory/` and `execution/` implement deterministic
  metrics.
- `dream_exe/evaluation/vlm/` implements optional post-run visual judging.

## Extend a model without changing the pipeline

Most users should not edit internal orchestration. Use the public contracts and
templates in `examples/custom_models/`:

- video generation: [VIDEO_MODELS.md](VIDEO_MODELS.md);
- region, tracking, depth, pose, or VLM: [CUSTOM_MODELS.md](CUSTOM_MODELS.md).

`models.json` registers an implementation. `runtime.json` selects optional
`exec` implementations. The benchmark never names or executes a caller-owned
Python factory.

## Before modifying core behavior

Locate the owning contract and focused tests first:

| Change | Contract/evidence to preserve |
|---|---|
| Workspace or benchmark paths | `dream_exe/bench/data/workspace.py`, repository schemas, protected input tree |
| Depth preset or checkpoint | Depth config/provenance, asset attestation, cache identity, frame alignment |
| Trajectory, gripper, or action | Artifact schemas, trajectory/action length alignment, coordinate frames |
| Simulator execution | Restore inputs, controller mode, checkpoint/dense traces, execution metrics, video |
| Evaluation | Metric definitions, denominators, source evidence, parser output |

Run focused tests for the changed owner, then the public documentation and
package tests. A structural change is not complete merely because imports pass;
the same representative input must still produce schema- and behavior-compatible
artifacts.
