# Input-specific experiment configuration

A case protocol may provide `input_configs` to keep settings for different
videos and reference inputs independent. Each entry has two fields:

```json
{
  "input": {
    "kind": "reference",
    "model_id": null,
    "prompt_variant": null,
    "reference_id": "w_gt_depth"
  },
  "values": {
    "video2traj": {},
    "action": {},
    "execution": {}
  }
}
```

For generated inputs, use `kind: "generated"`, explicit `model_id` and
`prompt_variant`, and `reference_id: null`. Identities must be unique. Once
`input_configs` is present, every requested video/reference must have an exact
entry: missing entries fail instead of falling back to another input's settings.

For these three stages, the selected entry is an alternative configuration
owner, not another override layer. Packaged defaults fill unspecified fields;
shared protocol settings and case route settings do not leak into the entry.
The evaluation stage retains its existing protocol. The `evaluation_oracle`
trajectory must select the exact `w_gt_depth` input configuration, so the
comparison reference and GT benchmark input use the same extraction settings.
Existing protocols without `input_configs` retain the route-based behavior.

Run-owned paths and the GT depth input switch remain run-owned. Input settings
cannot override them. All entries participate in the case protocol fingerprint;
resolved fields record their `case-input:` owner.

## Optional final depth smoothing

`video2traj.depth.base.final_smooth` supports `enabled` (boolean),
`bilateral_mode` (`"on"` or `"off"`), and positive finite `sigma_r`.
Defaults are `false`, `"off"`, and `0.02`. Historical nullable preset settings
must be resolved explicitly when importing configurations.

When enabled, smoothing follows base calibration, applies only to the first-frame
ROI (or the full frame if no ROI exists), and requires metric or calibrated
depth. It uses a 5-pixel OpenCV bilateral filter with spatial sigma 5 and
depth scaling by `sigma_r`. GT depth skips the entire base postprocess,
including final smoothing, even when smoothing is requested.

## Historical compatibility inputs

Paper-era trajectories and actions are never treated as an automatic cache or
as a replacement for the default fresh pipeline. A compatibility replay must
use a `dream-exe.historical-compatibility-manifest.v1` document and explicitly
run:

```text
dream-exe bench verify-compatibility-input \
  --workspace configs/workspace.json \
  --manifest /absolute/path/manifest.json \
  --artifact-root /absolute/path/archive-root
```

The manifest is a closed population: it includes every UID in the selected
collection, including explicit `unavailable` rows. Available rows bind the
canonical case, environment, reference, and reference-video identities plus
the relative path, byte size, and SHA256 of the historical EEF trajectory,
object trajectory, and action. This prevents outcome-selected imports, silent
denominator changes, path traversal, and stale artifact reuse. The verifier
does not relax fresh pose/depth cache identity checks.

Historical task-success replay has a second, independent configuration layer:
the paper-era task evaluator did not reuse the simulator execution settings
that produced the Table 3 trajectory metrics. Build that configuration with
`build_historical_task_evaluator_config`, using the archived task artifact's
controller fields plus the complete revision-pinned evaluator defaults. The
builder rejects incomplete defaults and never infers Table 4 evaluator
settings from a Table 3 execution config.
