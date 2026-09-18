# Shipped user configuration

This directory contains the two configuration files shipped for ordinary use.
The canonical explanation of path ownership, run specs, credentials, model
catalogs, runtime composition, and benchmark-owned protocol is
[`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md).

| File | Use it when | Normal action |
|---|---|---|
| `workspace.json` | Running the downloaded 101-case benchmark | Use unchanged when data is under repository-root `data/`; otherwise make a caller-owned copy and change only the required paths |
| `credentials.template.json` | Enabling optional hosted VLM evaluation | Copy to ignored `credentials.local.json`, set mode `0600`, and fill only the selected endpoint/key |

The first bundled run does **not** use either file above. It uses the ready-made
`examples/quickstart/workspace.json` directly; do not edit or configure that
workspace.

Full benchmark download, path setup, and validation are kept together in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md). Optional video-generator setup is in
[`docs/VIDEO_MODELS.md`](../docs/VIDEO_MODELS.md), and optional exec/eval model
composition is in [`docs/CUSTOM_MODELS.md`](../docs/CUSTOM_MODELS.md).
