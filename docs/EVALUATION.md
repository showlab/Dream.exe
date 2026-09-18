# Evaluation

Dream.exe separates evaluation into three independent consumers of saved
artifacts. None of them modifies the benchmark.

These are evaluation *families*, not model-catalog categories. The replaceable
VLM judge is catalog category `eval` and kind `vlm`. Depth estimation belongs
to `exec`; evaluation may read its saved depth output but never selects or
runs a depth model.

## Layout

| Area | Inputs | Published metrics |
|---|---|---|
| `trajectory` | predicted and ground-truth trajectories | HSD, DYN, NDTW |
| `execution` | action execution traces and task observations | E-SR, tracking nDTW, Pos95, Rot95, smoothness, SR-B, SR-P, Rel, Place, Art, Core |
| `vlm` | generated video plus benchmark-owned prompts | subject stability, physical plausibility, task adherence |

Generated videos are evaluated objects, not benchmark inputs. Prepared frame
grids, raw VLM replies, saved depth produced by `exec`, trajectories, actions,
execution videos, and metrics are written under workspace `work` or `outputs`.
Published JSON stores only relative artifact paths.

## VLM judges

The benchmark declares two judges in `bench/protocol/evaluation.json`:
`gemini-3-pro` and `qwen3-vl-plus`. Their endpoint and credential environment
variable names are declared there; secrets are never stored in the benchmark,
results, or tracked source files.

The released benchmark configuration uses hosted models, so exact paper
reproduction uses the corresponding APIs. The provider adapter speaks the
OpenAI-compatible API shape and can also connect to a local compatible server.
A local model or a different checkpoint is a valid new evaluation, but it is a
different judge and must not be reported as exact paper reproduction.

The bundled execution install does not install the optional VLM client. Add it
and create a private local credential file only when visual evaluation is
needed:

```bash
conda activate dream-exe
python -m pip install -e ".[vlm]"
cp configs/credentials.template.json configs/credentials.local.json
chmod 600 configs/credentials.local.json
```

Fill the matching `base_url` and `api_key` fields. The file is ignored by Git,
is excluded from packages, and is never copied into results. Environment
variables remain supported and take precedence, which is useful for managed
secret stores and CI.

### Evaluate one case

VLM evaluation supports the same single-case selection as execution and the
deterministic evaluators. The smallest runnable example uses the released
successful `rc_cheesybread_ep000001 / Kling3.0 / standard` candidate:

```bash
dream-exe evaluate visual \
  --credentials configs/credentials.local.json \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 \
  --prompt-variant standard \
  --judge qwen3-vl-plus \
  --rubric task_adherence
```

Alternatively, set `DREAM_EXE_GEMINI_BASE_URL` and
`DREAM_EXE_GEMINI_API_KEY` (or the Qwen equivalents). Set
`DREAM_EXE_CREDENTIALS_FILE` to avoid repeating `--credentials`.

The command evaluates only the UID passed to `--case`. It resolves the
candidate video from workspace outputs first and downloaded published results
second. It loads case metadata, the standard or enhanced generation prompt,
the benchmark-owned rubric template, sampling settings, and judge identity
directly from the benchmark. Work products are written to:

```text
work/evaluation/vlm/<case>/<model>/<variant>/<judge>/<rubric>/
├── media/
├── predictions/
└── results.csv
```

Use `--rubric all --judge both` to run the three rubrics with both declared
judges. The CLI computes the protocol-declared arithmetic mean, marks an item
`not_evaluated` if either judge is missing or invalid, saves every case report,
and still writes a one-case aggregate beside the case report.

### Evaluate the complete benchmark

After the full benchmark and matching candidate videos are available, change
only the workspace and scope:

```bash
dream-exe evaluate visual \
  --credentials configs/credentials.local.json \
  --workspace configs/workspace.json \
  --scope all \
  --candidate-model <model> \
  --prompt-variant standard \
  --judge both \
  --rubric all
```

Collection scope evaluates every UID declared by the benchmark collection and
writes both per-case reports and an overall aggregate.

## Public evaluation CLI

All four evaluation families support one case or a complete collection:

```bash
dream-exe evaluate visual ...
dream-exe evaluate trajectory ...
dream-exe evaluate executability ...
dream-exe evaluate task ...
dream-exe evaluate all ...
```

Use `--scope one --case <uid>` for a single case or `--scope all` for the
benchmark. Table 2 trajectory evaluation is deterministic
HSD/DYN/NDTW over EEF vis, EEF tcp, and OBJ; it is not VLM evaluation. Table 3
reports E-SR, nDTW, Pos95, Rot95, and Smth. Table 4 reports SR-B, SR-P, Rel,
Place, Art, and Core with independent valid-value denominators.

Reports are saved below:

```text
outputs/evaluations/<model>/<standard|enhanced>/
├── cases/<uid>/
└── aggregate/
```

The CLI prints the aggregate first. Saved source references use workspace-root
relative identities and never embed local absolute paths.

## Parser contract

Physical plausibility and task adherence responses must be exact JSON objects
with `score` and `reason`. `score` must be an integer from 1 through 5 and
`reason` must be non-empty text. Out-of-range, floating-point, boolean,
missing, or extra-field replies are retained as raw evidence and marked
`parse_error`; they are not included in scientific aggregates.
