# Visual VLM evaluation

The paper uses VLMs only for visual quality. Table 2 trajectory similarity is
deterministic HSD/DYN/NDTW code and is run with `dream-exe evaluate
trajectory`; it is not a trajectory-VLM score.

In the replaceable-model catalog, a VLM is category `eval`, kind `vlm`. It
consumes already saved videos and other artifacts; it never supplies depth,
trajectory, action, or simulator inputs.

The three visual rubrics are:

- `subject_stability`: robot and manipulated-object stability, score 1–15;
- `physical_plausibility`: physical realism, score 1–5;
- `task_adherence`: instruction adherence, score 1–5.

The VLM judges declared by the released protocol are `gemini-3-pro` and
`qwen3-vl-plus`. Each judges every item independently. `--judge both` takes the
arithmetic mean only when both scores are valid, which matches the paper; it
never averages a partial judge set.

## Credentials

The bundled execution install does not install the optional VLM client. Add it
to the main environment only when visual evaluation is needed:

```bash
conda activate dream-exe
python -m pip install -e ".[vlm]"
cp configs/credentials.template.json configs/credentials.local.json
chmod 600 configs/credentials.local.json
```

Fill the selected profile's `base_url` and `api_key`. The local file is ignored
by Git and never copied into benchmark or result JSON.

## One case, one judge, one rubric

```bash
dream-exe evaluate visual \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 --prompt-variant standard \
  --judge qwen3-vl-plus --rubric task_adherence \
  --credentials configs/credentials.local.json
```

## One case, both VLM judges, all visual rubrics

```bash
dream-exe evaluate visual \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 --prompt-variant standard \
  --judge both --rubric all \
  --credentials configs/credentials.local.json
```

## All four evaluation families

This runs visual quality, Table 2 trajectory similarity, Table 3 trajectory
executability, and Table 4 task-level execution. The pipeline result must
already exist for the last three families.

```bash
dream-exe evaluate all \
  --workspace examples/quickstart/workspace.json \
  --scope one --case rc_cheesybread_ep000001 \
  --candidate-model Kling3.0 --prompt-variant standard \
  --judge both --rubric all \
  --credentials configs/credentials.local.json
```

For the full release benchmark, replace the workspace and selection with:

```text
--workspace configs/workspace.json --scope all
```

Every case keeps its own report and raw provider reply. Full-benchmark runs also
write an overall aggregate. The first CLI section is the aggregate, followed
by selection and portable report paths.
