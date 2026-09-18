# Custom-model templates

This directory contains one categorized model catalog, one exec-only runtime
composition, and six minimal Python templates. The catalog groups models as:

- `video_gen`: local and polling-API video generation;
- `exec`: region, tracking, depth, and pose;
- `eval`: VLM judges that consume saved artifacts.

Start with a static check:

```bash
dream-exe models list \
  --models-config examples/custom_models/models.example.json \
  --category exec
dream-exe models check \
  --models-config examples/custom_models/models.example.json \
  --model example_tracker \
  --category exec --kind tracking
```

The templates construct and pass static contract checks. Their inference hooks
raise `NotImplementedError` until you connect a real model. Keep the catalog
beside the template files so each `./file.py:Class` factory stays portable.

Use [the video-model guide](../../docs/VIDEO_MODELS.md) only for `video_gen`
models or WAM-produced videos. Use the separate
[exec/eval model guide](../../docs/CUSTOM_MODELS.md) for `exec` region,
tracking, depth, and pose contracts and for `eval` VLM contracts.
