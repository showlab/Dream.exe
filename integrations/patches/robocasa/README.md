# RoboCasa source contract

This directory contains the reviewed source-only delta required to reproduce
the current Dream.exe simulator and task-success semantics. It does not
contain a RoboCasa checkout, simulator assets, episode datasets, generated
scenes, or benchmark results.

The two patches are pinned to public RoboCasa commit
`9a3a78680443734786c9784ab661413edb87067b`. The first patch owns the 13 task
files that produce binary-success and partial-success observations. The
second owns five runtime compatibility files. Their SHA-256 values, exact
target allowlists, and the deterministic post-patch tracked-source-tree digest
are recorded in `manifest.json`.

`python integrations/setup.py robocasa` clones the fixed public commit into the ignored
`external/RoboCasa` directory, applies both patches, rejects any extra tracked
source changes and any untracked path outside the declared
`robocasa/models/assets` data namespace, and verifies the post-patch tracked
tree. Its `--check` mode performs the same verification without network access
or writes. Package installation never runs this setup step.

Affected concrete tasks may require a writable provider tree because upstream
object loading creates temporary XML beside an asset descriptor. The workspace
therefore binds the verified checkout as `setup_path` and a detached runtime
copy as `path`. The explicit setup command creates and verifies both:

```bash
python integrations/setup.py robocasa --workspace configs/workspace.json
python integrations/setup.py --check robocasa --workspace configs/workspace.json
```

The runtime copy dereferences asset symlinks, rejects shared file identities,
and carries a path-free content receipt. It is ignored scratch and is never a
source, asset, or redistribution authority.

The tracked-tree digest covers each Git-tracked stage-zero path's index mode,
repository-relative name, byte length, and current bytes in sorted Git index
order. Untracked assets and datasets are deliberately excluded and must be
provided and attested separately. RoboCasa assets may populate the declared
asset namespace without changing the source digest; they are not verified by
this source manifest. Episode datasets remain outside the checkout.
