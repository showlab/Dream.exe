# README teaser

- `dream-exe-teaser.gif`: 1920 × 960, 18 GT demonstrations in a 6 × 3 grid, 12 fps, an approximately 8-second infinite loop. The repository README embeds this file and links it to the still image.
- `dream-exe-teaser.png`: 1920 × 960 still, sampled one third of the way through the shared cycle.
- `sources.json`: row-major case IDs, benchmark-relative GT paths, source SHA-256 hashes, original durations, and kitchen layout/style IDs.

The scenes were visually selected from the active 101-case collection for varied colors, materials, viewpoints, appliances and operations. All 18 layout/style pairs are distinct; this is a curated display, not a claim that there are only 18 unique benchmark scenes. Only ground-truth reference videos are used.

Each full demonstration is resampled to the same cycle duration, so all tiles start and loop together. Speeds differ from the original recordings. Frames keep their original square composition; no cropping, reversing, or generated imagery is used. GIF frame timing uses centiseconds and approximates 12 fps.

To rebuild from the repository's full benchmark installation:

```bash
python3 assets/teaser/build_teaser.py
```

Requires Pillow, NumPy, FFmpeg/FFprobe, URW P052 Bold Italic, Ubuntu Mono Regular and Ubuntu Medium. Font paths are defined in the build script; fonts are rasterized into the outputs and are not needed to view them. The builder reads the benchmark and writes only teaser assets; intermediate frames live in a temporary directory.
