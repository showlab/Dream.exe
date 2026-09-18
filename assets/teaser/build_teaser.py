#!/usr/bin/env python3
"""Rebuild the README collage from immutable benchmark GT videos.
Requires Python with Pillow/numpy, ffmpeg/ffprobe and the fonts below.
Run from any directory: python3 assets/teaser/build_teaser.py
"""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BENCH = ROOT / 'data/bench/cases'
UIDS = [
    'rc_breadandcheese_ep000001', 'rc_cheesybread_ep000016',
    'rc_openblenderlid_ep000017', 'rc_coffeeservemug_ep000000',
    'rc_closeoven_ep000000', 'rc_turnsinkspout_ep000006',
    'rc_closedrawer_ep000004', 'rc_openstandmixerhead_ep000004',
    'rc_makeicedcoffee_ep000008', 'rc_slideovenrack_ep000026',
    'rc_slidetoasterovenrack_ep000005', 'rc_cheesybread_ep000001',
    'rc_packdessert_ep000031', 'rc_slidetoasterovenrack_ep000052',
    'rc_turnsinkspout_ep000000', 'rc_closestandmixerhead_ep000012',
    'rc_turnontoaster_ep000002', 'rc_breadsetupslicing_ep000004',
]
FONTS = {
    'dream': '/usr/share/fonts/opentype/urw-base35/P052-BoldItalic.otf',
    'exe': '/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf',
    'subtitle': '/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf',
}
TILE, COLS, ROWS = 320, 6, 3
WIDTH, HEIGHT = TILE * COLS, TILE * ROWS
FPS, SECONDS = 12, 8
COUNT = FPS * SECONDS
TITLE = 'Can Video Generation Models Dream Executable Robot Manipulation?'


def overlay():
    # Smooth central scrim preserves the scenes while giving the title contrast.
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    alpha = 32 + 133 * np.exp(-((y - HEIGHT * .51) / (HEIGHT * .20)) ** 2) * (.65 + .35 * np.exp(-((x-WIDTH/2)/(WIDTH*.45))**2))
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = (6, 13, 20)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    layer = Image.fromarray(rgba)
    d = ImageDraw.Draw(layer)
    scale = WIDTH / 1536
    dream = ImageFont.truetype(FONTS['dream'], round(158 * scale))
    exe = ImageFont.truetype(FONTS['exe'], round(122 * scale))
    a = d.textlength('Dream', font=dream)
    b = d.textlength('.exe', font=exe)
    left = (WIDTH-a-b-9*scale)/2
    d.text((left, 412*scale), 'Dream', font=dream, anchor='ls', fill='#fff8ed', stroke_width=1)
    d.text((left+a+9*scale, 412*scale), '.exe', font=exe, anchor='ls', fill='#fff8ed')
    subtitle = ImageFont.truetype(FONTS['subtitle'], round(29 * scale))
    d.text((WIDTH/2, 468*scale), TITLE, font=subtitle, anchor='mm', fill='#fff8ed')
    return layer


def main():
    clips, records = [], []
    for uid in UIDS:
        path = BENCH / uid / 'references/video/gt.mp4'
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,nb_frames', '-of', 'json', str(path)]))['streams'][0]
        duration = float(probe['duration'])
        # Resample whole demonstrations to a common duration; no reversal or crop.
        raw = subprocess.check_output([
            'ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
            '-vf', f'setpts={SECONDS/duration}*(PTS-STARTPTS),fps={FPS},scale={TILE}:{TILE}:flags=lanczos,tpad=stop_mode=clone:stop_duration=1',
            '-frames:v', str(COUNT), '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(COUNT, TILE, TILE, 3)
        clips.append(arr)
        metadata = json.loads((BENCH / uid / 'env/task_runtime.json').read_text())
        records.append(dict(uid=uid, source=path.relative_to(ROOT).as_posix(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            duration_seconds=duration, layout_id=metadata['layout_id'],
                            style_id=metadata['style_id'], row=len(records)//COLS,
                            column=len(records)%COLS))
        print(f'Decoded {uid}', flush=True)
    layer = overlay()
    with tempfile.TemporaryDirectory(prefix='dream-teaser-') as work:
        work = Path(work)
        for frame in range(COUNT):
            canvas = Image.new('RGB', (WIDTH, HEIGHT))
            for i, clip in enumerate(clips):
                canvas.paste(Image.fromarray(clip[frame]), (i % COLS*TILE, i//COLS*TILE))
            canvas = Image.alpha_composite(canvas.convert('RGBA'), layer).convert('RGB')
            if frame == COUNT//3:
                canvas.save(OUT / 'dream-exe-teaser.png', optimize=True)
            canvas.save(work / f'{frame:03}.png')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '1', '-framerate', str(FPS),
                        '-i', str(work/'%03d.png'), '-filter_complex_threads', '1',
                        '-filter_complex', '[0:v]split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a',
                        '-loop', '0', str(OUT/'dream-exe-teaser.gif')], check=True)
    manifest = dict(title=TITLE, grid=dict(columns=COLS, rows=ROWS),
                    gif=dict(width=WIDTH, height=HEIGHT, fps=FPS, frames=COUNT, seconds=SECONDS, loop=True),
                    static=dict(width=WIDTH, height=HEIGHT, frame=COUNT//3),
                    timing='Full GT demonstrations resampled to a shared 8-second cycle; playback speeds differ. All begin together.',
                    selection='Manual visual selection across 101 active cases; varied kitchen colors, materials, viewpoints and operations. All 18 layout/style pairs distinct.',
                    fonts=FONTS, tiles=records)
    (OUT/'sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Wrote GIF, PNG, and sources.json', flush=True)


if __name__ == '__main__':
    main()
