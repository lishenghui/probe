"""Contact sheet per adapter: rows = variants (base, original, b75, ...), columns = frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', type=Path, help='directory with <adapter>__<variant>.mp4 and results.json')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 12, 24, 36, 48])
    parser.add_argument('--scale', type=float, default=0.4)
    args = parser.parse_args()
    results = json.loads((args.videos / 'results.json').read_text())['results']
    metrics = {(r['adapter'], r['variant']): r for r in results}
    for adapter in dict.fromkeys(r['adapter'] for r in results):
        variants = [r['variant'] for r in results if r['adapter'] == adapter and r['variant'] != 'original_repeat']
        rows = []
        for variant in variants:
            video = iio.imread(args.videos / f'{adapter}__{variant}.mp4', plugin='pyav')
            tiles = [Image.fromarray(video[min(i, len(video) - 1)]) for i in args.frames]
            w, h = (int(t * args.scale) for t in tiles[0].size)
            row = Image.new('RGB', (w * len(tiles) + 170, h), 'white')
            for j, tile in enumerate(tiles):
                row.paste(tile.resize((w, h)), (170 + j * w, 0))
            r = metrics[(adapter, variant)]
            label = [variant]
            if 'adapter_gpu_bytes' in r:
                label.append(f"{r['adapter_gpu_bytes']/1e6:.0f} MB")
            if 'deviation_ratio' in r:
                label += [f"dev {r['deviation_ratio']:.2f}", f"PSNR {r['psnr_vs_original']:.1f}"]
            ImageDraw.Draw(row).multiline_text((8, 8), '\n'.join(label), fill='black', spacing=6)
            rows.append(np.asarray(row))
        sheet = Image.fromarray(np.concatenate(rows, axis=0))
        out = args.videos / f'contact_{adapter}.jpg'
        sheet.save(out, quality=88)
        print(out)


if __name__ == '__main__':
    main()
