#!/usr/bin/env python
"""Reference / backbone / FYM frames side by side, one block per clip.

The metrics need backing by what the videos show (steps.md, What to report).
Each block is the reference row followed by one row per config, the same frame
indices in every row, so a motion that was transferred -- or not -- is visible
by reading down a column.

    python fym_bench/contact_sheet.py --runs /content/drive/MyDrive/ditflow/runs \\
        --clips camel,dog,bus --out /content/drive/MyDrive/ditflow/sheets
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fym_bench.videoio import read_frames  # noqa: E402

LABEL_W = 110


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", help="comma list (default: every clip with a finished cell)")
    ap.add_argument("--configs", default="backbone,fym")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cols", type=int, default=5, help="frames shown per row")
    ap.add_argument("--width", type=int, default=208, help="thumbnail width")
    ap.add_argument("--per-sheet", type=int, default=4, help="clips per image")
    args = ap.parse_args()

    root = Path(args.runs) / "fym_wan" / "davis"
    configs = args.configs.split(",")
    clips = args.clips.split(",") if args.clips else sorted(
        {p.parents[3].name for p in root.glob(f"*/subject/*/seed{args.seed}/done.json")})

    blocks = []
    for clip in clips:
        cells = {c: root / clip / "subject" / c / f"seed{args.seed}" for c in configs}
        done = {c: d for c, d in cells.items() if (d / "done.json").exists()}
        if not done:
            print(f"  {clip}: no finished cells, skipped")
            continue
        rows = [("reference", read_frames(next(iter(done.values())) / "original.mp4"))]
        rows += [(c, read_frames(done[c] / "results.mp4")) if c in done else (c, None) for c in configs]
        blocks.append((clip, rows))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for s in range(0, len(blocks), args.per_sheet):
        chunk = blocks[s:s + args.per_sheet]
        sample = next(f for _, rows in chunk for _, f in rows if f is not None)
        th = round(args.width * sample.shape[1] / sample.shape[2])
        n_rows = sum(len(rows) for _, rows in chunk)
        sheet = Image.new("RGB", (LABEL_W + args.cols * args.width, n_rows * th + 18 * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for clip, rows in chunk:
            draw.text((4, y + 2), clip, fill="black")
            y += 18
            for name, frames in rows:
                draw.text((4, y + th // 2 - 6), name, fill="black")
                if frames is None:
                    draw.text((LABEL_W + 4, y + th // 2 - 6), "(no finished cell)", fill="gray")
                else:
                    idx = np.linspace(0, len(frames) - 1, args.cols).round().astype(int)
                    for j, k in enumerate(idx):
                        sheet.paste(Image.fromarray(frames[k]).resize((args.width, th)), (LABEL_W + j * args.width, y))
                y += th
        out = Path(args.out) / f"sheet_{s // args.per_sheet + 1:02d}.png"
        sheet.save(out)
        print(f"wrote {out} ({', '.join(c for c, _ in chunk)})")


if __name__ == "__main__":
    main()
