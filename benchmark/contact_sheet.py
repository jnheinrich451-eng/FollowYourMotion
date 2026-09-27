#!/usr/bin/env python
"""Tile cut clips into contact sheets, one row per clip.

Captioning fifty sequences by opening fifty folders is the slowest step in
setting up the benchmark, and it is slow for no good reason: the model only ever
sees the 24 cut frames, so a handful of those per clip is the whole subject.

Rows are labelled with the clip id and, where the manifest carries it, how many
subjects actually move -- the multi-subject clips are the ones whose caption
needs a plural subject phrase, and spotting them from a sheet beats
cross-referencing a list.

    python benchmark/contact_sheet.py --manifest benchmark/davis50.csv \
        --out sheets --per-sheet 10
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_H = 22


def load_font(size=15):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick(frames, n):
    """n frames spread across the window, endpoints included."""
    if len(frames) <= n:
        return frames
    step = (len(frames) - 1) / (n - 1)
    return [frames[round(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="directory for the sheets")
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--cols", type=int, default=4, help="frames shown per clip")
    ap.add_argument("--width", type=int, default=240, help="thumbnail width")
    args = ap.parse_args()

    seen, clips = set(), []
    for r in csv.DictReader(open(args.manifest, newline="", encoding="utf-8")):
        if r["clip_id"] in seen:
            continue
        seen.add(r["clip_id"])
        clips.append(r)

    font = load_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheets = 0

    for start in range(0, len(clips), args.per_sheet):
        chunk = clips[start:start + args.per_sheet]
        tiles, rows = [], []
        for r in chunk:
            frames = sorted(Path(r["video_path"]).glob("*.jpg"))
            if not frames:
                continue
            ims = [Image.open(f).convert("RGB") for f in pick(frames, args.cols)]
            w = args.width
            h = round(ims[0].height * w / ims[0].width)
            ims = [im.resize((w, h), Image.LANCZOS) for im in ims]
            label = r["clip_id"]
            if r.get("n_salient_moving"):
                label += f"   [{r['n_salient_moving']} moving"
                label += f", {r.get('subjects','')}]"
            rows.append((ims, label, w, h))
            tiles.append(h + LABEL_H)

        if not rows:
            continue
        sheet_w = args.cols * args.width
        sheet = Image.new("RGB", (sheet_w, sum(tiles)), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for ims, label, w, h in rows:
            draw.rectangle([0, y, sheet_w, y + LABEL_H], fill=(24, 24, 28))
            draw.text((6, y + 3), label, fill=(255, 255, 255), font=font)
            y += LABEL_H
            for i, im in enumerate(ims):
                sheet.paste(im, (i * w, y))
            y += h

        sheets += 1
        path = out / f"sheet_{sheets:02d}.jpg"
        sheet.save(path, quality=88)
        print(f"{path}  {len(rows)} clips  {sheet.width}x{sheet.height}")

    print(f"\n{sheets} sheets covering {len(clips)} clips")


if __name__ == "__main__":
    main()
