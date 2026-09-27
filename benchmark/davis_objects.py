#!/usr/bin/env python
"""Label DAVIS manifest rows by how many subjects actually move.

DAVIS 2017 annotates multiple instances per sequence, but a raw instance count
mislabels: dog-gooses has five, four of which are geese covering 0.3% of frame.
So an instance counts as a subject only if it is both large enough to matter and
actually moving, measured over the same window cut.py kept -- not the whole
clip, which would credit motion the model never sees.

This exists to stratify, not to filter. Multi-subject clips are where camera and
object motion entangle worst -- AMF is one displacement field, so two objects
plus a moving camera superpose three motions with nothing separating them -- and
dropping them would remove the evidence while flattering every method measured.

Rigid vs articulated is left to you: it is a semantic call that masks do not
settle, and it is a quick pass to make while writing the captions anyway.

    python benchmark/davis_objects.py --manifest benchmark/davis50.csv \
        --annotations E:/DAVIS/Annotations/480p
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def measure(mask_dir, indices, area_min, path_min):
    """(n_instances, n_salient_moving) over the frames actually evaluated."""
    masks = [np.array(Image.open(Path(mask_dir) / f"{i:05d}.png")) for i in indices]
    h, w = masks[0].shape
    diag = (h ** 2 + w ** 2) ** 0.5
    ids = sorted(set(np.unique(masks[0]).tolist()) - {0})

    salient = 0
    for oid in ids:
        cents, areas = [], []
        for m in masks:
            ys, xs = np.nonzero(m == oid)
            if len(xs):
                cents.append((xs.mean(), ys.mean()))
                areas.append(len(xs))
        if len(cents) < 2:
            continue
        c = np.array(cents)
        # Path length, not net displacement: a subject that moves out and back
        # is still a moving subject, and net displacement would score it zero.
        path = np.linalg.norm(np.diff(c, axis=0), axis=1).sum() / diag
        if np.mean(areas) / (h * w) >= area_min and path >= path_min:
            salient += 1
    return len(ids), salient


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotations", required=True, help="DAVIS Annotations/480p")
    ap.add_argument("--out", help="defaults to editing --manifest in place")
    ap.add_argument("--area-min", type=float, default=0.01,
                    help="fraction of frame an instance must average (default: 0.01)")
    ap.add_argument("--path-min", type=float, default=0.05,
                    help="centroid path as a fraction of the diagonal (default: 0.05)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    if not rows:
        sys.exit(f"{args.manifest} has no rows")

    cache, missing = {}, []
    for r in rows:
        cid = r["clip_id"]
        if cid not in cache:
            mask_dir = Path(args.annotations) / cid
            meta = Path(r["video_path"]) / "meta.json"
            if not mask_dir.is_dir() or not meta.exists():
                missing.append(cid)
                cache[cid] = ("", "", "")
            else:
                idx = json.loads(meta.read_text(encoding="utf-8"))["indices"]
                n, sal = measure(mask_dir, idx, args.area_min, args.path_min)
                cache[cid] = (n, sal, "multi" if sal >= 2 else "single")
        r["n_instances"], r["n_salient_moving"], r["subjects"] = cache[cid]

    out = args.out or args.manifest
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    clips = {r["clip_id"]: r for r in rows}
    multi = sorted(c for c, r in clips.items() if r["subjects"] == "multi")
    print(f"labelled {len(clips)} clips (area>={args.area_min}, path>={args.path_min})")
    print(f"  single: {len(clips) - len(multi)}   multi: {len(multi)}")
    if missing:
        print("  ! no annotations or meta.json: " + ", ".join(sorted(set(missing))))
    print(f"\nwrote {out}")
    print("\nplural subject phrase needed for these captions:")
    for c in multi:
        print(f"  {c:22} {clips[c]['n_salient_moving']} moving "
              f"(of {clips[c]['n_instances']} annotated)")


if __name__ == "__main__":
    main()
