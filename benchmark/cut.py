#!/usr/bin/env python
"""Cut source clips to a fixed-length frame window for the benchmark.

Writes frame DIRECTORIES, never mp4, deliberately. motion_guidance.py:265
decodes video with torchvision.io.read_video, which this repo has already been
bitten by once -- see the "Drop torchvision video I/O: read_video/write_video
were removed upstream" commit, which fixed the Wan file and left the original
importing it at line 15. A directory of JPEGs takes the PIL branch at line 269
instead and sidesteps the question entirely.

Frame count and stride are separate decisions. 24 frames is what DiTFlow
generates and cannot change; the stride is what makes those 24 frames cover a
comparable amount of real time across splits shot at different rates:

    DAVIS      ~24 fps -> stride 2 -> 2.0 s
    MiraData   ~30 fps -> stride 2 -> 1.6 s
    MOVi-F      12 fps -> stride 1 -> 2.0 s   (24 frames is the entire clip)

MOVi-F is the binding constraint at 2.0 s, so everything else targets that.
Exact cross-split alignment is not available -- the only common divisor of 24,
12 and 30 fps is 6, which buys 12 frames in two seconds, fewer than DiTFlow can
generate. Record the effective fps per clip and keep the splits in separate
tables rather than pretending otherwise.

    python benchmark/cut.py --src E:/DAVIS/JPEGImages/480p/bmx-trees \
        --dst data/packed/davis/bmx-trees --src-fps 24

    python benchmark/cut.py --batch data/miradata_raw --dst data/packed/miradata \
        --split miradata --manifest benchmark/miradata.csv
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import imageio.v2 as imageio

REPO = Path(__file__).resolve().parent.parent
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")
FRAME_EXTS = (".jpg", ".jpeg", ".png")


def source_fps(src, override):
    """Frames per second of the source, or None if it cannot be determined."""
    if override:
        return float(override)
    if Path(src).is_dir():
        return None            # a frame folder carries no timing of its own
    try:
        meta = imageio.get_reader(str(src)).get_meta_data()
        fps = meta.get("fps")
        return float(fps) if fps else None
    except Exception:
        return None


def pick_indices(n_src, fps_src, frames, fps_out, start):
    """Nearest-neighbour source indices for `frames` samples at `fps_out`.

    Index selection only -- never interpolation or blending, which would create
    ghosting that corrupts the tracker the motion metrics depend on.
    """
    stride = max(1, int(round(fps_src / fps_out)))
    idx = start + np.arange(frames) * stride
    if idx[-1] >= n_src:
        need = int(idx[-1]) + 1
        raise ValueError(
            f"needs {need} source frames (start={start}, stride={stride}, "
            f"{frames} frames) but the source has {n_src}. Lower --fps-out, "
            f"--frames or --start.")
    return idx, stride


def load_frames(src, wanted):
    """Return the requested frame indices as RGB arrays, reading no more than needed."""
    src = Path(src)
    want, top = set(int(i) for i in wanted), int(max(wanted))

    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in FRAME_EXTS)
        if not files:
            sys.exit(f"no frames found in {src}")
        if top >= len(files):
            raise ValueError(f"{src} has {len(files)} frames, index {top} requested")
        return [np.asarray(Image.open(files[i]).convert("RGB")) for i in wanted], len(files)

    got, reader = {}, imageio.get_reader(str(src))
    try:
        for i, frame in enumerate(reader):
            if i in want:
                got[i] = np.asarray(frame)[..., :3]
            if i >= top:
                break
    finally:
        reader.close()
    if top not in got:
        raise ValueError(f"{src} ended at frame {len(got) and max(got)}, "
                         f"index {top} requested")
    return [got[int(i)] for i in wanted], top + 1


def count_frames(src):
    """Cheap upper bound on source length, for the index check."""
    src = Path(src)
    if src.is_dir():
        return sum(1 for p in src.iterdir() if p.suffix.lower() in FRAME_EXTS)
    try:
        return imageio.get_reader(str(src)).count_frames()
    except Exception:
        return 10 ** 9        # unknown; load_frames will raise if it runs out


def cut_one(src, dst, frames, fps_out, start, fps_src, quality):
    n_src = count_frames(src)
    idx, stride = pick_indices(n_src, fps_src, frames, fps_out, start)
    images, n_seen = load_frames(src, idx)

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.glob("*.jpg"):        # a re-cut must not leave stale frames behind
        p.unlink()

    digest = hashlib.sha256()
    for i, arr in enumerate(images):
        # Zero-padded so DiTFlow's int(stem.split('f')[-1]) sort stays correct.
        out = dst / f"{i:05d}.jpg"
        Image.fromarray(arr).save(out, quality=quality)
        digest.update(out.read_bytes())

    meta = {
        "source": str(src), "source_frames": n_seen,
        "source_fps": fps_src, "stride": stride,
        "effective_fps": round(fps_src / stride, 4),
        "duration_s": round(frames * stride / fps_src, 4),
        "start": start, "frames": frames,
        "indices": [int(i) for i in idx],
        "sha256": digest.hexdigest(),
        "cut_at": datetime.now().isoformat(timespec="seconds"),
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def manifest_path(dst):
    """Repo-relative where possible, absolute otherwise.

    On Windows a data drive and the repo drive have no relative path between
    them, and relpath raises rather than returning something usable. sweep.py
    takes absolute paths unchanged, so falling back costs nothing.
    """
    try:
        return os.path.relpath(dst, REPO).replace("\\", "/")
    except ValueError:
        return str(Path(dst).resolve())


def find_clips(root):
    """Every clip under root: video files, or subdirectories holding frames."""
    root = Path(root)
    clips = [p for p in sorted(root.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    clips += [p for p in sorted(root.iterdir())
              if p.is_dir() and any(q.suffix.lower() in FRAME_EXTS for q in p.iterdir())]
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="one video file or one directory of frames")
    ap.add_argument("--batch", help="directory of clips, each cut in turn")
    ap.add_argument("--dst", required=True, help="output directory (root, when --batch)")
    ap.add_argument("--frames", type=int, default=24,
                    help="frames to write; 24 is DiTFlow's default (default: 24)")
    ap.add_argument("--fps-out", type=float, default=12.0,
                    help="target effective fps, sets the stride (default: 12)")
    ap.add_argument("--start", type=int, default=0, help="first source frame index")
    ap.add_argument("--src-fps", type=float,
                    help="source fps; required for frame folders, which carry no timing")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--split", default="unknown", help="split name for the manifest")
    ap.add_argument("--manifest", help="write a sweep.py manifest for the cut clips")
    args = ap.parse_args()

    if bool(args.src) == bool(args.batch):
        sys.exit("pass exactly one of --src or --batch")

    jobs = ([(Path(args.src), Path(args.dst))] if args.src else
            [(c, Path(args.dst) / c.stem) for c in find_clips(args.batch)])
    if not jobs:
        sys.exit(f"no clips found in {args.batch}")

    rows, failed = [], []
    for src, dst in jobs:
        fps = source_fps(src, args.src_fps)
        if not fps:
            failed.append((src, "source fps unknown -- pass --src-fps"))
            continue
        try:
            meta = cut_one(src, dst, args.frames, args.fps_out,
                           args.start, fps, args.quality)
        except ValueError as e:
            failed.append((src, str(e)))
            continue
        print(f"{src.name} -> {dst}  stride={meta['stride']} "
              f"eff_fps={meta['effective_fps']} {meta['duration_s']}s")
        rows.append({
            "clip_id": dst.name, "split": args.split,
            "video_path": manifest_path(dst),
            "prompt_id": "subject", "prompt": "",
            "source": meta["source"], "source_fps": meta["source_fps"],
            "stride": meta["stride"], "effective_fps": meta["effective_fps"],
            "sha256": meta["sha256"],
        })

    if args.manifest and rows:
        with open(args.manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.manifest} with {len(rows)} rows -- fill in the prompt column")

    print(f"\n{len(rows)} cut, {len(failed)} skipped")
    for src, why in failed:
        print(f"  ! {Path(src).name}: {why}")


if __name__ == "__main__":
    main()
