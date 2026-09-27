#!/usr/bin/env python
"""Join local MiraData clips to RealCam-Vid, and emit a sweep.py manifest.

RealCam-Vid annotates clips drawn from RealEstate10K, DL3DV-10K and MiraData9K
with per-frame camera extrinsics, intrinsics, a metric align_factor, and two
captions. For the MiraData clips that means real video with object motion AND a
camera trajectory -- the combination the motion-transfer and camera-control
literatures each evaluate without.

What this writes per clip:

  <dst>/<clip_id>.npy    (N,4,4) camera extrinsics, for trajectory conditioning
  manifest row           captions, measured camera motion, provenance

Prompt classes follow DiTFlow: Caption reuses the reference's caption verbatim,
Subject swaps the subject and keeps the background, Scene describes something
else entirely. Only Caption can be filled automatically, and it is also the one
class that does not test transfer at all -- prompt and reference agree, so no
content change has to be survived. It belongs near the Layer 3 upper bound, not
in the headline row.

Subject and Scene rows are seeded with the caption behind a TODO marker, so the
edit is one noun rather than one sentence ("A warrior walks through a village"
-> "A bear walks through a village"). sweep.py refuses a row still carrying the
marker: an unedited prompt would otherwise run as a second Caption cell filed
under the Subject column, which nothing downstream would catch.

    python benchmark/realcam.py --clips "E:/MiraData 9K/81 frames" \
        --realcam E:/RealCam-Vid --dst E:/packed/miradata_traj \
        --manifest benchmark/miradata.csv --prompt-classes caption,subject,scene

Note: RealCam-Vid_train.npz is ~1 GB of pickled objects and is read whole.
Expect a few GB of RAM while it loads.
"""

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TODO = "TODO "   # sweep.py refuses any prompt still carrying this
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")


def load_records(realcam_root, splits):
    """Every RealCam-Vid record, keyed by the clip stem of its video_path."""
    by_stem = {}
    for split in splits:
        npz = Path(realcam_root) / f"RealCam-Vid_{split}.npz"
        if not npz.exists():
            print(f"  ! {npz} not found, skipping")
            continue
        print(f"  loading {npz.name} ...", flush=True)
        arr = np.load(npz, allow_pickle=True)["arr_0"]
        for rec in arr:
            stem = PurePosixPath(rec["video_path"]).stem
            # First split wins: test is the smaller, cleaner list, so pass it
            # first if a clip somehow appears in both.
            by_stem.setdefault(stem, (split, rec))
        print(f"    {len(arr)} records")
    return by_stem


def camera_motion(extrinsics):
    """Translation path length and net displacement, in the file's own units.

    Deliberately not multiplied by align_factor -- that factor's exact semantics
    are RealCam-Vid's to define, so it is recorded alongside rather than folded
    in. Use these for relative stratification, not as metres.
    """
    t = np.asarray(extrinsics, dtype=np.float64)[:, :3, 3]
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    return float(steps.sum()), float(np.linalg.norm(t[-1] - t[0]))


def band(value, edges):
    """Stratification band: low / mid / high against two cut points."""
    return "low" if value < edges[0] else ("mid" if value < edges[1] else "high")


def list_clips(clips_dir):
    """Raw video files, or the frame directories cut.py produces."""
    root = Path(clips_dir)
    vids = [p for p in sorted(root.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    cuts = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "meta.json").exists()]
    return vids + cuts


def clip_id(path):
    """Identifier for a clip, from a video file or a cut directory.

    Path.stem is wrong for the directories: these ids carry dots
    ("000000000001.5.002"), so stem strips ".002" as if it were an extension
    and the RealCam-Vid lookup silently misses every clip.
    """
    path = Path(path)
    return path.name if path.is_dir() else path.stem


def cut_indices(clip):
    """The source frame indices a cut directory kept, or None for a raw video.

    RealCam-Vid annotates every frame of the original clip. Once cut.py has
    decimated that clip by a stride, the trajectory has to be decimated by the
    same indices or the camera and the frames describe different moments --
    which nothing downstream would flag, because both arrays are individually
    well formed.
    """
    meta = Path(clip) / "meta.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8"))["indices"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="directory of local clips")
    ap.add_argument("--realcam", required=True, help="RealCam-Vid root")
    ap.add_argument("--dst", required=True, help="where to write trajectory .npy files")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="miradata",
                    help="name of YOUR benchmark split; sweep.py files runs under it")
    ap.add_argument("--prompt-classes", default="caption",
                    help="comma-separated: caption,subject,scene (default: caption)")
    ap.add_argument("--realcam-splits", default="test,train",
                    help="which RealCam-Vid npz files to search, in order "
                         "(unrelated to --split, which names YOUR benchmark split)")
    args = ap.parse_args()

    clips = list_clips(args.clips)
    if not clips:
        sys.exit(f"no clips found in {args.clips}")
    classes = [c.strip() for c in args.prompt_classes.split(",") if c.strip()]
    print(f"{len(clips)} local clips, prompt classes: {', '.join(classes)}")

    records = load_records(args.realcam,
                           [s.strip() for s in args.realcam_splits.split(",")])
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    matched, missing, rows = [], [], []
    for clip in clips:
        cid = clip_id(clip)
        hit = records.get(cid)
        if hit is None:
            missing.append(cid)
            continue
        src_split, rec = hit
        ext = np.asarray(rec["camera_extrinsics"], dtype=np.float64)
        n_full = int(ext.shape[0])
        idx = cut_indices(clip)
        if idx is not None:
            if max(idx) >= n_full:
                missing.append(f"{cid} (cut needs frame {max(idx)}, "
                               f"trajectory has {n_full})")
                continue
            ext = ext[idx]
        traj = dst / f"{cid}.npy"
        np.save(traj, ext)
        path_len, net = camera_motion(ext)
        matched.append((clip, rec, ext, path_len, net, traj, src_split, n_full))

    # Bands are relative to this clip set, so they describe the spread you
    # actually have rather than an absolute scale that may not apply.
    if matched:
        lens = sorted(m[3] for m in matched)
        edges = (lens[len(lens) // 3], lens[2 * len(lens) // 3])
    for clip, rec, ext, path_len, net, traj, src_split, n_full in matched:
        for cls in classes:
            rows.append({
                "clip_id": clip_id(clip), "split": args.split,
                "video_path": str(clip.resolve()),
                "prompt_id": cls,
                # Only the Caption class is the reference's own caption, and it
                # is the one class that does not test transfer -- prompt and
                # reference agree, so nothing has to survive a content change.
                # Subject and Scene are seeded with that caption behind a TODO
                # marker: edit the subject noun and drop the marker. sweep.py
                # refuses to run a row that still carries it, so a prompt you
                # forget to edit stops the sweep instead of quietly becoming a
                # duplicate Caption run in the Subject column.
                "prompt": (rec["short_caption"] if cls == "caption"
                           else TODO + rec["short_caption"]),
                "traj_path": str(traj.resolve()),
                "traj_frames": int(ext.shape[0]),
                "traj_source_frames": n_full,
                "cam_path_len": round(path_len, 6),
                "cam_net_disp": round(net, 6),
                "cam_band": band(path_len, edges),
                "align_factor": rec.get("align_factor"),
                "camera_scale": rec.get("camera_scale"),
                "source": f"RealCam-Vid/{src_split}:{rec['video_path']}",
                "caption_short": rec["short_caption"],
                "caption_long": rec.get("long_caption", ""),
            })

    if not rows:
        looked = ", ".join(clip_id(c) for c in clips[:5])
        sys.exit("no clips matched RealCam-Vid."
                 f"\n  looked up: {looked}{'...' if len(clips) > 5 else ''}"
                 "\n  check --realcam-splits (these clips are mostly in 'train')")

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nmatched {len(matched)}/{len(clips)} clips")
    if missing:
        print("  not in RealCam-Vid: " + ", ".join(missing))
    counts = {b: sum(1 for m in matched if band(m[3], edges) == b)
              for b in ("low", "mid", "high")}
    print(f"  camera bands: {counts}")
    print(f"\nwrote {args.manifest} with {len(rows)} rows "
          f"({len(matched)} clips x {len(classes)} prompt classes)")
    todo = sum(1 for r in rows if r["prompt"].startswith(TODO))
    if todo:
        print(f"  {todo} rows are seeded with the caption behind a {TODO.strip()} marker -- "
              f"swap the subject noun and drop the marker")


if __name__ == "__main__":
    main()
