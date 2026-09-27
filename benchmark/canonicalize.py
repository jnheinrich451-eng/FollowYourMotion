"""Bring a cut clip to the frozen canonical shape before extraction (S0 rule:
the extractor never resamples; configs/extract.yaml `canonical`).

    python benchmark/canonicalize.py --src E:/bench/packed/davis/bear --dst E:/bench/canon/davis/bear.mp4
    python benchmark/canonicalize.py --src E:/bench/packed/miradata --dst E:/bench/canon/miradata   # every cut dir inside

Rule, applied at NATIVE resolution first so the resize factor is exact:
  1. center-crop to the canonical aspect (3:2);   854x480 -> 720x480 (no resize)
  2. uniform resize to canonical H x W;           1280x720 -> crop 1080x720 -> resize x2/3 -> 720x480
Never squash: a non-uniform resize changes the intrinsics the camera metrics
depend on. Never resample time: the cut already holds `canonical.frames` frames.

Output: lossless RGB mp4 (libx264rgb, qp 0) + <stem>.canon.json recording the
crop box, resize factor, effective fps (from the cut's meta.json) and the
sha256 of the canonical frames. The written file is decoded back and compared
to the in-memory frames; a mismatch raises.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def canonical_from_config(path=REPO / "configs/extract.yaml") -> tuple[int, int, int]:
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["canonical"]
    if any(c[k] is None for k in ("frames", "height", "width")):
        raise SystemExit("configs/extract.yaml canonical is not frozen")
    return int(c["frames"]), int(c["height"]), int(c["width"])


def crop_box(h: int, w: int, H: int, W: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the centred crop with aspect W:H inside h x w."""
    if w * H >= h * W:                      # source is wider than canonical -> crop width
        cw = (h * W) // H
        cw -= cw % 2
        left = (w - cw) // 2
        return left, 0, left + cw, h
    ch = (w * H) // W                       # source is taller -> crop height
    ch -= ch % 2
    top = (h - ch) // 2
    return 0, top, w, top + ch


def canonicalize_frames(frames: np.ndarray, H: int, W: int):
    """(T,h,w,3) uint8 -> (T,H,W,3) uint8, plus the record of what was done."""
    T, h, w, _ = frames.shape
    l, t, r, b = crop_box(h, w, H, W)
    out = np.empty((T, H, W, 3), np.uint8)
    for i in range(T):
        im = Image.fromarray(frames[i]).crop((l, t, r, b))
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        out[i] = np.asarray(im)
    return out, {"native_hw": [h, w], "crop_ltrb": [l, t, r, b],
                 "resize_factor": W / (r - l), "canonical_hw": [H, W]}


def load_source(src: Path, n_frames: int) -> tuple[np.ndarray, dict]:
    """A cut directory (JPEGs + meta.json) or a video file. Refuses the wrong frame count."""
    meta = {}
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files])
        mp = src / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
    else:
        import imageio.v2 as iio
        rd = iio.get_reader(str(src))
        meta = {"container_fps": rd.get_meta_data().get("fps")}
        frames = np.stack([f[..., :3] for f in rd])
        rd.close()
    if frames.shape[0] != n_frames:
        raise SystemExit(f"{src}: {frames.shape[0]} frames, canonical is {n_frames} — "
                         "temporal resampling is cut.py's job, not this tool's")
    return frames, meta


def write_lossless(frames: np.ndarray, dst: Path, fps: float) -> None:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    T, H, W, _ = frames.shape
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24", str(dst)]
    p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))


def read_back(dst: Path) -> np.ndarray:
    import imageio.v2 as iio
    rd = iio.get_reader(str(dst))
    out = np.stack([f[..., :3] for f in rd])
    rd.close()
    return out


def process_one(src: Path, dst: Path, n: int, H: int, W: int) -> dict:
    frames, meta = load_source(src, n)
    canon, rec = canonicalize_frames(frames, H, W)
    fps = float(meta.get("effective_fps") or meta.get("container_fps") or 0)
    if not fps:
        raise SystemExit(f"{src}: no effective_fps in meta.json and no container fps")
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_lossless(canon, dst, fps)
    back = read_back(dst)
    if back.shape != canon.shape or not np.array_equal(back, canon):
        raise RuntimeError(f"{dst}: decoded frames differ from the canonical frames "
                           f"(shape {back.shape} vs {canon.shape}) — lossless write failed")
    rec |= {"src": str(src), "dst": str(dst), "frames": n, "fps": fps,
            "source_meta": meta, "canonical_sha256": hashlib.sha256(canon.tobytes()).hexdigest()}
    dst.with_suffix(".canon.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cut dir, video file, or a folder of cut dirs")
    ap.add_argument("--dst", required=True, help="output .mp4, or output folder when --src is a folder of cuts")
    ap.add_argument("--config", default=str(REPO / "configs/extract.yaml"))
    args = ap.parse_args()
    n, H, W = canonical_from_config(args.config)
    src, dst = Path(args.src), Path(args.dst)
    cuts = [src] if (src.is_file() or (src / "meta.json").exists()) else \
           sorted(p for p in src.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not cuts:
        raise SystemExit(f"nothing to do under {src}")
    for c in cuts:
        out = dst if len(cuts) == 1 and dst.suffix == ".mp4" else dst / f"{c.stem if c.is_file() else c.name}.mp4"
        rec = process_one(c, out, n, H, W)
        print(f"{c.name}: {rec['native_hw']} -> crop {rec['crop_ltrb']} -> x{rec['resize_factor']:.4f} "
              f"-> {rec['canonical_hw']} @ {rec['fps']} fps -> {out}")


if __name__ == "__main__":
    sys.exit(main())
