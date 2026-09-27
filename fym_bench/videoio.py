"""Lossless video I/O shared by the runner and the generator.

Same encoding as the ditflow repo's canonicalize.py (libx264rgb, qp 0): the
reference that FYM tunes on, the original.mp4 that collect.py tracks, and the
generated results.mp4 all round-trip bit-exactly, so compression never enters
a metric.
"""
import subprocess

import numpy as np


def read_frames(path):
    """Decode a video to (T,H,W,3) uint8."""
    import imageio.v2 as iio
    rd = iio.get_reader(str(path))
    out = np.stack([f[..., :3] for f in rd])
    rd.close()
    return out


def write_lossless(frames, dst, fps):
    """(T,H,W,3) uint8 -> lossless RGB mp4, decoded back and compared."""
    import imageio_ffmpeg
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    T, H, W, _ = frames.shape
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24", str(dst)]
    p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    back = read_frames(dst)
    if back.shape != frames.shape or not np.array_equal(back, frames):
        raise RuntimeError(f"{dst}: decoded frames differ from what was written "
                           f"({back.shape} vs {frames.shape}) -- lossless write failed")
