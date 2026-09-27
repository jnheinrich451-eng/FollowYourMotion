"""Camera trajectory metrics (Q1 protocol, Appendix A.1; design doc D10-D12).

Input convention for every public function that takes a trajectory: (T,4,4)
camera-to-world matrices with OpenCV camera axes (x right, y down, z forward).
Anything else goes through `convert_convention` FIRST, by name -- a wrong
convention must be a loud failure (test_camera_metrics: mismatch ~180 deg),
never a silent one.

Definitions:
  ATE          mean_t || s R_a c_est,t + t_a - c_gt,t ||      after sim(3) on the full trajectory
  CamRotErr    mean_t geodesic(R_a R_est,t, R_gt,t)  degrees   (sum also returned)
  CamTransErr  mean_t || c'_t - c_gt,t ||  unsquared L2         (sum also returned)
  RPE          relative pose est vs gt at stride 1, alignment-free
Scale s is fitted, reported, never divided out (cam_is_metric=False upstream).
"""
from __future__ import annotations

import numpy as np

_FLIP_YZ = np.diag([1.0, -1.0, -1.0])
CONVENTIONS = ("c2w_opencv", "c2w_opengl", "w2c_opencv", "w2c_opengl")


class DegenerateError(ValueError):
    """Trajectory has no translational extent: sim(3) is undefined."""


# --------------------------------------------------------------- conventions
def convert_convention(E, src: str) -> np.ndarray:
    """(T,4,4) in convention `src` -> c2w OpenCV. Named, tested, no guessing."""
    if src not in CONVENTIONS:
        raise ValueError(f"unknown convention {src!r}; one of {CONVENTIONS}")
    E = np.asarray(E, np.float64)
    if E.ndim != 3 or E.shape[1:] != (4, 4):
        raise ValueError(f"expected (T,4,4), got {E.shape}")
    out = E.copy()
    if src.startswith("w2c"):
        out = np.linalg.inv(out)
    if src.endswith("opengl"):
        # camera axes: X_gl = diag(1,-1,-1) X_cv  =>  R_cv = R_gl @ diag(1,-1,-1)
        out[:, :3, :3] = out[:, :3, :3] @ _FLIP_YZ
    return out


def to_camera0(E) -> np.ndarray:
    """Re-express c2w so that world = camera-0 (E[0] = I), as extract-v1 stores it."""
    E = np.asarray(E, np.float64)
    return np.linalg.inv(E[0])[None] @ E


# ----------------------------------------------------------------- alignment
def umeyama_sim3(src, dst, with_scale: bool = True):
    """Least-squares similarity src -> dst: returns (s, R, t) with dst ~ s R src + t.
    src, dst: (N,3). Raises DegenerateError when src has no spread."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 2:
        raise ValueError(f"need matching (N>=2,3) arrays, got {src.shape} {dst.shape}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    if var_s < 1e-12 * max(1.0, (xd ** 2).sum() / len(src)) or var_s == 0.0:
        raise DegenerateError("source trajectory has no translational extent")
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def procrustes_rotation(R_est, R_gt) -> np.ndarray:
    """Single R_a minimising sum_t ||R_a R_est,t - R_gt,t||_F (rotation-only alignment)."""
    M = np.einsum("tij,tkj->ik", np.asarray(R_gt, np.float64), np.asarray(R_est, np.float64))
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    return U @ S @ Vt


# ------------------------------------------------------------------- errors
def rot_err_deg(R_a, R_b) -> np.ndarray:
    """Geodesic angle between rotation stacks (T,3,3) -> (T,) degrees."""
    R_a = np.asarray(R_a, np.float64)
    R_b = np.asarray(R_b, np.float64)
    M = np.swapaxes(R_a, 1, 2) @ R_b                # R_a^T R_b
    cos = (np.trace(M, axis1=1, axis2=2) - 1.0) / 2.0
    skew = M - np.swapaxes(M, 1, 2)                 # 2 sin(theta) * [axis]_x
    sin = np.sqrt((skew ** 2).sum(axis=(1, 2)) / 8.0)
    # atan2 keeps full precision near 0 and near 180 deg, where arccos does not
    return np.degrees(np.arctan2(sin, cos))


def relative_poses(E) -> np.ndarray:
    """Delta_t = inv(E_t) E_{t+1}: camera-(t+1) expressed in camera-t. (T-1,4,4)."""
    E = np.asarray(E, np.float64)
    return np.linalg.inv(E[:-1]) @ E[1:]


def camera_errors(E_est, E_gt, *, align: str = "sim3", idx=None) -> dict:
    """All camera numbers for one clip. Both inputs c2w OpenCV, (T,4,4).

    align: "sim3" (default; falls back to "rot_only" with a flag if degenerate),
           "scale_only" (both trajectories anchored at frame 0 -> R_a = I, t_a = 0,
           least-squares s; the CameraCtrl/CamCo convention, and the right choice
           when a free R_a would be unconstrained by a near-linear path),
           "rot_only", or "none".
    idx:   optional frame indices applied to BOTH trajectories (frame matching).
    """
    E_est = np.asarray(E_est, np.float64)
    E_gt = np.asarray(E_gt, np.float64)
    if E_est.shape != E_gt.shape or E_est.ndim != 3 or E_est.shape[1:] != (4, 4):
        raise ValueError(f"shape mismatch {E_est.shape} vs {E_gt.shape}")
    if idx is not None:
        idx = np.asarray(idx, int)
        E_est, E_gt = E_est[idx], E_gt[idx]
    n = len(E_est)
    c_est, c_gt = E_est[:, :3, 3], E_gt[:, :3, 3]
    R_est, R_gt = E_est[:, :3, :3], E_gt[:, :3, :3]

    used = align
    if align == "sim3":
        try:
            s, R_a, t_a = umeyama_sim3(c_est, c_gt)
        except DegenerateError:
            used = "rot_only"
    if used == "rot_only":
        s, R_a = 1.0, procrustes_rotation(R_est, R_gt)
        t_a = c_gt.mean(0) - R_a @ c_est.mean(0)
    elif used == "scale_only":
        den = float((c_est ** 2).sum())
        s = float((c_est * c_gt).sum() / den) if den > 1e-12 else 1.0
        R_a, t_a = np.eye(3), np.zeros(3)
    elif used == "none":
        s, R_a, t_a = 1.0, np.eye(3), np.zeros(3)
    elif used != "sim3":
        raise ValueError(f"align must be sim3|scale_only|rot_only|none, got {align!r}")

    c_al = (s * (R_a @ c_est.T)).T + t_a
    R_al = R_a[None] @ R_est
    d = np.linalg.norm(c_al - c_gt, axis=1)
    th = rot_err_deg(R_al, R_gt)

    # RPE: alignment-free; translation of the estimate scaled by s so units match
    D_est, D_gt = relative_poses(E_est), relative_poses(E_gt)
    rpe_rot = rot_err_deg(D_est[:, :3, :3], D_gt[:, :3, :3])
    rpe_tr = np.linalg.norm(s * D_est[:, :3, 3] - D_gt[:, :3, 3], axis=1)

    return {
        "n": int(n), "align_used": used, "scale": float(s),
        "ate": float(d.mean()),
        "rot_mean_deg": float(th.mean()), "rot_sum_deg": float(th.sum()),
        "trans_mean": float(d.mean()), "trans_sum": float(d.sum()),
        "rpe_rot_deg": float(rpe_rot.mean()) if n > 1 else float("nan"),
        "rpe_trans": float(rpe_tr.mean()) if n > 1 else float("nan"),
        "net_displacement_gt": float(np.linalg.norm(c_gt[-1] - c_gt[0])),
        "per_frame_rot_deg": th, "per_frame_trans": d,
    }
