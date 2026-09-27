"""T4.2 of the Q1 protocol: extractor camera vs RealCam-Vid on MiraData.

Agreement between two ESTIMATORS, not accuracy. Where they diverge, at least
one is wrong and the clip is flagged; neither is called right.

    python benchmark/q1_camera_realcam.py --extract ditflow_extract/extract/extract-v1 \
        --manifest benchmark/miradata.csv --out results

Frame matching: the RealCam trajectory holds the 24 cut frames; their source
indices are in the cut's meta.json (`indices`), never in miradata.csv's
`traj_source_frames` (a count). Extraction ran on source frames 0..80, so
the matched set is indices < n_extracted.

RealCam-Vid convention (its README): 4x4 relative-scale WORLD-TO-CAMERA,
OpenCV axes. Converted by name; both trajectories re-anchored to the first
matched frame before sim(3).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import camera_metrics as cm                          # noqa: E402

REALCAM_CONVENTION = "w2c_opencv"


def match_indices(cut_dir: Path, n_extracted: int) -> np.ndarray:
    meta = json.loads((cut_dir / "meta.json").read_text(encoding="utf-8"))
    idx = np.asarray(meta["indices"], int)
    return idx[idx < n_extracted]


def score_clip(raw_npz: Path, traj_npy: Path, cut_dir: Path) -> dict:
    with np.load(raw_npz) as z:
        E_est_full = z["cam_extrinsics"].astype(np.float64)
        conv = str(z["cam_convention"])
    assert conv.startswith("c2w_opencv"), conv
    W = np.load(traj_npy).astype(np.float64)                       # (24,4,4) w2c
    idx = match_indices(cut_dir, len(E_est_full))
    if len(idx) < 3:
        return {"n": int(len(idx)), "error": "too few matched frames"}
    keep = np.arange(len(W))[: len(idx)]                             # traj frame k <-> source idx[k]
    E_gt = cm.to_camera0(cm.convert_convention(W[keep], REALCAM_CONVENTION))
    E_est = cm.to_camera0(E_est_full[idx])
    # both anchored at the first matched frame -> scale_only is the literature's
    # definition and does not leak an unconstrained R_a into RotErr on a near-
    # linear path; sim(3) ATE is reported alongside as the protocol's primary
    e = cm.camera_errors(E_est, E_gt, align="scale_only")
    e3 = cm.camera_errors(E_est, E_gt, align="sim3")
    return {k: v for k, v in e.items() if not k.startswith("per_frame")} | {
        "ate_sim3": e3["ate"], "rot_mean_deg_sim3": e3["rot_mean_deg"], "scale_sim3": e3["scale"],
        "matched_source_indices": idx.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True, help="extract-v1 output root")
    ap.add_argument("--manifest", default="benchmark/miradata.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--clips", nargs="*", help="subset of clip ids (default: all with raw.npz)")
    ap.add_argument("--floor-rot-deg", type=float, default=None,
                    help="divergence flag threshold; None until Q1 Kubric floor exists")
    args = ap.parse_args()

    rows = {r["clip_id"]: r for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
            if r["prompt_id"] == "caption"}
    root = Path(args.extract)
    ids = args.clips or sorted(p.name for p in root.iterdir() if (p / "raw.npz").exists())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for cid in ids:
        r = rows.get(cid)
        if r is None or not (root / cid / "raw.npz").exists():
            print(f"skip {cid}: not in manifest or not extracted"); continue
        rec = {"clip_id": cid, "band_provisional": r["cam_band"], "realcam_path_len": r["cam_path_len"]}
        rec |= score_clip(root / cid / "raw.npz", Path(r["traj_path"]), Path(r["video_path"]))
        if args.floor_rot_deg is not None and "rot_mean_deg" in rec:
            rec["diverges"] = bool(rec["rot_mean_deg"] > args.floor_rot_deg)
        recs.append(rec)
        print(f"{cid} band={rec['band_provisional']:4s} n={rec.get('n')} "
              f"rot={rec.get('rot_mean_deg', float('nan')):.2f}deg trans={rec.get('trans_mean', float('nan')):.4f} "
              f"s={rec.get('scale', float('nan')):.3f} rpe_rot={rec.get('rpe_rot_deg', float('nan')):.2f} "
              f"align={rec.get('align_used')}")
    (out / "q1_camera_realcam.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    md = ["# T4.2 — extractor camera vs RealCam-Vid (agreement, not accuracy)", "",
          "| clip | band (prov.) | n | CamRotErr° (scale-only) | CamTransErr (scale-only) | ATE sim(3) | RotErr° sim(3) | RPE rot°/frame | RPE trans | s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        g = lambda k, f=".3f": format(r.get(k, float("nan")), f)
        md.append(f"| {r['clip_id']} | {r['band_provisional']} | {r.get('n')} | {g('rot_mean_deg','.2f')} | {g('trans_mean','.4f')} "
                  f"| {g('ate_sim3','.4f')} | {g('rot_mean_deg_sim3','.2f')} | {g('rpe_rot_deg','.2f')} | {g('rpe_trans','.4f')} | {g('scale')} |")
    md += ["", "Divergence flags need the Kubric floor (`--floor-rot-deg`); none applied." if args.floor_rot_deg is None
           else f"Divergence flag: CamRotErr > {args.floor_rot_deg}°."]
    (out / "q1_camera_realcam.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out / 'q1_camera_realcam.md'}")


if __name__ == "__main__":
    main()
