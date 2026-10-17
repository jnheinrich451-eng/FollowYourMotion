#!/usr/bin/env python
"""Resumable FYM / Wan-T2V floor runner for the ditflow MiraData pairs.

Design: ditflow/docs/fym-pairs-design-v1.md. Same procedure and recipe as run_davis.py (imported, not
copied); only the input side differs:

  reference    the pair's ditflow ref21 (pairs/refs/<pair>__ref21.mp4) -- the exact frames every other
               method is scored against -- resized whole-frame to 832x480 like the DAVIS runs
  tuning text  the reference's own caption (manifest column `caption`)
  prompt       the pair's prompt, identical to every other row of that pair
  output       21 frames at 832x480, mapped back onto the 720x480 reference (exact inverse of the
               whole-frame resize), written as <pair>__<row>__seed<k>.mp4 + .json into pairs/inbox/,
               which ditflow's poster_score_colab.ipynb files and scores

Rows: fym (per-clip spatial + temporal LoRA) and fym_noguide (same checkpoint, no LoRA: FYM's floor).
Run records (logs, LoRAs, done.json / failed.json) go to <runs>/<pair>/<row>/seed<k>/. A cell with
done.json is skipped; finished tuning (tune/tune.json) is reused, so a recycled runtime resumes.

    python fym_bench/run_pairs.py --manifest /content/ditflow/benchmark/pairs_manifest.csv \\
        --refs /content/drive/MyDrive/ditflow_transfer/pairs/refs \\
        --inbox /content/drive/MyDrive/ditflow_transfer/pairs/inbox \\
        --runs /content/drive/MyDrive/ditflow_transfer/fym_runs --models /content/models/Wan2.1-T2V-1.3B
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fym_bench.run_davis import (RECIPE, StageFailed, gpu_name, prepare_tuning,  # noqa: E402
                                 repo_commit, run_stage, stage_env, tune)
from fym_bench.videoio import read_frames, write_lossless  # noqa: E402

ROWS = ("fym_noguide", "fym")          # floor first: cheap, and independent of tuning failures
REF_HW = (480, 720)                    # the ditflow comparison shape


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resize_frames(frames, w, h):
    import numpy as np
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.LANCZOS)) for f in frames])


def prepare_reference(ref_mp4, ref_json, n, height, width):
    """The pair's ref21 -> FYM's input size, whole frame (no crop), as on DAVIS."""
    frames = read_frames(ref_mp4)
    if len(frames) != n:
        raise StageFailed("prep", "short_clip", f"{ref_mp4} has {len(frames)} frames, need {n}")
    if frames.shape[1:3] != REF_HW:
        raise StageFailed("prep", "other", f"{ref_mp4} is {frames.shape[1:3]}, expected {REF_HW}")
    meta = json.loads(Path(ref_json).read_text(encoding="utf-8"))
    info = {"ref21": str(ref_mp4), "ref21_sha256": sha256(ref_mp4), "source_frames": meta.get("source_frames"),
            "source_fps": meta.get("source_fps"), "native_hw": list(REF_HW), "resized_wh": [width, height]}
    return resize_frames(frames, width, height), info, float(meta.get("source_fps") or 12.0)


def map_back(frames):
    """Exact inverse of the whole-frame resize: 832x480 -> 720x480 (the reference's own pixels)."""
    return resize_frames(frames, REF_HW[1], REF_HW[0])


def run_cell(pair, row, r, ref, ref_info, fps, args, work, get_prep):
    cell = Path(args.runs) / pair / row / f"seed{args.seed}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "failed.json").unlink(missing_ok=True)
    models = Path(args.models)
    gen_dir = work / row
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen = [sys.executable, "fym_bench/generate.py", "--models", models, "--prompt", r["prompt"],
           "--out", gen_dir / "results.mp4", "--seed", args.seed, "--height", args.height,
           "--width", args.width, "--num_frames", args.frames, "--fps", fps]
    tune_rec = None
    if row == "fym_noguide":
        gen += ["--mode", "backbone"]
    else:
        tj = cell / "tune" / "tune.json"
        if tj.exists():
            tune_rec = json.loads(tj.read_text(encoding="utf-8"))
            print("    fym: reusing finished tuning", flush=True)
        else:
            print("    fym: tuning (~50 min)", flush=True)
            tune_rec = tune(pair, r["caption"], False, get_prep(), models, work, cell)
        gen += ["--mode", "fym", "--head_types", cell / "tune" / "head_types.pt",
                "--spatial_lora", cell / "tune" / "lora_spatial.ckpt",
                "--temporal_lora", cell / "tune" / "lora_temporal.ckpt",
                "--split_seed", RECIPE["split_seed"], "--lora_rank", RECIPE["lora_rank"],
                "--lora_alpha", RECIPE["lora_alpha"]]

    elapsed, peak = run_stage("generate", gen, cell / "generate.log", stage_env())
    raw = read_frames(gen_dir / "results.mp4")
    if len(raw) != args.frames:
        raise StageFailed("generate", "other", f"{len(raw)} frames generated, expected {args.frames}")
    shutil.copyfile(gen_dir / "results.mp4", cell / "results.mp4")          # raw output, kept
    out = Path(args.inbox) / f"{pair}__{row}__seed{args.seed}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_lossless(map_back(raw), out, fps)                                  # comparison shape, for scoring
    rec = {
        "pair": pair, "row": row, "seed": args.seed, "prompt": r["prompt"],
        "tuning_caption": r["caption"] if tune_rec else None, "reference": ref_info,
        "geometry": "generated at 832x480 from the whole 720x480 ref21; mapped back by the inverse resize",
        "frames_used": list(range(args.frames)), "elapsed_s": round(elapsed, 1),
        "tune_s": tune_rec["tune_s"] if tune_rec else 0, "peak_gpu_mb": peak, "gpu": args.gpu,
        "checkpoint": args.checkpoint, "repo_commit": args.commit, "recipe": RECIPE if tune_rec else None,
        "raw_output": str(cell / "results.mp4"), "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (cell / "done.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")         # written last
    print(f"    {row}: ok (tune {rec['tune_s']:.0f}s, generate {elapsed:.0f}s) -> {out.name}", flush=True)


def fail_cell(args, pair, row, err):
    cell = Path(args.runs) / pair / row / f"seed{args.seed}"
    cell.mkdir(parents=True, exist_ok=True)
    tail = ""
    if err.log and Path(err.log).exists():
        tail = Path(err.log).read_text(encoding="utf-8", errors="replace")[-3000:]
    (cell / "failed.json").write_text(json.dumps({"pair": pair, "row": row, "stage": err.stage,
                                                  "reason_code": err.code, "reason": err.detail,
                                                  "log_tail": tail}, indent=1), encoding="utf-8")
    print(f"    {row}: FAILED at {err.stage} ({err.code}: {err.detail})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="ditflow benchmark/pairs_manifest.csv (needs a caption column)")
    ap.add_argument("--refs", required=True, help="Drive pairs/refs (the ref21 files and their json)")
    ap.add_argument("--inbox", required=True, help="Drive pairs/inbox (scoring picks outputs up here)")
    ap.add_argument("--runs", required=True, help="Drive folder for FYM run records")
    ap.add_argument("--models", required=True, help="local Wan2.1-T2V-1.3B folder")
    ap.add_argument("--work", default="/content/fym_pairs_work")
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--pairs", help="comma list of <clip>__<target> (default: every gated manifest row)")
    ap.add_argument("--allow-provisional", action="store_true", help="include rows whose window is not confirmed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["camera_gate_pass"] == "True"]
    if not args.allow_provisional:
        held = [r["clip_id"] for r in rows if r.get("start_source", "").startswith("PROVISIONAL")]
        rows = [r for r in rows if not r.get("start_source", "").startswith("PROVISIONAL")]
        if held:
            print(f"held back {len(held)} rows with a provisional window (use --allow-provisional once confirmed)")
    if any(not r.get("caption") for r in rows):
        sys.exit("manifest rows without a caption: " + ", ".join(r["clip_id"] for r in rows if not r.get("caption")))
    want = set(args.pairs.split(",")) if args.pairs else None
    todo_rows = args.rows.split(",")
    models = Path(args.models)
    rev = models / "REVISION"
    args.checkpoint = rev.read_text(encoding="utf-8").strip() if rev.exists() else "Wan-AI/Wan2.1-T2V-1.3B@unknown"
    args.commit, args.gpu = repo_commit(), gpu_name()
    sizes = {"frames": args.frames, "height": args.height, "width": args.width}
    print(f"{len(rows)} gated pairs x {todo_rows}; seed {args.seed}; {args.checkpoint}; repo {args.commit}; gpu {args.gpu}")

    t_all = time.time()
    for i, r in enumerate(rows, 1):
        pair = f"{r['clip_id']}__{r['target']}"
        if want and pair not in want:
            continue
        ref_mp4, ref_json = Path(args.refs) / f"{pair}__ref21.mp4", Path(args.refs) / f"{pair}__ref21.json"
        if not ref_mp4.exists():
            print(f"[{i}/{len(rows)}] {pair}: no ref21 yet (run ditflow_pairs_colab Cell 2 first), skipped", flush=True)
            continue
        todo = [x for x in todo_rows if not (Path(args.runs) / pair / x / f"seed{args.seed}" / "done.json").exists()]
        print(f"[{i}/{len(rows)}] {pair}: {todo or 'all done'}", flush=True)
        if not todo or args.dry_run:
            continue
        work = Path(args.work) / pair
        work.mkdir(parents=True, exist_ok=True)
        try:
            frames, ref_info, fps = prepare_reference(ref_mp4, ref_json, args.frames, args.height, args.width)
            ref = work / "original.mp4"
            write_lossless(frames, ref, fps)
        except StageFailed as e:
            for x in todo:
                fail_cell(args, pair, x, e)
            continue
        prep = {}

        def get_prep():
            if "err" in prep:
                raise prep["err"]
            if "val" not in prep:
                try:
                    prep["val"] = prepare_tuning(r["caption"], ref, work, models, sizes)
                except StageFailed as e:
                    prep["err"] = e
                    raise
            return prep["val"]

        for x in todo:
            try:
                run_cell(pair, x, r, ref, ref_info, fps, args, work, get_prep)
            except StageFailed as e:
                fail_cell(args, pair, x, e)
        print(f"  elapsed so far {(time.time() - t_all) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
