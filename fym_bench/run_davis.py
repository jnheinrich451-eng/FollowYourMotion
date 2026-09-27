#!/usr/bin/env python
"""Resumable FYM / backbone runner for the ditflow DAVIS50 manifest.

FYM releases no weights, so every clip is its own tuning run:

  prep         first 21 of the clip's 24 cut frames, resized to 832x480 as a
               whole frame (no crop, so collect.py's resized DAVIS masks stay
               aligned), written lossless; the cut is checked against the
               manifest's sha256 first
  backbone     generate with the clean base model
  fym          data_process -> Collect_attn_map -> spatial LoRA -> temporal LoRA
               -> generate, README recipe at the protocol's frames and size
  fym_biasfix  the same, with the pretrained q/k/v biases carried through the
               head split (not in the release; a check, not a headline row)

Cells land in <runs>/fym_wan/davis/<clip>/subject/<config>/seed<seed>/ with
original.mp4, results.mp4 and done.json (written last), which is what
benchmark/collect.py reads. A cell with done.json is skipped and finished
tuning (tune/tune.json) is reused, so a recycled Colab runtime resumes.
A failure writes failed.json with a reason code and the run moves on.

    python fym_bench/run_davis.py --manifest davis50_local.csv \\
        --runs /content/drive/MyDrive/ditflow/runs \\
        --models /content/models/Wan2.1-T2V-1.3B --clips camel
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fym_bench.videoio import write_lossless  # noqa: E402

RUN_TAG = "fym_wan"
SPATIAL = "self_attn.q_spatial,self_attn.k_spatial,self_attn.v_spatial"
TEMPORAL = "self_attn.q_temporal,self_attn.k_temporal,self_attn.v_temporal"

# README Steps 5-6, recorded once. Frames and size come from the protocol
# instead of the README's 45 frames at 544x544 (see steps.md, Protocol).
RECIPE = {
    "steps_per_epoch": 500, "max_epochs": 3, "learning_rate": 1e-4,
    "lora_rank": 16, "lora_alpha": 16, "accumulate_grad_batches": 1,
    "spatial_lora_wd": 0.1, "temporal_lora_wd": 0.5,
    "use_gradient_checkpointing": True, "split_seed": 0, "tune_seed": 0,
}
CONFIGS = {"backbone": None, "fym": False, "fym_biasfix": True}   # value = copy_bias

FAILURE_CODES = [
    ("oom", r"CUDA out of memory|out of memory|OutOfMemoryError"),
    ("nan", r"\bnan\b|inf.*loss|assert.*finite"),
    ("no_kernel", r"no kernel image is available"),
    ("missing_dep", r"ModuleNotFoundError|ImportError"),
    ("interrupted", r"KeyboardInterrupt"),
]


class StageFailed(Exception):
    def __init__(self, stage, code, detail, log=None):
        super().__init__(f"{stage}: {code}: {detail}")
        self.stage, self.code, self.detail, self.log = stage, code, detail, log


# ------------------------------------------------------------------ helpers ---

def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       text=True).strip().splitlines()[0]
    except Exception:
        return "unknown"


class PeakMemory:
    """Polls nvidia-smi while a stage runs; every stage is its own process."""

    def __enter__(self):
        self.peak, self._stop = 0, threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _poll(self):
        while not self._stop.wait(1.0):
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                               "--format=csv,noheader,nounits"], text=True)
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def repo_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=REPO, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def stage_env(**extra):
    env = dict(os.environ)
    # The repo's own diffsynth must win over a PyPI diffsynth in site-packages.
    env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_stage(stage, cmd, log_path, env):
    """Run one stage as its own process (clean CUDA state, own log). Returns (seconds, peak MB)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log, PeakMemory() as mem:
        log.write(" ".join(map(str, cmd)) + "\n\n")
        log.flush()
        proc = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        code = next((c for c, pat in FAILURE_CODES if re.search(pat, tail, re.IGNORECASE)), "other")
        raise StageFailed(stage, code, f"exit code {proc.returncode}", log_path)
    print(f"      {stage}: {elapsed:.0f}s, peak {mem.peak} MB", flush=True)
    return elapsed, mem.peak


# ---------------------------------------------------------------- reference ---

def prepare_reference(row, n_frames, height, width):
    """First n cut frames, whole-frame resize, after checking the cut's hash."""
    import numpy as np
    from PIL import Image
    src = Path(row["video_path"])
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    digest = hashlib.sha256()
    for p in files:                      # cut.py hashes the JPEG bytes in frame order
        digest.update(p.read_bytes())
    if digest.hexdigest() != row["sha256"]:
        raise StageFailed("prep", "sha_mismatch", f"{src} does not hash to the manifest's sha256")
    if len(files) < n_frames:
        raise StageFailed("prep", "short_clip", f"{src} has {len(files)} frames, need {n_frames}")
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8")) if (src / "meta.json").exists() else {}
    ims = [Image.open(p).convert("RGB") for p in files[:n_frames]]
    frames = np.stack([np.asarray(im.resize((width, height), Image.LANCZOS)) for im in ims])
    info = {"frames_used": [p.name for p in files[:n_frames]],
            "davis_indices": meta.get("indices", [])[:n_frames],
            "native_wh": list(ims[0].size), "resized_wh": [width, height],
            "fps": float(row.get("effective_fps") or meta.get("effective_fps") or 12.0)}
    return frames, info


# ------------------------------------------------------------------- tuning ---

def prepare_tuning(caption, ref_mp4, work, models, sizes):
    """data_process + head classification. Neither touches the split, so every
    FYM config of a clip shares them. Returns (dir, stage seconds, stage peaks, heads)."""
    import torch
    d = work / "dataset"
    logs = work / "shared_logs"
    if d.exists():
        shutil.rmtree(d)
    (d / "train").mkdir(parents=True)
    shutil.copyfile(ref_mp4, d / "train" / "ref.mp4")
    with open(d / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "text"])
        w.writerow(["ref.mp4", caption])
    env = stage_env(FYM_SEED=RECIPE["tune_seed"])
    times, peaks = {}, {}
    times["data_process"], peaks["data_process"] = run_stage("data_process", [
        sys.executable, "examples/wanvideo/EffiVMT_train_wan_t2v.py", "--task", "data_process",
        "--dataset_path", d, "--output_path", work / "data_process",
        "--text_encoder_path", models / "models_t5_umt5-xxl-enc-bf16.pth",
        "--vae_path", models / "Wan2.1_VAE.pth", "--tiled",
        "--num_frames", sizes["frames"], "--height", sizes["height"], "--width", sizes["width"],
    ], logs / "data_process.log", env)
    if not (d / "train" / "ref.mp4.tensors.pth").exists():
        raise StageFailed("data_process", "other", "no .tensors.pth written (clip shorter than --num_frames?)",
                          logs / "data_process.log")
    times["collect_attn"], peaks["collect_attn"] = run_stage("collect_attn", [
        sys.executable, "examples/wanvideo/Collect_attn_map.py",
        "--dataset_path", d, "--dit_path", models / "diffusion_pytorch_model.safetensors",
    ], logs / "collect_attn.log", env)
    heads = [[int(x) for x in layer] for layer in torch.load(d / "head_types.pt")]
    bad = [i for i, layer in enumerate(heads) if len(set(layer)) < 2]
    if bad:
        # split_QKV cannot build an empty spatial or temporal group, so tuning would crash
        raise StageFailed("collect_attn", "empty_head_group",
                          f"layers {bad} have every head of one type", logs / "collect_attn.log")
    return {"dir": d, "logs": logs, "times": times, "peaks": peaks, "heads": heads}


def tune(clip, caption, copy_bias, prep, models, work, cell):
    """Spatial then temporal LoRA. Artifacts and logs are copied to cell/tune."""
    d, out_dir = prep["dir"], cell / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    for log in prep["logs"].glob("*.log"):
        shutil.copyfile(log, out_dir / log.name)
    times, peaks = dict(prep["times"]), dict(prep["peaks"])
    env = stage_env(FYM_SEED=RECIPE["tune_seed"], FYM_HEAD_TYPES=d / "head_types.pt",
                    FYM_SPLIT_SEED=RECIPE["split_seed"], **({"FYM_SPLIT_COPY_BIAS": 1} if copy_bias else {}))
    common = ["--task", "train", "--train_architecture", "lora", "--dataset_path", d,
              "--dit_path", models / "diffusion_pytorch_model.safetensors",
              "--steps_per_epoch", RECIPE["steps_per_epoch"], "--max_epochs", RECIPE["max_epochs"],
              "--learning_rate", RECIPE["learning_rate"], "--lora_rank", RECIPE["lora_rank"],
              "--lora_alpha", RECIPE["lora_alpha"], "--accumulate_grad_batches", RECIPE["accumulate_grad_batches"],
              "--use_gradient_checkpointing"]
    ckpt, seen = {}, {}
    for stage, targets, extra in (
            ("spatial", SPATIAL, ["--spatial_lora_wd", RECIPE["spatial_lora_wd"]]),
            ("temporal", TEMPORAL, ["--train_temporal_lora", "--temporal_lora_wd", RECIPE["temporal_lora_wd"]])):
        out = work / ("biasfix" if copy_bias else "release") / f"lora_{stage}"
        if out.exists():
            shutil.rmtree(out)
        if stage == "temporal":
            extra = extra + ["--pretrained_spatial_lora_path", ckpt["spatial"]]
        times[f"{stage}_lora"], peaks[f"{stage}_lora"] = run_stage(f"{stage}_lora", [
            sys.executable, "examples/wanvideo/EffiVMT_train_wan_t2v_Head.py", *common,
            "--output_path", out, "--out_file_name", f"lora_{stage}", "--lora_target_modules", targets, *extra,
        ], out_dir / f"{stage}_lora.log", env)
        # Some Lightning releases version the name per epoch; the newest file is the last epoch.
        found = sorted(out.glob("*.ckpt"), key=lambda q: q.stat().st_mtime)
        if not found:
            raise StageFailed(f"{stage}_lora", "other", f"no checkpoint in {out}", out_dir / f"{stage}_lora.log")
        ckpt[stage], seen[stage] = found[-1], [q.name for q in found]

    shutil.copyfile(d / "head_types.pt", out_dir / "head_types.pt")
    for stage, src in ckpt.items():
        shutil.copyfile(src, out_dir / f"lora_{stage}.ckpt")
    record = {"clip_id": clip, "tuning_prompt": caption, "copy_bias": copy_bias, "recipe": RECIPE,
              "stage_s": {k: round(v, 1) for k, v in times.items()}, "stage_peak_gpu_mb": peaks,
              "ckpts_seen": seen, "head_types": prep["heads"],
              "temporal_heads_per_layer": [layer.count(0) for layer in prep["heads"]],
              "tune_s": round(sum(times.values()), 1)}
    # Written last: its presence is what marks tuning as reusable.
    (out_dir / "tune.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# --------------------------------------------------------------------- cells ---

def cell_dir(args, clip, cfg):
    return Path(args.runs) / RUN_TAG / "davis" / clip / "subject" / cfg / f"seed{args.seed}"


def is_done(cell):
    res = cell / "results.mp4"
    return (cell / "done.json").exists() and res.exists() and res.stat().st_size > 0


def run_cell(clip, cfg, rows, ref, ref_info, args, work, get_prep):
    cell = cell_dir(args, clip, cfg)
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "failed.json").unlink(missing_ok=True)
    models = Path(args.models)
    shutil.copyfile(ref, cell / "original.mp4")
    subject, caption = rows["subject"]["prompt"], rows["caption"]["prompt"]

    gen_dir = work / cfg
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen = [sys.executable, "fym_bench/generate.py", "--models", models, "--prompt", subject,
           "--out", gen_dir / "results.mp4", "--seed", args.seed, "--height", args.height,
           "--width", args.width, "--num_frames", args.frames, "--fps", ref_info["fps"]]
    tune_rec = None
    if cfg == "backbone":
        gen += ["--mode", "backbone"]
    else:
        copy_bias = CONFIGS[cfg]
        tj = cell / "tune" / "tune.json"
        if tj.exists():
            tune_rec = json.loads(tj.read_text(encoding="utf-8"))
            print(f"    {cfg}: reusing finished tuning", flush=True)
        else:
            print(f"    {cfg}: tuning", flush=True)
            tune_rec = tune(clip, caption, copy_bias, get_prep(), models, work, cell)
        gen += ["--mode", "fym", "--head_types", cell / "tune" / "head_types.pt",
                "--spatial_lora", cell / "tune" / "lora_spatial.ckpt",
                "--temporal_lora", cell / "tune" / "lora_temporal.ckpt",
                "--split_seed", RECIPE["split_seed"], "--lora_rank", RECIPE["lora_rank"],
                "--lora_alpha", RECIPE["lora_alpha"]] + (["--copy_bias"] if copy_bias else [])

    elapsed, peak_gen = run_stage("generate", gen, cell / "generate.log", stage_env())
    gen_stats = json.loads((gen_dir / "results.gen.json").read_text(encoding="utf-8"))
    shutil.copyfile(gen_dir / "results.mp4", cell / "results.mp4")

    peak_tune = max(tune_rec["stage_peak_gpu_mb"].values(), default=0) if tune_rec else 0
    done = {
        "cell": f"{RUN_TAG}/davis/{clip}/subject/{cfg}/seed{args.seed}",
        "clip_id": clip, "prompt_id": "subject", "config": cfg, "seed": args.seed,
        "prompt": subject, "video_path": rows["subject"]["video_path"],
        "elapsed_s": round(elapsed, 1), "sample_s": gen_stats["elapsed_s"],
        "tune_s": tune_rec["tune_s"] if tune_rec else 0,
        "peak_gpu_mb": max(peak_gen, peak_tune), "peak_gpu_mb_generate": peak_gen,
        "gpu": args.gpu, "checkpoint": args.checkpoint, "repo_commit": args.commit,
        "reference": ref_info, "generation": gen_stats["settings"],
        "recipe": RECIPE if tune_rec else None,
        "temporal_heads_per_layer": tune_rec["temporal_heads_per_layer"] if tune_rec else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (cell / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(f"    {cfg}: ok (tune {done['tune_s']:.0f}s, generate {elapsed:.0f}s)", flush=True)


def fail_cell(args, clip, cfg, err):
    cell = cell_dir(args, clip, cfg)
    cell.mkdir(parents=True, exist_ok=True)
    tail = err.log.read_text(encoding="utf-8", errors="replace")[-4000:] if err.log and err.log.exists() else ""
    (cell / "failed.json").write_text(json.dumps({
        "cell": f"{RUN_TAG}/davis/{clip}/subject/{cfg}/seed{args.seed}", "clip_id": clip, "config": cfg,
        "stage": err.stage, "reason_code": err.code, "reason": err.detail, "gpu": args.gpu,
        "timestamp": datetime.now().isoformat(timespec="seconds"), "log_tail": tail,
    }, indent=2), encoding="utf-8")
    print(f"    {cfg}: FAILED at {err.stage} ({err.code}: {err.detail})", flush=True)


def accounting(runs):
    """Attempted / completed / failed-by-reason, recomputed from the run tree."""
    acc = {}
    for d in (Path(runs) / RUN_TAG / "davis").glob("*/subject/*/seed*"):
        a = acc.setdefault(d.parent.name, {"attempted": 0, "completed": 0, "failed": {}})
        a["attempted"] += 1
        if (d / "done.json").exists():
            a["completed"] += 1
        elif (d / "failed.json").exists():
            code = json.loads((d / "failed.json").read_text(encoding="utf-8"))["reason_code"]
            a["failed"][code] = a["failed"].get(code, 0) + 1
    (Path(runs) / RUN_TAG / "accounting.json").write_text(json.dumps(acc, indent=2), encoding="utf-8")
    return acc


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="remapped davis50 manifest (remap.py --verify)")
    ap.add_argument("--runs", required=True, help="run tree root, on Drive")
    ap.add_argument("--models", required=True, help="local Wan2.1-T2V-1.3B folder")
    ap.add_argument("--work", default="/content/fym_work", help="local scratch for tuning")
    ap.add_argument("--configs", default="backbone,fym", help=f"comma list of {list(CONFIGS)}")
    ap.add_argument("--clips", help="comma list of clip_ids (default: every clip in the manifest)")
    ap.add_argument("--limit", type=int, help="stop after N clips")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = args.configs.split(",")
    if set(configs) - CONFIGS.keys():
        sys.exit(f"unknown configs {sorted(set(configs) - CONFIGS.keys())}")
    if (args.frames - 1) % 4 or args.height % 16 or args.width % 16:
        sys.exit("Wan needs frames = 4k+1 and height/width divisible by 16")

    by_clip = {}
    for r in csv.DictReader(open(args.manifest, newline="", encoding="utf-8")):
        by_clip.setdefault(r["clip_id"], {})[r["prompt_id"]] = r
    clips = args.clips.split(",") if args.clips else sorted(by_clip)
    missing = [c for c in clips if c not in by_clip or {"subject", "caption"} - by_clip[c].keys()]
    if missing:
        sys.exit(f"clips without both subject and caption rows in the manifest: {missing}")
    clips = clips[:args.limit] if args.limit else clips

    models = Path(args.models)
    absent = [n for n in ("diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth",
                          "Wan2.1_VAE.pth", "google/umt5-xxl") if not (models / n).exists()]
    if absent:
        sys.exit(f"{models} is missing {absent}")
    rev = models / "REVISION"
    args.checkpoint = rev.read_text(encoding="utf-8").strip() if rev.exists() else "Wan-AI/Wan2.1-T2V-1.3B@unknown"
    args.commit, args.gpu = repo_commit(), gpu_name()
    print(f"{len(clips)} clips x {configs}; checkpoint {args.checkpoint}; repo {args.commit}; gpu {args.gpu}")
    if args.commit.endswith("-dirty"):
        print("  ! uncommitted changes: repo_commit will not identify the code that ran")
    if args.dry_run:
        for c in clips:
            print(f"  {c}: {by_clip[c]['subject']['video_path']}")
        return

    sizes = {"frames": args.frames, "height": args.height, "width": args.width}
    t_all = time.time()
    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip}", flush=True)
        todo = [c for c in configs if not is_done(cell_dir(args, clip, c))]
        if not todo:
            print("    all configs done, skipping", flush=True)
            continue
        rows = by_clip[clip]
        work = Path(args.work) / clip
        work.mkdir(parents=True, exist_ok=True)
        try:
            frames, ref_info = prepare_reference(rows["subject"], args.frames, args.height, args.width)
            ref = work / "original.mp4"
            write_lossless(frames, ref, ref_info["fps"])
        except StageFailed as e:
            for cfg in todo:
                fail_cell(args, clip, cfg, e)
            continue

        prep = {}

        def get_prep():
            # Run once per clip, on the first FYM config that needs it; a failure here
            # fails every FYM config of the clip, since each would fail the same way.
            if "err" in prep:
                raise prep["err"]
            if "val" not in prep:
                try:
                    prep["val"] = prepare_tuning(rows["caption"]["prompt"], ref, work, Path(args.models), sizes)
                except StageFailed as e:
                    prep["err"] = e
                    raise
            return prep["val"]

        for cfg in todo:
            try:
                run_cell(clip, cfg, rows, ref, ref_info, args, work, get_prep)
            except StageFailed as e:
                fail_cell(args, clip, cfg, e)
        print(f"  elapsed so far {(time.time() - t_all) / 60:.0f} min", flush=True)

    print("\naccounting:", json.dumps(accounting(args.runs), indent=2))


if __name__ == "__main__":
    main()
