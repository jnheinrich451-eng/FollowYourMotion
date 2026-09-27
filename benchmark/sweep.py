#!/usr/bin/env python
"""Resumable sweep runner for DiTFlow benchmark cells.

One cell = (clip, prompt, config, seed). Every cell gets its own output
directory, which is what stops `original.mp4` -- overwritten by every run of
motion_guidance.py -- from detaching the reference video from the output it
produced. After a hundred cells you cannot reconstruct that pairing by hand.

A cell counts as done only once `done.json` is written, and that happens after
the outputs are verified on disk. A run killed mid-cell leaves no marker, so
re-issuing the same command resumes exactly where it stopped. Colab recycles
runtimes without warning and a sweep should survive it.

    # plan the sweep without touching the GPU
    python benchmark/sweep.py --manifest benchmark/davis.csv --out E:/bench/runs --dry-run

    # smoke test: one cell, then stop -- gives you a real per-cell time
    python benchmark/sweep.py --manifest benchmark/davis.csv --out E:/bench/runs --limit 1

    # the real thing
    python benchmark/sweep.py --manifest benchmark/davis.csv \
        --out E:/bench/runs --prune-embeds

--out may sit on any drive; it need not be near the repo, and paths containing
spaces are safe because commands are passed to subprocess as a list rather than
through a shell. On Colab, point --out at /content/drive/MyDrive/... instead, so
a recycled runtime loses nothing.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
GPU_NAME = None      # filled once at startup, recorded on every cell
MANIFEST_COLUMNS = ["clip_id", "split", "video_path", "prompt_id", "prompt"]

# A runner says how to invoke one method. Everything that is not a
# hyperparameter lives here, so a second repo is a YAML edit rather than a code
# change -- each external method gets its own interpreter (their diffusers pins
# will conflict), its own working directory, and its own output filename.
#
# Placeholders resolve against the manifest row, plus {out} and {seed}, plus any
# scalar at the top of the grid file. Unresolved ones are caught during --dry-run
# rather than forty cells into a sweep.
DEFAULT_RUNNERS = {
    "ditflow": {
        "python": None,                # None -> the interpreter running sweep.py
        "cwd": None,                   # None -> this repo
        "script": "motion_guidance.py",
        "args": ["-v", "{video_path}", "-p", "{prompt}",
                 "--model", "{model}", "-n", "{video_length}",
                 "--output_path", "{out}", "--seed", "{seed}"],
        "expect": "results.mp4",
    }
}


# ---------------------------------------------------------------- manifest ---

def read_manifest(path):
    """Rows of (clip_id, split, video_path, prompt_id, prompt)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"manifest {path} has no rows")
    missing = [c for c in MANIFEST_COLUMNS if c not in rows[0]]
    if missing:
        sys.exit(f"manifest {path} is missing columns: {', '.join(missing)}")

    for i, r in enumerate(rows, start=2):  # start=2 -> spreadsheet line number
        for col in MANIFEST_COLUMNS:
            if not (r.get(col) or "").strip():
                sys.exit(f"manifest {path} line {i}: empty {col}")
        # A seeded prompt that was never edited would run as a duplicate of the
        # Caption class under another prompt_id, silently corrupting that column.
        if r["prompt"].strip().startswith("TODO"):
            sys.exit(f"manifest {path} line {i}: prompt still marked TODO "
                     f"({r['clip_id']}/{r['prompt_id']}) -- edit it or drop the row")
        # Any *_path column is resolved against the repo, so optional columns
        # such as traj_path or first_frame_path work the same as video_path.
        for col, val in list(r.items()):
            if col.endswith("_path") and val and not os.path.isabs(val):
                r[col] = str((REPO / val).resolve())
    return rows


def init_manifest(video_dir, out_path, split):
    """Emit a manifest stub listing every video found, prompts left blank.

    Prompts are the one thing that cannot be inferred. Filling them by hand is
    the point: DiTFlow uses three prompt classes per video (Caption / Subject /
    Scene) and which class a row belongs to has to be a deliberate choice.
    """
    videos = sorted(p for p in Path(video_dir).iterdir()
                    if p.suffix.lower() in (".mp4", ".gif", ".webm"))
    if not videos:
        sys.exit(f"no videos found in {video_dir}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for v in videos:
            w.writerow({"clip_id": v.stem, "split": split,
                        "video_path": os.path.relpath(v, REPO).replace("\\", "/"),
                        "prompt_id": "subject", "prompt": ""})
    print(f"wrote {out_path} with {len(videos)} rows -- fill in the prompt column")


# -------------------------------------------------------------------- grid ---

def read_grid(path):
    grid = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("seeds", "configs"):
        if key not in grid:
            sys.exit(f"grid {path} is missing '{key}'")
    grid["runners"] = {**DEFAULT_RUNNERS, **(grid.get("runners") or {})}
    for name, cfg in grid["configs"].items():
        runner = (cfg or {}).get("runner", "ditflow")
        if runner not in grid["runners"]:
            sys.exit(f"config '{name}' names runner '{runner}', which is not defined")
    return grid


def runner_for(cell, grid):
    cfg = grid["configs"][cell["config"]] or {}
    return grid["runners"][cfg.get("runner", "ditflow")]


def spec_hash(cell, grid):
    """Digest of everything that determines this cell's command.

    Scoped to the cell, not the whole grid: defining a runner for a second
    method must not mark every already-finished cell of the first as drifted.
    """
    cfg = grid["configs"][cell["config"]] or {}
    payload = {
        "scalars": {k: v for k, v in grid.items() if not isinstance(v, (dict, list))},
        "config": cfg,
        "runner": grid["runners"][cfg.get("runner", "ditflow")],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL, text=True).strip()[:12]
    except Exception:
        return "unknown"


# ------------------------------------------------------- run-time telemetry ---
# These fields exist only while a cell is running and cannot be reconstructed
# afterwards. Wall-clock is the Time(s) column; peak memory justifies tiering
# decisions; a reason code turns "some runs failed" into a run-accounting table
# of attempted / completed / excluded. Backfilled from memory they are guesses.

# Ordered: the first pattern that matches wins, so put the specific ones first.
FAILURE_CODES = [
    ("oom", r"CUDA out of memory|out of memory|OutOfMemoryError"),
    ("nan", r"nan|inf.*loss|assert.*finite"),
    ("no_kernel", r"no kernel image is available"),
    ("missing_dep", r"ModuleNotFoundError|ImportError"),
    ("checkpoint", r"HFValidationError|RepositoryNotFound|401 Client Error"),
    ("interrupted", r"KeyboardInterrupt"),
]


def classify_failure(log_tail):
    """Reason code for the run-accounting table, or 'other'."""
    import re
    for code, pattern in FAILURE_CODES:
        if re.search(pattern, log_tail, re.IGNORECASE):
            return code
    return "other"


def gpu_query(fields):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True).strip().splitlines()
        return out[0].split(", ") if out else None
    except Exception:
        return None


class PeakMemory:
    """Sample whole-device memory while a subprocess runs.

    The generation happens in a child process, so max_memory_allocated() here
    would read zero. Polling nvidia-smi measures the whole device rather than
    the process, which on a dedicated GPU is the number that matters anyway --
    it is what decides whether the config fits.
    """

    def __init__(self, interval=3.0):
        self.interval, self.peak, self._stop = interval, None, None

    def __enter__(self):
        import threading
        self._stop = threading.Event()

        def poll():
            while not self._stop.wait(self.interval):
                v = gpu_query("memory.used")
                if v:
                    try:
                        self.peak = max(self.peak or 0, int(float(v[0])))
                    except ValueError:
                        pass

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)
        return False


def write_env(out_root, grid_path):
    """One environment snapshot per sweep root, referenced by every cell.

    A repo updated mid-sweep makes early and late runs different methods, so the
    commit and the library versions have to be on record at run time.
    """
    def version(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    gpu = gpu_query("name,memory.total,driver_version")
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "grid": str(grid_path),
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "diffusers": version("diffusers"),
        "transformers": version("transformers"),
        "gpu": gpu[0] if gpu else None,
        "gpu_total_mb": gpu[1] if gpu else None,
        "driver": gpu[2] if gpu else None,
    }
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (root / f"run_env_{stamp}.json").write_text(json.dumps(env, indent=2),
                                                encoding="utf-8")
    return env


# ------------------------------------------------------------------- cells ---

def build_cells(rows, grid):
    cells = []
    for row in rows:
        for config in grid["configs"]:
            for seed in grid["seeds"]:
                cells.append({**row, "config": config, "seed": int(seed),
                              "_tag": grid.get("run_tag") or ""})
    return cells


def cell_parts(cell):
    """Path components for a cell, run_tag first when the grid sets one.

    Without the tag, a CogVideoX-5B sweep would land on top of a 2B sweep: the
    directory names are identical, is_done() would see the 2B done.json and skip
    every cell. The spec_hash mismatch warns, but a warning is not a re-run.
    """
    tag = cell.get("_tag") or ""
    return ([tag] if tag else []) + [
        cell["split"], cell["clip_id"], cell["prompt_id"],
        cell["config"], "seed" + str(cell["seed"])]


def cell_dir(out_root, cell):
    return Path(out_root).joinpath(*cell_parts(cell))


def cell_key(cell):
    return "/".join(cell_parts(cell))


def is_done(d, expect):
    """Done means the marker exists AND the output is really on disk.

    Checking the marker alone is not enough: Drive can report a written file
    that is still syncing, and a zero-byte output would otherwise be counted as
    a finished cell and never re-run.
    """
    marker, video = d / "done.json", d / expect
    return marker.exists() and video.exists() and video.stat().st_size > 0


def runner_cwd(runner):
    return Path(runner["cwd"]).expanduser().resolve() if runner.get("cwd") else REPO


def build_cmd(cell, grid, d):
    runner = runner_for(cell, grid)
    # Placeholder namespace: every grid scalar (model, video_length, ...), then
    # the manifest row and the cell's own fields, then the per-cell output dir.
    scalars = {k: v for k, v in grid.items() if not isinstance(v, (dict, list))}
    ns = {**scalars, **cell, "out": str(d)}

    cwd = runner_cwd(runner)
    cmd = [runner.get("python") or sys.executable, str(cwd / runner["script"])]
    for a in runner["args"]:
        try:
            cmd.append(str(a).format(**ns))
        except KeyError as missing:
            sys.exit(f"config '{cell['config']}': runner arg {a!r} needs {missing}, "
                     f"which is in neither the manifest nor the grid")
    for k, v in (grid["configs"][cell["config"]] or {}).items():
        if k == "runner":
            continue
        if isinstance(v, bool):
            if v:
                cmd.append("--" + k)
        else:
            cmd += ["--" + k, str(v)]
    return cmd


# ----------------------------------------------------------------- running ---

def run_cell(cell, grid, d, shash, prune_embeds):
    """Execute one cell in a subprocess. Returns (ok, elapsed_seconds).

    A subprocess per cell costs a model reload, but it isolates CUDA memory and
    guarantees no state -- KV injection flags, cached motion features, the RNG --
    leaks from one cell into the next. That class of bug does not crash; it
    quietly corrupts a column of the results table.
    """
    runner = runner_for(cell, grid)
    d.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(cell, grid, d)
    started = time.time()

    with open(d / "log.txt", "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        with PeakMemory() as mem:
            proc = subprocess.run(cmd, cwd=str(runner_cwd(runner)), stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started

    tail = (d / "log.txt").read_text(encoding="utf-8", errors="replace")[-4000:]
    video = d / runner["expect"]

    if proc.returncode != 0 or not video.exists() or video.stat().st_size == 0:
        reason = ("exit code " + str(proc.returncode) if proc.returncode != 0
                  else runner["expect"] + " missing or empty")
        (d / "failed.json").write_text(json.dumps({
            "cell": cell_key(cell), "reason": reason,
            "reason_code": classify_failure(tail),
            "config": cell["config"], "clip_id": cell["clip_id"],
            "prompt_id": cell["prompt_id"], "seed": cell["seed"],
            "peak_gpu_mb": mem.peak, "gpu": GPU_NAME, "cmd": cmd,
            "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "log_tail": tail,
        }, indent=2), encoding="utf-8")
        return False, elapsed

    if prune_embeds:
        # Optimised latents/embeddings are written every guidance timestep and
        # are not needed for MF or IQ. On Drive they dominate the sweep's size.
        shutil.rmtree(d / "embeds", ignore_errors=True)

    outputs = {p.name: p.stat().st_size for p in sorted(d.glob("*.mp4"))}
    (d / "failed.json").unlink(missing_ok=True)
    (d / "done.json").write_text(json.dumps({
        "cell": cell_key(cell), "cmd": cmd,
        "clip_id": cell["clip_id"], "prompt_id": cell["prompt_id"],
        "prompt": cell["prompt"], "video_path": cell["video_path"],
        "config": cell["config"], "seed": cell["seed"],
        "runner": (grid["configs"][cell["config"]] or {}).get("runner", "ditflow"),
        "grid_scalars": {k: v for k, v in grid.items()
                         if not isinstance(v, (dict, list))},
        "spec_hash": shash, "git_sha": git_sha(),
        "elapsed_s": round(elapsed, 1),
        "peak_gpu_mb": mem.peak, "gpu": GPU_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "outputs": outputs,
    }, indent=2), encoding="utf-8")
    return True, elapsed


def fmt(seconds):
    return str(timedelta(seconds=int(seconds)))


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="CSV: " + ",".join(MANIFEST_COLUMNS))
    ap.add_argument("--grid", default=str(REPO / "benchmark" / "grid.yaml"))
    ap.add_argument("--out", help="output root (put this on Drive when on Colab)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--limit", type=int, help="run at most N pending cells then stop")
    ap.add_argument("--only-split", action="append", default=[])
    ap.add_argument("--only-config", action="append", default=[])
    ap.add_argument("--only-clip", action="append", default=[])
    ap.add_argument("--only-prompt", action="append", default=[],
                    metavar="CLASS", help="restrict to prompt classes, e.g. subject")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-run cells that previously failed")
    ap.add_argument("--prune-embeds", action="store_true",
                    help="delete embeds/ after a successful cell to save disk")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--init-manifest", metavar="VIDEO_DIR",
                    help="scan a directory of videos and emit a manifest stub")
    ap.add_argument("--split", default="davis", help="split name for --init-manifest")
    args = ap.parse_args()

    if args.init_manifest:
        if not args.manifest:
            sys.exit("--init-manifest needs --manifest to say where to write")
        return init_manifest(args.init_manifest, args.manifest, args.split)
    if not args.manifest or not args.out:
        sys.exit("--manifest and --out are both required")

    global GPU_NAME
    rows = read_manifest(args.manifest)
    grid = read_grid(args.grid)
    cells = build_cells(rows, grid)

    if args.only_split:
        cells = [c for c in cells if c["split"] in args.only_split]
    if args.only_config:
        cells = [c for c in cells if c["config"] in args.only_config]
    if args.only_clip:
        cells = [c for c in cells if c["clip_id"] in args.only_clip]
    if args.only_prompt:
        cells = [c for c in cells if c["prompt_id"] in args.only_prompt]
    if not cells:
        sys.exit("no cells match those filters")

    pending, done, failed = [], 0, 0
    for c in cells:
        d = cell_dir(args.out, c)
        if is_done(d, runner_for(c, grid)["expect"]):
            done += 1
            # A cell finished under a different grid is not comparable to the
            # rest of the table. Say so loudly; do not silently re-run it.
            try:
                prev = json.loads((d / "done.json").read_text(encoding="utf-8"))
                if prev.get("spec_hash") != spec_hash(c, grid):
                    print("  ! " + cell_key(c) + " was run under spec "
                          + str(prev.get("spec_hash")) + ", current is "
                          + spec_hash(c, grid))
            except Exception:
                pass
        elif (d / "failed.json").exists() and not args.retry_failed:
            failed += 1
        else:
            pending.append(c)

    scalars = " ".join(k + "=" + str(v) for k, v in grid.items()
                       if not isinstance(v, (dict, list)))
    print("grid " + Path(args.grid).name + "  " + scalars
          + "  git=" + git_sha())
    print(str(len(cells)) + " cells: " + str(done) + " done, " + str(failed)
          + " failed (use --retry-failed), " + str(len(pending)) + " pending")

    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("nothing to do")
        return

    # Build every command up front. A manifest missing a column that some
    # runner's args reference should stop the sweep here, not forty cells in.
    for c in pending:
        build_cmd(c, grid, cell_dir(args.out, c))

    # Every input a pending cell names must exist before the first model loads.
    # The expensive failure is not a bad path, it is a bad path discovered N
    # times: on Colab /content is wiped when the runtime recycles while --out on
    # Drive survives, so a resumed sweep finds its results intact and its inputs
    # gone, then fails identically for every remaining cell with a reason code
    # that describes the symptom. One stat per distinct path costs nothing.
    inputs = {}
    for c in pending:
        for col, val in c.items():
            if col.endswith("_path") and val:
                inputs.setdefault(val, col)
    absent = sorted((p, col) for p, col in inputs.items() if not Path(p).exists())
    if absent:
        print(f"{len(absent)} of {len(inputs)} input paths do not exist:")
        for p, col in absent[:5]:
            print(f"  [{col}] {p}")
        if len(absent) > 5:
            print(f"  ... and {len(absent) - 5} more")
        sys.exit("inputs are missing -- re-run the data/remap step "
                 "(on Colab, /content is wiped when the runtime recycles)")

    if args.dry_run:
        print("\n-- dry run, " + str(len(pending)) + " cells would run --")
        for c in pending[:20]:
            runner = (grid["configs"][c["config"]] or {}).get("runner", "ditflow")
            print("  " + cell_key(c) + "   [" + runner + "]")
            print("    " + " ".join(build_cmd(c, grid, cell_dir(args.out, c))))
        if len(pending) > 20:
            print("  ... and " + str(len(pending) - 20) + " more")
        return

    env = write_env(args.out, args.grid)
    GPU_NAME = env["gpu"]
    print(f"env: {env['gpu'] or 'no GPU'} | torch {env['torch']} | "
          f"diffusers {env['diffusers']}")

    durations, ok_count, fail_count = [], 0, 0
    sweep_started = time.time()

    for i, c in enumerate(pending, start=1):
        d = cell_dir(args.out, c)
        eta = ("  eta " + fmt(statistics.median(durations) * (len(pending) - i + 1))
               if durations else "")
        print("[" + str(i) + "/" + str(len(pending)) + "] " + cell_key(c) + eta, flush=True)

        ok, elapsed = run_cell(c, grid, d, spec_hash(c, grid), args.prune_embeds)
        durations.append(elapsed)
        if ok:
            ok_count += 1
            print("    ok in " + fmt(elapsed), flush=True)
        else:
            fail_count += 1
            print("    FAILED in " + fmt(elapsed) + " -- see " + str(d / "log.txt"), flush=True)
            if args.stop_on_fail:
                break

    print("\n" + str(ok_count) + " ok, " + str(fail_count) + " failed, total "
          + fmt(time.time() - sweep_started))
    if fail_count:
        print("re-run the same command to retry unfinished cells "
              "(add --retry-failed to include the failures)")


if __name__ == "__main__":
    main()
