# FYM on the DiTFlow DAVIS benchmark — test instructions

## Goal
Measure whether Follow-Your-Motion transfers reference motion better than its own base
model, on the same clips, prompts, frames, seed and metrics as the saved DiTFlow benchmark.
The claim is the paired per-clip gain `fym − backbone`, with a bootstrap CI.
Do NOT compare absolute scores against the saved DiTFlow/CogVideoX-5B numbers: different
backbone, so a separate table.

## Step 0 — answer before any GPU run (record the answers)
1. Released checkpoint: general model that takes a new reference video at inference, or
   LoRAs tuned on FYM's demo videos? If demo LoRAs → tune once per DAVIS clip (see Tuning).
2. Base model: Wan2.1 T2V or I2V, and which size/revision? If I2V, the first frame must come
   from the backbone row's output (frame 0), never from the reference video.
3. Supported frame counts and resolution.

## Files to copy from the ditflow repo
| File | Purpose |
|---|---|
| `benchmark/davis50.csv` | Manifest: 50 clips × {caption, subject} prompts, frame-folder paths, sha256 |
| `benchmark/davis_subject_map.csv` | Subject-swap table (reference only) |
| `benchmark/remap.py` | Rewrites `video_path` prefixes for your machine; `--verify` checks they exist |
| `benchmark/collect.py` | Scoring (CoTracker MF, direction, CLIP, LPIPS) |
| `benchmark/analyse.py` | Paired bootstrap contrasts |
| `docs/scores_davis.csv` | Saved DiTFlow anchor (CogVideoX-5B) — context only |
| Frame folders `E:\bench\packed\davis\<clip_id>\` | The cut 24-frame JPEG windows. Use as is, do not re-cut |
| DAVIS `Annotations/480p/` | Subject masks for the masked metrics |

Fix the paths first:
    python benchmark/remap.py --manifest benchmark/davis50.csv --out davis50_local.csv \
        --from "E:\bench\packed" --to "<your packed root>" --verify

## Protocol (held fixed; everything else = FYM repo defaults, recorded)
- Clips: all 50 `clip_id`s in the manifest.
- Generation prompt: the `prompt_id=subject` row (headline).
- Tuning prompt (if tuning): the `prompt_id=caption` row of the same clip.
- Reference frames: the first 21 frames (sorted) of each folder (2.0 s window at ~12 fps).
  If FYM needs a different count, subsample from the same window and record the indices.
- Seed: 1.
- Method internals (LoRA rank, steps, LR, sampler, CFG, resolution): FYM repo defaults,
  written down once.

Rows per clip:
- `fym` — FYM output.
- `backbone` — same base checkpoint, same subject prompt, seed, resolution, frames and steps,
  no FYM weights. Required: all claims are contrasts against it.

## Output layout (collect.py reads this directly)
    <runs>/fym_wan/davis/<clip_id>/subject/<config>/seed1/
        original.mp4   # the reference frames the method used, at output resolution/frame count
        results.mp4    # generated video
        done.json      # written last, only after both videos exist

`done.json` minimum fields:
    {"cell": "fym_wan/davis/<clip_id>/subject/<config>/seed1",
     "clip_id": "<clip_id>", "prompt_id": "subject", "config": "fym|backbone", "seed": 1,
     "elapsed_s": <generation s>, "tune_s": <tuning s or 0>, "peak_gpu_mb": ..., "gpu": "...",
     "checkpoint": "<HF id + revision>", "repo_commit": "<FYM git sha>"}

## Run order
1. Pilot one clip (`camel`), both rows. Run collect.py on it. Check that the videos look
   right and the CSV has values. Only then run all 50.
2. Full run, both rows. Log failures with a reason (oom / nan / other); don't silently skip.

## Scoring
    python benchmark/collect.py --runs <runs> --manifest davis50_local.csv \
        --annotations <DAVIS>/Annotations/480p --out scores_fym.csv \
        --cotracker-ref <pinned tag/commit> --scratch ""
    python benchmark/collect.py --runs <runs> --manifest davis50_local.csv \
        --annotations <DAVIS>/Annotations/480p --out scores_fym.csv \
        --cotracker-ref <same pin> --scratch "" --control c1
    python benchmark/analyse.py --scores scores_fym.csv --reference backbone

## What to report
- Metrics: `mf` (primary), `mf_masked`, `dir_cos`, `iq`, `lpips_bg`.
  Dropped: `temp_cons`, `clip_i_subject` (a frozen video beats every method on them).
  No claims from: `dir_cos_masked`, `subject_consistency`.
- Paired Δ `fym − backbone` per metric, 10,000-resample bootstrap over clips, 95% CI.
  A difference is real only if the CI excludes zero. No bolding, no winner claims.
- C1 row as the floor.
- Split results by `subjects` (single vs multi).
- Run accounting: attempted / completed / failed by reason; tuning time and GPU per clip.
- A contact sheet of a few clips (reference / backbone / fym), so the metrics are backed
  by what the videos actually show.

## Known limits (state them)
- One seed; seed-to-seed variance unmeasured.
- `mf_masked` uses masks from the reference, so it's weaker when the generated subject
  isn't where the reference subject was.
- At N≈50, method rankings on MF are not stable. Report gains over the backbone, not a ranking.
- DAVIS has no camera ground truth; camera-vs-object claims need the MiraData split.
