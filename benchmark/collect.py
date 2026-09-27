#!/usr/bin/env python
"""Score a sweep into a tidy CSV: one row per cell, one column per metric.

Walks the run tree for `done.json`, joins each cell back to its manifest row,
and computes metrics from the reference and generated videos sitting beside it.
Resumable -- cells already present in the output are skipped, so a killed run
resumes rather than restarting.

    python benchmark/collect.py --runs E:/bench/runs --manifest benchmark/davis50.csv \
        --annotations E:/DAVIS/Annotations/480p --out benchmark/scores_davis.csv

Metrics, and why each is here:

  mf              DiTFlow's Motion Fidelity, whole frame. Required for GATE 5
                  (port equivalence) because it is the published number.
  mf_masked       The same, restricted to the annotated subject. MF takes a max
                  over every track in the frame, so with a 55x55 grid it asks
                  "does ANY track resemble each reference track", not "does the
                  subject move correctly". Masking removes the escape route.
  dir_cos         Cosine between the mean net displacement of reference and
                  generated tracks. MF is near-invariant to a global direction
                  flip -- one correctly-moving track in 200 lifts it from -0.99
                  to +0.99 -- so direction needs its own column or an inverted
                  output scores as a success.
  dir_cos_masked  The same on subject tracks only.
  iq              CLIPScore between frames and the prompt. DiTFlow's IQ.
  temp_cons       CLIP cosine between consecutive frames. Reported because the
                  literature reports it; a frozen video scores ~1.0, which is
                  what control row C1 exists to expose.
  subject_consistency
                  DINO cosine between the subject crop at frame 0 and every
                  later frame, within the generated video. Not against the
                  reference: the subject is meant to change, so reference
                  similarity would penalise success.

mf and dir_cos are deliberately separate: the pair is the evidence for whether
the metric can see a failure a viewer can.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ metrics ---

def _unit_displacements(tracks):
    """(N,T,2) tracks -> unit per-step displacement vectors, (N,T-1,2)."""
    d = np.diff(tracks, axis=1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-8)


def motion_fidelity(ref, gen):
    """DiTFlow's MF, reimplemented from eval/motion_fidelity_score.py.

    Kept identical on purpose -- including the max over generated tracks and the
    identity subtraction -- because GATE 5 compares against their published
    number and a 'fixed' MF would not be the same quantity.
    """
    t = min(ref.shape[1], gen.shape[1])
    d1, d2 = _unit_displacements(ref[:, :t]), _unit_displacements(gen[:, :t])
    sim = np.einsum("ntc,mtc->nmt", d1, d2).mean(-1)
    sim = sim - np.eye(sim.shape[0], sim.shape[1])
    return float(sim.max(axis=1).mean())


def direction_cosine(ref, gen):
    """Cosine between mean net displacement of the two track sets.

    Net rather than per-step: a subject that moves out and back has near-zero
    net displacement, and so does its match, which is the honest answer. Per-step
    means would cancel direction entirely.
    """
    a = (ref[:, -1] - ref[:, 0]).mean(axis=0)
    b = (gen[:, -1] - gen[:, 0]).mean(axis=0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")          # no net motion -- direction is undefined
    return float(np.dot(a, b) / (na * nb))


# ------------------------------------------------------------------- inputs ---

def read_frames(path, scratch=None):
    """Decode a video to (T,H,W,3).

    Copies to local disk first when `scratch` is given. The run tree usually
    lives on a Drive FUSE mount, and streaming a decode off it can stall
    indefinitely -- a whole-file copy either succeeds or raises, which is a
    failure mode you can see and retry.
    """
    import imageio.v3 as iio
    src = Path(path)
    if scratch:
        import shutil
        local = Path(scratch) / src.name
        shutil.copyfile(src, local)
        src = local
    return np.stack([np.asarray(f)[..., :3]
                     for f in iio.imiter(str(src), plugin="FFMPEG")])


def subject_mask(ann_dir, clip_id, size):
    """First-frame binary mask of all annotated instances, resized to `size`.

    CoTracker seeds its grid from frame 0, so only that frame's mask matters.
    DAVIS annotates at 854x480 while the videos are resized to 720x480, hence
    the resize -- a mask at the wrong scale silently seeds tracks off-subject.
    """
    from PIL import Image
    p = Path(ann_dir) / clip_id / "00000.png"
    if not p.exists():
        return None
    m = Image.open(p).resize(size, Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def write_csv(path, rows_by_cell):
    """Rewrite the whole file, union of every row's columns.

    Whole-file rather than append: the header is fixed at first write, so a
    later run adding masked columns could neither widen it nor update a row
    without duplicating it. Called every --flush-every cells, because `path` is
    usually on a Drive FUSE mount where each write costs real latency.
    """
    cols = []
    for r in rows_by_cell.values():
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows_by_cell.values())


def crop_to_mask(frames, mask, size=224):
    """Crop every frame to the mask's bounding box, resized. None if too small."""
    from PIL import Image
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return [np.asarray(Image.fromarray(f[y0:y1, x0:x1]).resize((size, size)))
            for f in frames]


# ------------------------------------------------------------------ runners ---

class Tracker:
    """CoTracker, loaded once. Pin the ref: D9 wants one tracker for all rows.

    Caches by video content hash. Every config of a clip writes its own
    original.mp4, but they are the same reference through the same resize -- so
    without a cache the reference is re-tracked once per config. On a 6-config
    grid that is 12 tracker calls per clip where 7 suffice, and tracking is the
    entire cost of scoring.
    """

    def __init__(self, ref=None, grid_size=55, device="cuda"):
        import torch
        repo = "facebookresearch/co-tracker" + (f":{ref}" if ref else "")
        if not ref:
            print("  ! CoTracker is unpinned -- an upstream change mid-sweep "
                  "would split MF across two trackers (see D9)")
        self.torch = torch
        self.grid_size = grid_size
        self.device = device
        self.model = torch.hub.load(repo, "cotracker3_offline").to(device)
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, frames, mask=None):
        import hashlib
        key = (hashlib.sha1(frames.tobytes()).hexdigest(),
               None if mask is None else hashlib.sha1(mask.tobytes()).hexdigest())
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        t = self.torch
        v = t.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        m = None if mask is None else t.from_numpy(mask)[None][None].float().to(self.device)
        with t.no_grad():
            tracks, _ = self.model(v, grid_size=self.grid_size, segm_mask=m)
        out = tracks[0].cpu().numpy().transpose(1, 0, 2)    # (T,L,2) -> (L,T,2)
        self._cache[key] = out
        return out


class Clip:
    def __init__(self, device="cuda", name="ViT-B/32"):
        import clip, torch
        self.torch, self.clip = torch, clip
        self.model, self.preprocess = clip.load(name, device)
        self.device = device

    def _embed(self, frames):
        from PIL import Image
        t = self.torch
        with t.no_grad():
            ims = t.cat([self.preprocess(Image.fromarray(f)).unsqueeze(0)
                         for f in frames]).to(self.device)
            e = self.model.encode_image(ims)
            return e / e.norm(dim=-1, keepdim=True)

    def score(self, frames, prompt):
        """CLIPScore, frame-to-prompt, averaged. DiTFlow's IQ."""
        t = self.torch
        with t.no_grad():
            text = self.clip.tokenize([prompt], truncate=True).to(self.device)
            tf = self.model.encode_text(text)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            return float((self._embed(frames) @ tf.T).mean().item())

    def subject_identity(self, frames, mask):
        """CLIP-I: subject crop at frame 0 vs every later frame.

        Same reference choice as the DINO variant -- the generated video's own
        first frame, not the reference video's. Reported alongside DINO because
        the two disagree often enough that one alone is a weak claim.
        """
        crops = crop_to_mask(frames, mask)
        if crops is None:
            return float("nan")
        e = self._embed(crops)
        return float((e[0:1] * e[1:]).sum(-1).mean().item())

    def temporal_consistency(self, frames):
        """Mean cosine between consecutive frames' embeddings.

        Reported because the literature reports it, not because it discriminates:
        a frozen video scores ~1.0. That is exactly what control row C1 is for --
        if C1 lands within noise of the best method here, the column is dropped
        from the main table rather than presented as competence.
        """
        e = self._embed(frames)
        return float((e[:-1] * e[1:]).sum(-1).mean().item())


class Dino:
    """Subject consistency: does the subject stay itself across the video?

    Compared against the generated video's own first frame, not the reference.
    The subject is *supposed* to change -- that is the task -- so similarity to
    the reference subject would penalise success. What is well defined is whether
    whatever was generated remains coherent over time.
    """

    def __init__(self, device="cuda"):
        import torch
        self.torch = torch
        self.device = device
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(device).eval()

    def subject_consistency(self, frames, mask):
        import torch.nn.functional as F
        from PIL import Image
        t = self.torch
        crops = crop_to_mask(frames, mask)
        if crops is None:
            return float("nan")
        x = t.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(self.device) / 255
        mean = t.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = t.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with t.no_grad():
            e = self.model((x - mean) / std)
        e = F.normalize(e, dim=-1)
        return float((e[0:1] * e[1:]).sum(-1).mean().item())


class Lpips:
    """Background preservation: did the scene survive the subject swap?

    Unlike the subject metrics, this one *is* measured against the reference.
    In the Subject prompt class the background wording is held byte-identical,
    so the background is supposed to be preserved -- divergence here is
    shape-support leakage, which is the known distortion mode.

    The subject region is zeroed in both images before comparison. That
    introduces an edge, but an identical edge in both, so it contributes to
    neither's advantage. Cropping instead would change the receptive field per
    clip, which is worse.
    """

    def __init__(self, device="cuda"):
        import lpips, torch
        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device)

    def background(self, ref, gen, mask):
        t = self.torch
        keep = (1 - mask)[None, :, :, None]
        n = min(len(ref), len(gen))
        a = t.from_numpy((ref[:n] * keep).astype(np.float32)).permute(0, 3, 1, 2)
        b = t.from_numpy((gen[:n] * keep).astype(np.float32)).permute(0, 3, 1, 2)
        a = (a / 127.5 - 1).to(self.device)
        b = (b / 127.5 - 1).to(self.device)
        with t.no_grad():
            return float(self.model(a, b).mean().item())


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="sweep output root")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotations", help="DAVIS Annotations/480p, for masked metrics")
    ap.add_argument("--cotracker-ref", help="pin CoTracker to a commit or tag")
    ap.add_argument("--grid-size", type=int, default=55, help="DiTFlow uses 55")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="score at most N cells then stop")
    ap.add_argument("--control", choices=["c1"],
                    help="score a control row instead of the generated videos. "
                         "c1 = reference frame 0 held for the whole clip -- the "
                         "degenerate floor. Any metric where c1 lands within "
                         "noise of the best method has no dynamic range and "
                         "leaves the main table")
    ap.add_argument("--no-iq", action="store_true", help="skip CLIP metrics")
    ap.add_argument("--scratch", default="/content/scratch",
                    help="local dir to copy videos through; '' to read "
                         "the run tree directly")
    ap.add_argument("--flush-every", type=int, default=10,
                    help="rewrite the CSV every N cells (Drive writes are slow)")
    ap.add_argument("--no-dino", action="store_true",
                    help="skip subject consistency (needs --annotations)")
    ap.add_argument("--no-lpips", action="store_true",
                    help="skip background LPIPS (needs --annotations)")
    args = ap.parse_args()

    if args.scratch:
        Path(args.scratch).mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    by_key = {(r["clip_id"], r["prompt_id"]): r for r in rows}

    cells = sorted(p.parent for p in Path(args.runs).rglob("done.json"))
    if not cells:
        sys.exit(f"no completed cells under {args.runs}")

    out = Path(args.out)
    # What this invocation will produce. A cell counts as done only if the CSV
    # already has ALL of these for it -- otherwise re-running with --annotations
    # to add the masked columns would skip every cell as "already scored" and
    # silently do nothing, which is exactly what resuming on the cell key alone
    # caused.
    expected = ["mf", "dir_cos"]
    if args.annotations:
        expected += ["mf_masked", "dir_cos_masked"]
        if not args.no_dino:
            expected.append("subject_consistency")
        if not args.no_lpips:
            expected.append("lpips_bg")
    if not args.no_iq:
        expected += ["iq", "temp_cons"]
        if args.annotations:
            expected.append("clip_i_subject")

    # Existing results are held by cell and the whole file is rewritten after
    # each one. Appending would be cheaper, but the header is fixed at first
    # write, so a later run adding masked columns could neither widen it nor
    # update a row without duplicating it.
    existing = {}
    if out.exists():
        for r in csv.DictReader(open(out, newline="", encoding="utf-8")):
            existing[r["cell"]] = r
    scored = {c for c, r in existing.items()
              if all((r.get(m) or "").strip() for m in expected)}
    if existing:
        print(f"{len(existing)} rows in {out}; {len(scored)} already complete "
              f"for this metric set ({', '.join(expected)})")

    todo, seen = [], set()
    for d in cells:
        meta = json.loads((d / "done.json").read_text(encoding="utf-8"))
        key = (meta["clip_id"], meta["prompt_id"])
        if key not in by_key:
            continue                      # cell from a different manifest
        if args.control:
            # One control row per (clip, prompt): C1 does not depend on which
            # method produced the cell, only on the reference beside it.
            if key in seen:
                continue
            seen.add(key)
            meta = dict(meta, config=args.control.upper(),
                        cell="/".join((args.control.upper(),) + key))
        if meta["cell"] in scored:
            continue
        todo.append((d, meta, by_key[key]))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(cells)} cells found, {len(todo)} to score")
    if not todo:
        return

    tracker = Tracker(args.cotracker_ref, args.grid_size, args.device)
    clip = None if args.no_iq else Clip(args.device)
    dino = None if (args.no_dino or not args.annotations) else Dino(args.device)
    lp = None
    if args.annotations and not args.no_lpips:
        try:
            lp = Lpips(args.device)
        except ImportError:
            print('  ! lpips not installed -- skipping background LPIPS')

    results = []
    for i, (d, meta, row) in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {meta['cell']}", end="", flush=True)

        def step(name):
            # Printed as it starts, so a hang names the stage it hung in
            # instead of leaving the whole cell as the suspect.
            print(f" .{name}", end="", flush=True)
        try:
            step("read")
            ref = read_frames(d / "original.mp4", args.scratch)
            gen = read_frames(d / "results.mp4", args.scratch)
        except Exception as e:
            print(f"    skipped: {type(e).__name__}: {e}")
            continue

        if args.control == "c1":
            # The degenerate video: nothing moves. Scored through the identical
            # path as every method, so the comparison is like for like.
            gen = np.repeat(ref[:1], len(ref), axis=0)

        h, w = ref.shape[1:3]
        prior = existing.get(meta["cell"], {})
        need = {m for m in expected if not (prior.get(m) or "").strip()}

        rec = {
            "cell": meta["cell"], "run_tag": meta["cell"].split("/")[0],
            "split": row["split"], "clip_id": meta["clip_id"],
            "prompt_id": meta["prompt_id"], "config": meta["config"],
            "seed": meta["seed"], "prompt": row["prompt"],
            "elapsed_s": meta.get("elapsed_s"),
        }
        # Only compute what is missing. A second pass adding masked columns
        # would otherwise redo the two unmasked tracker calls -- the most
        # expensive step -- for values already sitting in the CSV.
        if need & {"mf", "dir_cos"}:
            step("track")
            rt, gt = tracker(ref), tracker(gen)
            rec["mf"] = motion_fidelity(rt, gt)
            rec["dir_cos"] = direction_cosine(rt, gt)
        # Carry stratification columns through so grouping needs no second join.
        for col in ("cam_band", "subjects", "n_salient_moving", "cam_path_len"):
            if col in row:
                rec[col] = row[col]

        mask = (subject_mask(args.annotations, meta["clip_id"], (w, h))
                if args.annotations else None)
        if mask is not None and mask.any():
            if need & {"mf_masked", "dir_cos_masked"}:
                step("track-masked")
                rtm, gtm = tracker(ref, mask), tracker(gen, mask)
                rec["mf_masked"] = motion_fidelity(rtm, gtm)
                rec["dir_cos_masked"] = direction_cosine(rtm, gtm)
            if dino and "subject_consistency" in need:
                step("dino")
                rec["subject_consistency"] = dino.subject_consistency(gen, mask)
            if clip and "clip_i_subject" in need:
                rec["clip_i_subject"] = clip.subject_identity(gen, mask)
            if lp and "lpips_bg" in need:
                step("lpips")
                rec["lpips_bg"] = lp.background(ref, gen, mask)

        if clip and need & {"iq", "temp_cons"}:
            step("clip")
            rec["iq"] = clip.score(gen, row["prompt"])
            rec["temp_cons"] = clip.temporal_consistency(gen)

        el = time.time() - t0
        done_n = i
        eta = el * (len(todo) - i) / 60
        print(f"  {el:.0f}s  (eta {eta:.0f} min)", flush=True)
        results.append(rec)
        # Merge over any earlier row for this cell, so re-running with more
        # metrics updates in place instead of appending a second row.
        existing[rec["cell"]] = {**existing.get(rec["cell"], {}), **rec}
        scored.add(rec["cell"])
        # Flush periodically, not every cell: `out` is usually on a Drive FUSE
        # mount where each write costs a second or two of latency, and the whole
        # file is rewritten to let the header widen.
        if i % args.flush_every == 0 or i == len(todo):
            write_csv(out, existing)

    # Ctrl-C, a recycled runtime or a decode error on the last cell would
    # otherwise discard up to --flush-every scored cells. Tracking is the whole
    # cost here, so losing nine of them is an hour.
    write_csv(out, existing)
    print(f"\nscored {len(results)} cells -> {out}")
    total = tracker.calls + tracker.hits
    print(f"tracker: {tracker.calls} calls, {tracker.hits} served from cache "
          f"({100 * tracker.hits / max(1, total):.0f}% saved)")


if __name__ == "__main__":
    main()
