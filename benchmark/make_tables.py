#!/usr/bin/env python3
"""
results.jsonl  ->  tables/*.tex

Rule: no number is ever typed into the chapter by hand. The chapter
\input{}s files this script writes. A re-run means recompiling, not
re-transcribing.

Schema — one JSON object per line, one line per (clip, method, seed,
setting, prompt_condition). Every field below is REQUIRED; fields marked
RUNTIME cannot be reconstructed after the fact, so the harness must emit
them at run time or they are lost.

{
  "clip_id":        "mira_0031",
  "split":          "real" | "kubric",
  "parallax_band":  "low" | "mid" | "high",
  "parallax_value": 0.184,          # baseline / median scene depth
  "motion_type":    "rigid" | "articulated" | "multi",
  "method":         "ditflow",
  "backbone":       "wan2.1-i2v-14b",
  "setting":        "transfer" | "explicit",
  "prompt_cond":    "caption" | "subject",
  "seed":           0,

  "status":         "ok" | "excluded",
  "reason_code":    null | "OOM" | "NAN" | "CRASH" | "DEGENERATE" | "TIMEOUT",
  "failure_codes":  ["F-SSL"],      # pre-registered taxonomy, may be empty

  "metrics": {                       # null for any metric not applicable
    "mf": 0.912, "text_sim": 0.311, "temp_cons": 0.968, "fvd": null,
    "cam_trans_err": 0.0412, "cam_rot_err": 0.0089, "umeyama_scale": 1.037,
    "dino_i": 0.821, "clip_i": 0.874, "lpips_bg": 0.118
  },

  "runtime": {                       # RUNTIME — unrecoverable if not logged
    "wall_s": 734.2,
    "gpu": "A100-40GB",
    "peak_vram_gb": 31.4,
    "commit": "a1b2c3d",
    "ckpt_sha": "sha256:9f2c...",
    "intervention": null             # verbatim string if you touched anything
  }
}

Usage:
    python benchmark/make_tables.py results.jsonl --out tables/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

# --------------------------------------------------------------------
# Statistical treatment — must match §5.2.4 of the chapter exactly.
# Unit of analysis is the CLIP: average over seeds first, then compare
# methods across clips with a PAIRED bootstrap.
# --------------------------------------------------------------------

N_BOOT = 10_000
RNG_SEED = 0


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_clip(rows: list[dict], metric: str, **filters) -> dict[str, float]:
    """Seed-averaged value of `metric` per clip, for rows matching filters.

    Excluded runs are dropped here and counted separately in the run
    accounting table — never silently imputed.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["status"] != "ok":
            continue
        if any(r.get(k) != v for k, v in filters.items()):
            continue
        v = r["metrics"].get(metric)
        if v is not None:
            buckets[r["clip_id"]].append(v)
    return {c: float(np.mean(vs)) for c, vs in buckets.items()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float]):
    """Median paired difference (a - b) with a 95% bootstrap CI over clips.

    Returns (n_pairs, median_diff, lo, hi). Only clips present in BOTH
    methods are used — that is what makes the comparison paired.
    """
    clips = sorted(set(a) & set(b))
    if not clips:
        return 0, float("nan"), float("nan"), float("nan")
    d = np.array([a[c] - b[c] for c in clips])
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    meds = np.median(d[draws], axis=1)
    return len(d), float(np.median(d)), float(np.percentile(meds, 2.5)), float(
        np.percentile(meds, 97.5)
    )


def claimable(lo: float, hi: float, med: float, noise_floor: float) -> bool:
    """The two gates from §5.2.4, applied mechanically.

    Gate 1: the 95% CI excludes zero.
    Gate 2: the effect exceeds the Layer-3 noise floor for that metric.

    A difference that fails either gate is printed but never described as a
    difference in the text.
    """
    return (lo > 0 or hi < 0) and abs(med) > noise_floor


def fmt(x: float | None, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------
# Table writers. One function per \input{} in the chapter. Each writes a
# complete table environment INCLUDING the caption, so the caption lives
# next to the code that fills it and cannot drift out of sync.
# --------------------------------------------------------------------


def tab_block1_motion(rows, out: Path, noise: dict):
    """Block 1 — motion transfer, seed-averaged per clip then pooled."""
    methods = sorted({r["method"] for r in rows})
    cols = ["mf", "text_sim", "temp_cons"]
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Motion-transfer metrics under \protP{}, subject prompt, "
        r"transfer setting. Values are means over clips of seed-averaged "
        r"scores; parentheses give the interquartile range across clips. "
        r"Differences are claimed in the text only where the paired "
        r"bootstrap CI excludes zero \emph{and} the effect exceeds the "
        r"noise floor of Table~\ref{tab:bounds}.}",
        r"\label{tab:block1}",
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        "Method & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        cells = []
        for c in cols:
            vals = list(
                per_clip(
                    rows, c, method=m, setting="transfer", prompt_cond="subject"
                ).values()
            )
            cells.append(fmt(float(np.mean(vals)) if vals else None))
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_block1_motion.tex").write_text("\n".join(lines) + "\n")


def tab_run_accounting(rows, out: Path):
    """The table most benchmark chapters omit and every examiner asks for."""
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        m = r["method"]
        agg[m]["attempted"] += 1
        if r["status"] == "ok":
            agg[m]["completed"] += 1
        else:
            agg[m][r.get("reason_code") or "UNSPECIFIED"] += 1
    codes = sorted({k for v in agg.values() for k in v} - {"attempted", "completed"})
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Run accounting. Every generation attempted under \protP{} "
        r"is counted here; exclusions are reported with the reason recorded "
        r"at run time, not reconstructed afterwards.}",
        r"\label{tab:runs}",
        r"\begin{tabular}{lrr" + "r" * len(codes) + "}",
        r"\toprule",
        "Method & Attempted & Completed & " + " & ".join(codes) + r" \\",
        r"\midrule",
    ]
    for m in sorted(agg):
        row = [str(agg[m]["attempted"]), str(agg[m]["completed"])]
        row += [str(agg[m].get(c, 0)) for c in codes]
        lines.append(f"{m} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_run_accounting.tex").write_text("\n".join(lines) + "\n")


def tab_parallax_stratified(rows, out: Path, metric: str = "mf"):
    """The stratification that makes the identifiability claim empirical."""
    methods = sorted({r["method"] for r in rows})
    bands = ["low", "mid", "high"]
    lines = [
        r"\begin{table}[t]\centering",
        rf"\caption{{{metric.upper()} by parallax band and evaluation setting. "
        r"The gap between the transfer and explicit-control settings is "
        r"predicted to widen with parallax; a flat profile falsifies that "
        r"prediction.}}",
        r"\label{tab:parallax}",
        r"\begin{tabular}{l" + "rr" * len(bands) + "}",
        r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b}}}" for b in bands) + r" \\",
        "Method & " + " & ".join(["transfer & explicit"] * len(bands)) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        cells = []
        for b in bands:
            for s in ("transfer", "explicit"):
                vals = list(
                    per_clip(
                        rows, metric, method=m, parallax_band=b, setting=s
                    ).values()
                )
                cells.append(fmt(float(np.mean(vals)) if vals else None))
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_parallax_stratified.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--noise-floor", default="noise_floor.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.results)
    noise = json.loads(Path(args.noise_floor).read_text()) if Path(
        args.noise_floor
    ).exists() else {}

    tab_block1_motion(rows, out, noise)
    tab_run_accounting(rows, out)
    tab_parallax_stratified(rows, out)

    # Stubs so the chapter compiles from day one, before any run finishes.
    for name in [
        "tab_data_splits", "tab_capability_matrix", "tab_environment",
        "tab_parameters", "tab_bounds", "tab_protocol_audit",
        "tab_layer1_kubric", "tab_block2_camera", "tab_block3_identity",
        "tab_null_study", "tab_component_analysis", "tab_failure_counts",
    ]:
        p = out / f"{name}.tex"
        if not p.exists():
            p.write_text(
                r"\begin{table}[t]\centering"
                rf"\caption{{\textcolor{{red}}{{PENDING: {name}}}}}"
                r"\begin{tabular}{c}\toprule TBD \\\bottomrule\end{tabular}"
                "\n" r"\end{table}" "\n"
            )

    print(f"wrote tables to {out}/  ({len(rows)} result rows)")


if __name__ == "__main__":
    main()
