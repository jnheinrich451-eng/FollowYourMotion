#!/usr/bin/env python
"""Apply a subject-substitution map to the Subject rows of a manifest.

The map is a CSV of (clip_id, locomotion, find, replace). Only the leading
subject phrase is rewritten -- everything after it stays byte-identical to the
Caption prompt, so the Caption-vs-Subject delta isolates the subject change
instead of confounding it with a change in wording, length or specificity.

Substitutes are chosen within a locomotion class, because AMF constrains a
per-patch displacement field rather than an abstract action. A car translates
rigidly: every patch on it shares one displacement. A bear's limbs do not. Ask a
bear to satisfy a car's field and it must either slide stiffly or move naturally
and miss the field -- what you would then measure is that conflict, not the
method. Hence vehicle -> vehicle, ridden quadruped -> ridden quadruped, biped ->
biped.

    python benchmark/apply_subjects.py --manifest benchmark/miradata.csv \
        --map benchmark/subject_map.csv
"""

import argparse
import csv
import sys
from pathlib import Path

TODO = "TODO "


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--map", required=True, help="clip_id,locomotion,find,replace")
    ap.add_argument("--out", help="defaults to editing --manifest in place")
    ap.add_argument("--prompt-id", default="subject", help="which class to rewrite")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    subs = {r["clip_id"]: r for r in
            csv.DictReader(open(args.map, newline="", encoding="utf-8"))}
    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))

    done, skipped, nomatch = 0, [], []
    for r in rows:
        if r["prompt_id"] != args.prompt_id:
            continue
        sub = subs.get(r["clip_id"])
        if sub is None:
            skipped.append(f"{r['clip_id']}: not in the map")
            continue
        text = r["prompt"]
        seeded = text.startswith(TODO)
        if not seeded and text.strip():
            skipped.append(f"{r['clip_id']}: already written, left alone")
            continue
        base = text[len(TODO):] if seeded else (r.get("caption_short") or "")
        if not base.startswith(sub["find"]):
            # The map keys off the caption's own opening words. If they have
            # changed, rewriting anyway would silently produce a prompt that is
            # not the caption-minus-subject, so refuse instead.
            nomatch.append(f"{r['clip_id']}: expected {sub['find']!r}, "
                           f"caption starts {base[:40]!r}")
            continue
        r["prompt"] = sub["replace"] + base[len(sub["find"]):]
        done += 1

    for label, items in (("skipped", skipped), ("no match", nomatch)):
        for i in items:
            print(f"  ! {label}: {i}")
    print(f"\nrewrote {done} {args.prompt_id} prompts")

    if args.dry_run:
        print("(dry run, nothing written)")
        return
    out = args.out or args.manifest
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
