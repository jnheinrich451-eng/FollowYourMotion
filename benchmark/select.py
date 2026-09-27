#!/usr/bin/env python
"""Subset a manifest to a named clip list and expand it across prompt classes.

Two jobs that belong together, because doing them by hand is where a benchmark
quietly stops being reproducible.

Subsetting. cut.py emits every clip that survived the stride, which for DAVIS is
80 of 90 -- more than DiTFlow's protocol of 50. Which 50 you keep has to be a
recorded rule rather than a decision made in a spreadsheet, so selection reads
from the dataset's own split files, takes the first N of each in file order, and
writes the rule into every row.

Expansion. A clip needs one row per prompt class. Caption reuses the reference's
own caption; Subject swaps the subject noun and keeps the rest verbatim, so the
Caption-vs-Subject delta isolates the content change instead of confounding it
with a change in prompt style.

    python benchmark/select.py --manifest benchmark/davis.csv --out benchmark/davis50.csv \
        --from E:/DAVIS/ImageSets/2017/val.txt:25 \
        --from E:/DAVIS/ImageSets/2017/train.txt:25 \
        --prompt-classes caption,subject
"""

import argparse
import csv
import sys
from pathlib import Path

TODO = "TODO "   # sweep.py refuses any prompt still carrying this


def parse_source(spec):
    """'path/to/val.txt:25' -> (Path, 25). A bare path means take everything."""
    path, _, count = spec.rpartition(":")
    # A Windows drive letter is not a count: "E:/DAVIS/val.txt" must not split.
    if not path or not count.isdigit():
        return Path(spec), None
    return Path(path), int(count)


def read_list(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest to subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="sources", action="append", default=[],
                    metavar="LIST[:N]",
                    help="clip-id list file, optionally with how many to take; "
                         "repeatable, applied in order")
    ap.add_argument("--captions", metavar="CSV",
                    help="clip_id,caption -- fills the Caption rows, from which "
                         "the Subject rows are then seeded")
    ap.add_argument("--prompt-classes", default="caption,subject",
                    help="comma-separated (default: caption,subject)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    if not rows:
        sys.exit(f"{args.manifest} has no rows")
    by_id = {}
    for r in rows:
        by_id.setdefault(r["clip_id"], r)
    classes = [c.strip() for c in args.prompt_classes.split(",") if c.strip()]

    # Re-running on this tool's own output is the DAVIS workflow: build the file,
    # write the 50 captions by hand, run again to seed the Subject rows from them.
    # DAVIS ships no captions, so without this you would write 100 sentences
    # instead of 50 sentences and 50 single-word edits.
    written = {(r["clip_id"], r["prompt_id"]): r["prompt"]
               for r in rows if r.get("prompt", "").strip()}
    captions = {cid: p for (cid, cls), p in written.items() if cls == "caption"}
    if args.captions:
        # An authored caption file wins: it is the reviewable artifact, and the
        # manifest is regenerated from it rather than the other way round.
        for r in csv.DictReader(open(args.captions, newline="", encoding="utf-8")):
            if r.get("caption", "").strip():
                captions[r["clip_id"]] = r["caption"].strip()
                written.pop((r["clip_id"], "caption"), None)

    chosen, rule = [], []
    if args.sources:
        for spec in args.sources:
            path, take = parse_source(spec)
            if not path.exists():
                sys.exit(f"list file not found: {path}")
            # File order, not alphabetical: it is the dataset's own ordering and
            # does not shift when a clip is added or drops out.
            avail = [c for c in read_list(path) if c in by_id and c not in chosen]
            picked = avail[:take] if take else avail
            if take and len(picked) < take:
                print(f"  ! {path.name}: wanted {take}, only {len(picked)} available "
                      f"(the rest are absent from the manifest)")
            chosen += picked
            rule.append(f"{path.name}[:{len(picked)}]")
    else:
        chosen = list(by_id)
        rule.append("all")

    out_rows = []
    for cid in chosen:
        base = by_id[cid]
        caption = (base.get("caption_short") or "").strip() or captions.get(cid, "")
        for cls in classes:
            row = dict(base)
            row["prompt_id"] = cls
            existing = written.get((cid, cls), "").strip()
            if existing and not existing.startswith(TODO):
                row["prompt"] = existing      # never overwrite a prompt you wrote
            elif not caption:
                row["prompt"] = ""            # DAVIS: nothing to seed from yet
            else:
                row["prompt"] = caption if cls == "caption" else TODO + caption
            row["selection"] = " + ".join(rule)
            out_rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    blank = sum(1 for r in out_rows if not r["prompt"].strip())
    todo = sum(1 for r in out_rows if r["prompt"].startswith(TODO))
    print(f"selected {len(chosen)} clips by {' + '.join(rule)}")
    print(f"wrote {args.out}: {len(out_rows)} rows "
          f"({len(chosen)} clips x {len(classes)} classes)")
    if blank:
        print(f"  {blank} rows need a prompt written from scratch")
    if todo:
        print(f"  {todo} rows seeded behind {TODO.strip()} -- swap the subject noun")


if __name__ == "__main__":
    main()
