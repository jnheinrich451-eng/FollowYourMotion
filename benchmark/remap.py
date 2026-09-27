#!/usr/bin/env python
"""Rewrite the *_path columns of a manifest for a different machine.

cut.py and realcam.py bake absolute paths into the manifest on purpose -- they
are what make a row reproducible, and on Windows a data drive and the repo drive
have no relative path between them anyway. Moving the same benchmark to Colab
therefore needs the roots translated, which is this, and only this: nothing else
in the row changes, so the sha256 and provenance still describe the same bytes.

    python benchmark/remap.py --manifest benchmark/davis50.csv \
        --out /content/davis50.csv --from "E:\\bench\\packed" --to /content/data/packed
"""

import argparse
import csv
import sys
from pathlib import PurePosixPath, PureWindowsPath


def translate(value, src, dst):
    """Swap the src prefix for dst, normalising separators to POSIX."""
    norm = value.replace("\\", "/")
    src_n = src.replace("\\", "/").rstrip("/")
    if not norm.lower().startswith(src_n.lower()):
        return value, False
    tail = norm[len(src_n):].lstrip("/")
    return str(PurePosixPath(dst.rstrip("/")) / tail) if tail else dst, True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="src", required=True, metavar="PREFIX",
                    help="path root as written in the manifest")
    ap.add_argument("--to", dest="dst", required=True, metavar="PREFIX",
                    help="path root on this machine")
    ap.add_argument("--verify", action="store_true",
                    help="check every rewritten path exists; use when --to is a "
                         "root on THIS machine, not when preparing a manifest "
                         "for another one")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    if not rows:
        sys.exit(f"{args.manifest} has no rows")
    cols = [c for c in rows[0] if c.endswith("_path")]
    if not cols:
        sys.exit(f"{args.manifest} has no *_path columns to remap")

    hits = {c: 0 for c in cols}
    for r in rows:
        for c in cols:
            if r.get(c):
                r[c], ok = translate(r[c], args.src, args.dst)
                hits[c] += ok

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{args.src}  ->  {args.dst}")
    for c in cols:
        state = "ok" if hits[c] == len(rows) else f"ONLY {hits[c]}/{len(rows)}"
        print(f"  {c:18} {hits[c]}/{len(rows)} rewritten  [{state}]")
    if any(h == 0 for h in hits.values()):
        print("\n  ! a column matched nothing -- check --from against the manifest")

    if args.verify:
        # A manifest pointing at nothing produces one identical failure per
        # cell, with a reason code that describes the symptom rather than this
        # cause. Cheaper to stat every path once than to read 300 logs.
        import os
        missing = [(r["clip_id"], c, r[c]) for r in rows for c in cols
                   if r.get(c) and not os.path.exists(r[c])]
        if missing:
            print(f"\n  ! {len(missing)} of {len(rows) * len(cols)} paths do not exist:")
            for cid, c, path in missing[:5]:
                print(f"      {cid} [{c}] {path}")
            sys.exit("  --to is probably wrong for this machine")
        print(f"  verified: all {len(rows) * len(cols)} paths exist")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
