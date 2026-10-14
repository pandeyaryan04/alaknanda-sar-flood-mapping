r"""
02_fetch_sen1floods11.py
========================
Download the Sen1Floods11 training data from Google Cloud Storage.
Resumable, verified, and splittable across two machines.

WHAT THIS IS
------------
Sen1Floods11 (Bonafilia et al. 2020) is the labelled dataset the flood models
learn from. Public Google Cloud bucket, no account and no login needed.

  HAND-LABELLED   446 chips of 512 x 512, each mask drawn by a person.
                  This is the ground truth. Small. Start here.

  WEAKLY-LABELLED 4,385 chips whose labels were generated automatically, so
                  they are noisier. Use: pre-train on a lot of rough data,
                  then fine-tune on the small set of good data. Large and
                  OPTIONAL - do not download it until you have decided you
                  need it.

Parallel folders hold different versions of each chip:

  S1Hand           Sentinel-1 radar image           <- MODEL INPUT
  LabelHand        hand-drawn flood mask            <- MODEL TARGET
  S2Hand           matching Sentinel-2 optical image
  JRCWaterHand     JRC permanent water
  S1OtsuLabelHand  what plain Otsu produced, for the baseline comparison

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Always survey first. Downloads nothing, tells you exactly how big each set is:

    python 02_fetch_sen1floods11.py --list

Then fetch the minimum needed to train:

    python 02_fetch_sen1floods11.py --get core

Sets:  core  S1Hand + LabelHand + splits          (the training minimum)
       hand  all five hand-labelled folders + splits
       weak  weakly-labelled radar chips + labels (large)
       all   everything

--------------------------------------------------------------------------
SPLITTING ACROSS TWO LAPTOPS
--------------------------------------------------------------------------
Only worth doing for the weak set. Run on laptop A:

    python 02_fetch_sen1floods11.py --get weak --shard 1/2

and on laptop B, at the same time:

    python 02_fetch_sen1floods11.py --get weak --shard 2/2

The split is by a hash of each filename, so the two machines never overlap
and never leave a gap. No coordination needed. Copy laptop B's
data/sen1floods11 folder over laptop A's afterwards (merge, do not replace),
then confirm the merge worked:

    python 02_fetch_sen1floods11.py --get weak --verify

--verify checks every expected file against the bucket's own byte count and
prints exactly what is missing. Run the plain --get again to fill any gaps.

--------------------------------------------------------------------------
INTERRUPTIONS
--------------------------------------------------------------------------
Safe to stop at any point with Ctrl+C, and safe after a power cut. Re-run the
same command: files already on disk at the right size are skipped, and a
half-written file is discarded rather than kept.

NOTE ON LICENCE
---------------
The Sen1Floods11 repository and paper do not state a licence. Coursework use
is uncontroversial; before anything derived from it is published, ask the
authors in writing. Recorded as a constraint in the methodology.

================================================================================
HOW TO RUN THIS  (copy the lines, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"

Survey first. Downloads nothing, prints the size of every folder:
    python 02_fetch_sen1floods11.py --list

The training minimum, about 1.25 GB:
    python 02_fetch_sen1floods11.py --get core

Everything hand-labelled, about 4 GB. This is the useful one:
    python 02_fetch_sen1floods11.py --get hand

The radar pre-training set, about 6.8 GB:
    python 02_fetch_sen1floods11.py --get weak

Check for gaps without downloading:
    python 02_fetch_sen1floods11.py --get weak --verify

ALREADY RUN, 25 August 2026: --get all completed. 19,654 files, 32.4 GB, zero
failures, every file checked against the bucket's own byte count. Nothing to do
here unless a file gets deleted, in which case re-run the same command and only
the gaps are fetched.

Safe to stop with Ctrl+C at any point. Safe after a power cut. Re-running skips
whatever is already on disk at the right size.
================================================================================
"""

__stage__ = "Sen1Floods11 training data retrieval"
__reads__ = ["Sen1Floods11 public bucket"]
__writes__ = ["sen1floods11 chips", "manifest.csv"]

import argparse
import csv
import hashlib
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("\nERROR: requests not installed.\n    conda activate fyp\n    pip install requests\n")

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
DEST = os.path.join("data", "sen1floods11")

# Verified against the live bucket, 20 August 2026.
FOLDERS = {
    "s1_hand":      "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "label_hand":   "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "s2_hand":      "v1.1/data/flood_events/HandLabeled/S2Hand/",
    "jrc_hand":     "v1.1/data/flood_events/HandLabeled/JRCWaterHand/",
    "otsu_hand":    "v1.1/data/flood_events/HandLabeled/S1OtsuLabelHand/",
    "splits":       "v1.1/splits/flood_handlabeled/",
    "s1_weak":      "v1.1/data/flood_events/WeaklyLabeled/S1Weak/",
    "otsu_weak":    "v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/",
    "s2index_weak": "v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/",
    "s2_weak":      "v1.1/data/flood_events/WeaklyLabeled/S2Weak/",
}
SETS = {
    "core": ["s1_hand", "label_hand", "splits"],
    "hand": ["s1_hand", "label_hand", "s2_hand", "jrc_hand", "otsu_hand", "splits"],
    "weak": ["s1_weak", "otsu_weak"],
    "all":  list(FOLDERS.keys()),
}
DESC = {
    "s1_hand":      "Sentinel-1 radar chips, hand-labelled   (MODEL INPUT)",
    "label_hand":   "hand-drawn flood masks                  (MODEL TARGET)",
    "s2_hand":      "Sentinel-2 optical chips, same locations",
    "jrc_hand":     "JRC permanent water mask",
    "otsu_hand":    "Otsu output, for baseline comparison",
    "splits":       "train / validation / test / Bolivia chip lists",
    "s1_weak":      "Sentinel-1 radar chips, weakly-labelled",
    "otsu_weak":    "Otsu labels for the weakly-labelled set",
    "s2index_weak": "Sentinel-2 index labels, weakly-labelled",
    "s2_weak":      "Sentinel-2 optical chips, weakly-labelled",
}
RETRIES = 4


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:,.0f} B" if u == "B" else f"{n:,.1f} {u}"
        n /= 1024.0


def parse_shard(text):
    """'1/2' -> (0, 2). One-based for humans, zero-based inside."""
    if not text:
        return (0, 1)
    try:
        i, m = text.split("/")
        i, m = int(i), int(m)
        if m < 1 or not (1 <= i <= m):
            raise ValueError
        return (i - 1, m)
    except ValueError:
        sys.exit(f"\nERROR: --shard must look like 1/2 or 2/2. Got '{text}'.\n")


def in_shard(name, idx, m):
    if m == 1:
        return True
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % m == idx


def list_folder(prefix, session):
    items, token = [], None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        for attempt in range(RETRIES):
            try:
                r = session.get(API, params=params, timeout=60)
                r.raise_for_status()
                break
            except Exception as exc:
                if attempt == RETRIES - 1:
                    print(f"    ! listing failed for {prefix}: {exc}")
                    return items
                time.sleep(2 ** attempt)
        body = r.json()
        for it in body.get("items", []):
            if not it["name"].endswith("/"):
                items.append({"name": it["name"], "size": int(it.get("size", 0))})
        token = body.get("nextPageToken")
        if not token:
            return items


def survey(keys, session, shard):
    idx, m = shard
    tag = f"  (shard {idx+1}/{m})" if m > 1 else ""
    print(f"\nSurveying the bucket{tag}. Nothing is downloaded.\n")
    print(f"  {'set':<14} {'files':>7} {'size':>11}   description")
    print("  " + "-" * 74)
    listing, tf, tb = {}, 0, 0
    for k in keys:
        allitems = list_folder(FOLDERS[k], session)
        items = [i for i in allitems if in_shard(i["name"], idx, m)]
        listing[k] = items
        sz = sum(i["size"] for i in items)
        tf += len(items)
        tb += sz
        extra = f"  (of {len(allitems)})" if m > 1 else ""
        print(f"  {k:<14} {len(items):>7} {human(sz):>11}   {DESC[k]}{extra}")
    print("  " + "-" * 74)
    print(f"  {'TOTAL':<14} {tf:>7} {human(tb):>11}")
    return listing, tf, tb


def fetch_one(session, rel, size, out):
    tmp = out + ".part"
    for attempt in range(RETRIES):
        try:
            with session.get(MEDIA + rel, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            got = os.path.getsize(tmp)
            if got != size:
                raise IOError(f"size mismatch: expected {size}, got {got}")
            os.replace(tmp, out)
            return True, ""
        except Exception as exc:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt == RETRIES - 1:
                return False, str(exc)
            time.sleep(2 ** attempt)
    return False, "exhausted retries"


def download(items, session):
    done = skipped = failed = 0
    got_bytes = 0
    t0 = time.time()
    for n, it in enumerate(items, 1):
        rel = it["name"]
        out = os.path.join(DEST, *rel.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == it["size"]:
            skipped += 1
        else:
            ok, err = fetch_one(session, rel, it["size"], out)
            if ok:
                done += 1
                got_bytes += it["size"]
            else:
                failed += 1
                print(f"\n    ! {rel}: {err}")
        if n % 10 == 0 or n == len(items):
            el = max(time.time() - t0, 0.1)
            rate = got_bytes / el
            left = len(items) - n
            eta = (left * (got_bytes / max(done, 1)) / rate / 60) if done and rate else 0
            sys.stdout.write(
                f"\r    {n}/{len(items)} ({100.0*n/len(items):5.1f}%)  "
                f"new {done}  skip {skipped}  fail {failed}  "
                f"{human(rate)}/s  eta {eta:4.0f}m   "
            )
            sys.stdout.flush()
    print()
    return done, skipped, failed


def write_manifest(listing, keys, shard):
    idx, m = shard
    suffix = f"_shard{idx+1}of{m}" if m > 1 else ""
    path = os.path.join(DEST, f"manifest{suffix}.csv")
    missing = []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "path", "expected_bytes", "on_disk_bytes", "ok"])
        for k in keys:
            for it in listing[k]:
                out = os.path.join(DEST, *it["name"].split("/"))
                sz = os.path.getsize(out) if os.path.exists(out) else -1
                ok = sz == it["size"]
                if not ok:
                    missing.append(it["name"])
                w.writerow([k, it["name"], it["size"], sz, "yes" if ok else "NO"])
    return path, missing


def main():
    ap = argparse.ArgumentParser(
        description="Download Sen1Floods11. Resumable, verified, splittable.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="survey the bucket, download nothing")
    g.add_argument("--get", choices=sorted(SETS.keys()), help="which set to download")
    ap.add_argument("--shard", metavar="N/M",
                    help="download only part N of M, e.g. 1/2 on one laptop and 2/2 on the other")
    ap.add_argument("--verify", action="store_true",
                    help="with --get: check what is on disk against the bucket, download nothing")
    args = ap.parse_args()

    shard = parse_shard(args.shard)
    session = requests.Session()
    keys = SETS["all"] if args.list else SETS[args.get]

    listing, nf, nb = survey(keys, session, shard)

    if args.list:
        core = sum(sum(i["size"] for i in listing.get(k, [])) for k in SETS["core"])
        weak = sum(sum(i["size"] for i in listing.get(k, [])) for k in SETS["weak"])
        print("\n  Nothing downloaded. Suggested next step:\n")
        print(f"      python 02_fetch_sen1floods11.py --get core"
              f"      ({human(core)} - enough to train)")
        print(f"\n  The optional weak-label set is {human(weak)}. Only fetch it if you")
        print("  have decided you need pre-training. If you do, and you have a second")
        print("  machine on a separate connection:")
        print("      laptop A:  python 02_fetch_sen1floods11.py --get weak --shard 1/2")
        print("      laptop B:  python 02_fetch_sen1floods11.py --get weak --shard 2/2")
        print(f"\n  Keep at least {human(2*(core+weak))} free before starting anything large.\n")
        return

    os.makedirs(DEST, exist_ok=True)

    if args.verify:
        print(f"\nVerifying '{args.get}' in {os.path.abspath(DEST)} (no download)\n")
        path, missing = write_manifest(listing, keys, shard)
        print(f"  manifest : {path}")
        if missing:
            print(f"\n  {len(missing)} file(s) missing or wrong size. First few:")
            for mname in missing[:10]:
                print(f"    {mname}")
            if len(missing) > 10:
                print(f"    ... and {len(missing)-10} more")
            print(f"\n  Fix by re-running without --verify:")
            print(f"      python 02_fetch_sen1floods11.py --get {args.get}"
                  + (f" --shard {shard[0]+1}/{shard[1]}" if shard[1] > 1 else ""))
        else:
            print("\n  Complete. Every expected file present at the right size.")
        print()
        return

    idx, m = shard
    tag = f", shard {idx+1} of {m}" if m > 1 else ""
    print(f"\nDownloading '{args.get}'{tag} into {os.path.abspath(DEST)}")
    print(f"  {nf} files, {human(nb)}. Safe to interrupt and re-run.\n")

    grand = {"new": 0, "skip": 0, "fail": 0}
    try:
        for k in keys:
            print(f"  [{k}] {DESC[k]}")
            dn, sk, fl = download(listing[k], session)
            grand["new"] += dn
            grand["skip"] += sk
            grand["fail"] += fl
    except KeyboardInterrupt:
        print("\n\n  Stopped. Nothing corrupted - re-run the same command to continue.\n")
        return

    path, missing = write_manifest(listing, keys, shard)
    print("\n" + "=" * 68)
    print("RESULT")
    print("=" * 68)
    print(f"  downloaded this run : {grand['new']}")
    print(f"  already present     : {grand['skip']}")
    print(f"  failed              : {grand['fail']}")
    print(f"  manifest            : {path}")
    if missing:
        print(f"\n  {len(missing)} file(s) still missing or wrong size.")
        print("  Re-run the same command; only those will be fetched.")
    else:
        print("\n  Every file verified against the byte count reported by the bucket.")
    if m > 1:
        print(f"\n  This was shard {idx+1} of {m}. When the other machine finishes,")
        print("  merge its data/sen1floods11 folder into this one, then run:")
        print(f"      python 02_fetch_sen1floods11.py --get {args.get} --verify")
        print("  to confirm the merge left no gaps.")
    print()


if __name__ == "__main__":
    main()
