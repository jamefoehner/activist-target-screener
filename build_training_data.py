#!/usr/bin/env python3
"""
RL project, phase 1: build training data for a small model.

  1. Collect more 13D filings (originals AND 13D/A amendments) from your EDGAR data,
     excluding the 100 hand-labeled filings and every non-campaign exclusion.
  2. Extract Item 4 with the same parser as extract_item4.py.
  3. Sample ~600, balanced across activists and filing types.
  4. Label them with Claude ("silver" labels) using the labeling guide, the same
     prompt as your best evaluation setup.
  5. Write data/rl/train.jsonl and dev.jsonl (silver) and test_gold.jsonl (your human
     labels, never used for training).

Usage
  python3 build_training_data.py --user-agent "Name email@example.com" --dry-run
  python3 build_training_data.py --user-agent "Name email@example.com" --n 600
  # cheaper test run:  --n 50 --labeler claude-haiku-5-5

Needs extract_item4.py and run_eval.py in the same folder, ANTHROPIC_API_KEY set.
Filings and Claude answers are cached, so reruns don't repeat downloads or API calls.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from extract_item4 import Edgar, extract_item4
from run_eval import DEMANDS, Anthropic, Fake, build_prompt, cached_call, parse

OUT = Path("data/rl")


def gold_record(r, max_chars):
    label = {d: int(r[d]) for d in DEMANDS}
    label.update(seats=int(r["seats"]), proxy_fight=r["proxy_fight"], tone=r["tone"])
    return {"id": f"gold-{int(r['item_id'])}", "text": r["item4_text"][:max_chars], "label": label,
            "activist": r["activist"], "ambiguous": int(r.get("ambiguous", 0) or 0), "source": "human"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", required=True)
    ap.add_argument("--filings", default=None,
                    help="CSV of all 13D/13D-A filings from edgar_13d_activists.py (auto-detected in data/ if omitted)")
    ap.add_argument("--n", type=int, default=600, help="filings to silver-label")
    ap.add_argument("--labeler", default="claude-sonnet-5-5")
    ap.add_argument("--provider", choices=["anthropic", "fake"], default="anthropic")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="truncate Item 4 to this many characters (~3,000 tokens) to fit small-model context")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--guide", default="LABELING_GUIDE.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude-jsonl", nargs="*", default=[],
                    help="jsonl files whose ids must not be sampled again (e.g. data/rl/train.jsonl data/rl/dev.jsonl)")
    ap.add_argument("--only-out", default=None,
                    help="write all silver-labeled filings to data/rl/<name>.jsonl and stop; leaves train/dev/test_gold alone")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- 1. the pool of filings ----
    path = Path(args.filings) if args.filings else next(
        (p for p in [Path("data/activist_13d_filings.csv"), Path("data/activist_filings.csv"),
                     Path("data/all_filings.csv")] if p.exists()), None)
    if path is None:
        cands = [p.name for p in Path("data").glob("*.csv")]
        raise SystemExit("Couldn't find the all-filings CSV from edgar_13d_activists.py. Pass it with --filings. "
                         f"CSVs in data/: {', '.join(sorted(cands))}")
    allf = pd.read_csv(path, dtype={"accession": str})
    need = {"accession", "url", "activist"}
    if not need <= set(allf.columns):
        raise SystemExit(f"{path} needs columns {need}; it has {list(allf.columns)}")
    form_col = next((c for c in ["form", "form_type", "type"] if c in allf.columns), None)
    allf["kind"] = allf[form_col].astype(str).str.contains("/A").map({True: "amendment", False: "original"}) \
        if form_col else "unknown"

    gold_sample = pd.read_csv("data/label_sample.csv", dtype={"accession": str})
    excluded = set(pd.read_csv("data/exclusions.csv", dtype=str)["accession"]) if Path("data/exclusions.csv").exists() else set()
    for fp in args.exclude_jsonl:
        excluded |= {json.loads(l)["id"] for l in open(fp)}
    pool = allf[~allf["accession"].isin(set(gold_sample["accession"]) | excluded)].drop_duplicates("accession")
    print(f"Pool: {len(pool)} filings ({pool['kind'].value_counts().to_dict()}) after removing "
          f"{len(gold_sample)} gold and {len(excluded)} excluded filings")

    # balanced sample: equal share per activist x kind, filled up proportionally if a cell runs short
    cells = pool.groupby(["activist", "kind"])
    per = max(1, int(args.n * 1.6) // max(1, cells.ngroups))      # oversample: some won't parse
    picks = pd.concat([g.sample(min(len(g), per), random_state=args.seed) for _, g in cells])
    print(f"Sampled {len(picks)} candidates to download (aiming for {args.n} usable)")
    if args.dry_run:
        print(picks.groupby(["activist", "kind"]).size().unstack(fill_value=0))
        print(f"\nDry run. Labeling {args.n} filings with {args.labeler} at ~4,500 input tokens each is "
              f"~{args.n * 4500 / 1e6:.1f}M input tokens; check the pricing page for cost.")
        return

    # ---- 2. extract Item 4 ----
    edgar = Edgar(args.user_agent, Path("cache/filings"))
    texts = []
    for i, r in enumerate(picks.itertuples(index=False), 1):
        try:
            t, how = extract_item4(edgar.get(r.url, r.accession))
        except Exception:
            t, how = None, "error"
        if t and 200 <= len(t) <= 60000:
            texts.append({"accession": r.accession, "activist": r.activist, "kind": r.kind, "text": t})
        if i % 100 == 0:
            print(f"  downloaded {i}/{len(picks)}, usable so far {len(texts)}")
    rng.shuffle(texts)
    texts = texts[:args.n]
    print(f"Usable Item 4 texts: {len(texts)}")

    # ---- 3. silver labels ----
    guide = Path(args.guide).read_text()
    system = build_prompt("guide", guide, pd.DataFrame())
    backend = Anthropic() if args.provider == "anthropic" else Fake()
    silver, failed = [], 0
    for i, t in enumerate(texts, 1):
        res = cached_call(backend, args.provider, args.labeler, "silver", t["accession"], system,
                          f"<item4>\n{t['text']}\n</item4>")
        lab = parse(res.get("text", ""))
        if lab is None:
            failed += 1
            continue
        silver.append({"id": t["accession"], "text": t["text"][:args.max_chars], "label": lab,
                       "activist": t["activist"], "kind": t["kind"], "source": args.labeler,
                       "truncated": len(t["text"]) > args.max_chars})
        if i % 50 == 0:
            print(f"  labeled {i}/{len(texts)}")
    print(f"Silver-labeled {len(silver)} filings ({failed} unparseable replies dropped)")

    if args.only_out:
        with open(OUT / f"{args.only_out}.jsonl", "w") as f:
            for row in silver:
                f.write(json.dumps(row) + "\n")
        lab = pd.DataFrame([r["label"] for r in silver])
        print(f"Saved {len(silver)} filings to {OUT}/{args.only_out}.jsonl")
        print((lab[DEMANDS].mean() * 100).round(1).astype(str).add("%").to_string())
        return

    # ---- 4. splits ----
    rng.shuffle(silver)
    n_dev = max(1, int(len(silver) * args.dev_frac))
    dev, train = silver[:n_dev], silver[n_dev:]
    gold = pd.read_csv("data/labels_pass1.csv").merge(gold_sample[["item_id", "activist", "item4_text"]], on="item_id")
    gold = gold[~gold["notes"].fillna("").str.upper().str.startswith("EXCLUDE")]
    for d in DEMANDS:
        gold[d] = gold[d].fillna(0).astype(int) if d in gold else 0
    gold["seats"] = gold["seats"].fillna(0).astype(int)
    test = [gold_record(r, args.max_chars) for _, r in gold.iterrows()]

    for name, rows in [("train", train), ("dev", dev), ("test_gold", test)]:
        with open(OUT / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    (OUT / "system_prompt.txt").write_text(system)

    lab = pd.DataFrame([r["label"] for r in train])
    print(f"\nSaved to {OUT}/: train {len(train)}, dev {len(dev)}, test_gold {len(test)} (human labels, never train on these)")
    print("Share of training filings positive for each field (watch for rare categories):")
    print((lab[DEMANDS].mean() * 100).round(1).astype(str).add("%").to_string())
    print(f"Truncated to {args.max_chars:,} characters: {sum(r['truncated'] for r in train)} of {len(train)} training filings")


if __name__ == "__main__":
    main()
