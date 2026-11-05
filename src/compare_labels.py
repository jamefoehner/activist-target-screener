#!/usr/bin/env python3
"""
Compare two sets of labels on the same filings: agreement and Cohen's kappa.

  python3 compare_labels.py               # you (pass 1) vs. second labeler (B)
  python3 compare_labels.py --other 2     # you (pass 1) vs. your own pass 2

Writes data/agreement_<other>.csv (per field) and data/disagreements_<other>.csv
(every disagreement, side by side, with the Item 4 text) so you can see which
rules need tightening.

Kappa corrects raw agreement for agreement expected by chance. Rough guide:
  > 0.80 excellent · 0.60–0.80 good · 0.40–0.60 moderate · < 0.40 rules unclear
Kappa is unstable when a label is rare (e.g. checked on 1 of 20 filings), so read
it alongside raw agreement and the number of positives.
"""
import argparse
from pathlib import Path

import pandas as pd

DEMANDS = ["board_seats", "sale_or_review", "breakup_spin", "capital_return", "operations_costs",
           "management_change", "governance", "oppose_deal", "capital_discipline", "engagement_only"]


def kappa(a, b):
    a, b = pd.Series(a).astype(str).reset_index(drop=True), pd.Series(b).astype(str).reset_index(drop=True)
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--other", default="B", help="B (second labeler) or 2 (your pass 2)")
    args = ap.parse_args()

    p1 = pd.read_csv("data/labels_pass1.csv")
    other = pd.read_csv(f"data/labels_pass{args.other}.csv")
    sample = pd.read_csv("data/label_sample.csv")[["item_id", "activist", "subject_name", "item4_text"]]
    if "capital_discipline" not in p1.columns:
        print("Note: pass 1 has no capital_discipline column yet; run review_capital_discipline.py first. "
              "Skipping that category.")
    m = p1.merge(other, on="item_id", suffixes=("_1", "_o"))
    m = m[~m["notes_1"].fillna("").str.upper().str.startswith("EXCLUDE")]
    print(f"{len(m)} filings labeled by both (excluding non-campaigns)\n")

    rows, dis = [], []
    fields = [d for d in DEMANDS if f"{d}_1" in m and f"{d}_o" in m]
    for f in fields + ["proxy_fight", "tone"]:
        a, b = m[f"{f}_1"].fillna(0), m[f"{f}_o"].fillna(0)
        if f in fields:
            a, b = a.astype(int), b.astype(int)
        rows.append({"field": f, "agreement": (a.astype(str) == b.astype(str)).mean(), "kappa": kappa(a, b),
                     "positives_you": int((a == 1).sum()) if f in fields else None,
                     "positives_other": int((b == 1).sum()) if f in fields else None})
        for i in m.index[a.astype(str) != b.astype(str)]:
            dis.append({"item_id": m.at[i, "item_id"], "field": f, "you": m.at[i, f"{f}_1"],
                        "other": m.at[i, f"{f}_o"], "your_notes": m.at[i, "notes_1"],
                        "their_notes": m.at[i, "notes_o"] if "notes_o" in m else ""})

    s1, so = m["seats_1"].fillna(0).astype(int), m["seats_o"].fillna(0).astype(int)
    rows.append({"field": "seats (exact)", "agreement": (s1 == so).mean(), "kappa": kappa(s1, so)})
    for i in m.index[s1 != so]:
        dis.append({"item_id": m.at[i, "item_id"], "field": "seats", "you": s1[i], "other": so[i],
                    "your_notes": m.at[i, "notes_1"], "their_notes": m.at[i, "notes_o"] if "notes_o" in m else ""})
    any_amb = (m["ambiguous_1"].fillna(0).astype(int) | m["ambiguous_o"].fillna(0).astype(int)).astype(bool)

    res = pd.DataFrame(rows)
    res.to_csv(f"data/agreement_{args.other}.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))

    # all demand fields identical on a filing
    exact = (m[[f"{d}_1" for d in fields]].fillna(0).astype(int).values ==
             m[[f"{d}_o" for d in fields]].fillna(0).astype(int).values).all(axis=1)
    print(f"\nFilings where every demand field matched: {exact.mean():.0%} "
          f"(flagged ambiguous by either: {exact[any_amb.values].mean():.0%} of {any_amb.sum()}; "
          f"not flagged: {exact[~any_amb.values].mean():.0%} of {(~any_amb).sum()})")

    d = pd.DataFrame(dis)
    if len(d):
        d = d.merge(sample, on="item_id", how="left")
        d.to_csv(f"data/disagreements_{args.other}.csv", index=False)
        print(f"\n{len(d)} disagreements saved to data/disagreements_{args.other}.csv. By field:")
        print(d["field"].value_counts().to_string())


if __name__ == "__main__":
    main()
