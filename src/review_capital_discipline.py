#!/usr/bin/env python3
"""
After adding the capital_discipline category: add the column to your pass-1
labels (default 0) and list the items worth reviewing for it.

  python3 review_capital_discipline.py

Writes a backup (data/labels_pass1_backup.csv), adds the capital_discipline
column to data/labels_pass1.csv if missing, and saves
data/capital_discipline_review.csv: items whose notes or Item 4 text mention
capital allocation, acquisitions, capex, leverage or returns on capital.
Open each in the labeling app (pass 1, "Go to item #") and decide.
"""
import re
import shutil
from pathlib import Path

import pandas as pd

LABELS, SAMPLE = Path("data/labels_pass1.csv"), Path("data/label_sample.csv")
NOTE_RE = re.compile(r"capital discipline|capital allocation|capex|leverage|acquisition|returns? on|roic|"
                     r"financial discipline|capital structure|spending", re.I)
TEXT_RE = re.compile(r"capital allocation|capital expenditure|capital spending|returns? on (?:invested )?capital|"
                     r"reduc\w* (?:its )?(?:leverage|debt)|acquisition strateg|pursu\w* (?:additional )?acquisitions|"
                     r"financial discipline|capital structure", re.I)

shutil.copy(LABELS, LABELS.with_name("labels_pass1_backup.csv"))
lab = pd.read_csv(LABELS)
if "capital_discipline" not in lab.columns:
    lab["capital_discipline"] = 0
    lab.to_csv(LABELS, index=False)
    print("Added capital_discipline column (all 0) to", LABELS)

sample = pd.read_csv(SAMPLE)
m = lab.merge(sample[["item_id", "activist", "subject_name", "item4_text"]], on="item_id", how="left")
rows = []
for _, r in m.iterrows():
    if str(r.get("notes", "")).startswith("EXCLUDE"):
        continue
    hits = set(h.lower() for h in NOTE_RE.findall(str(r.get("notes", ""))))
    hits |= set(h.lower() for h in TEXT_RE.findall(str(r["item4_text"])))
    if hits:
        rows.append({"item_id": r["item_id"], "activist": r["activist"], "company": r["subject_name"],
                     "matched": ", ".join(sorted(hits)), "notes": r.get("notes", "")})
out = pd.DataFrame(rows).sort_values("item_id")
out.to_csv("data/capital_discipline_review.csv", index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)
print(f"\n{len(out)} items to review (saved to data/capital_discipline_review.csv):")
print(out[["item_id", "activist", "company", "matched"]].to_string(index=False))
print("\nReminder: check capital_discipline only when a direction is stated (stop/limit acquisitions, cut or re-evaluate "
      "capex, reduce debt, improve returns on capital or capital allocation). If you check it on an item marked "
      "engagement_only, uncheck engagement_only.")
