#!/usr/bin/env python3
"""
Cleanup scan — flag initial 13Ds that probably aren't new activist campaigns,
using the patterns found during hand-labeling.

Reads   data/item4_texts.csv        (from extract_item4.py: Item 4 text for every campaign)
        data/activist_campaigns.csv (for repeat-company checks)
        data/exclusions.csv         (already excluded)
Writes  data/cleanup_candidates.csv  every flagged filing, with reasons and a text snippet
        data/proposed_exclusions.csv filings in "likely exclude" groups, in exclusions.csv
                                     format, for you to review before copying over

Usage
  python3 cleanup_scan.py

Groups
  likely exclude (not a campaign start):
    control         activist controls the company (IPO of its own subsidiary, GP control)
    amendment_text  "original" 13D whose Item 4 amends an earlier one
    insider         activist (or its founder) already a director / chairman
    inherited       shares received in a spin-off or separation
    prior_13d       a previous 13D on the company already existed
    restructuring   equity from a bankruptcy / recapitalization
  review (may be a real campaign, but the date or security may be off):
    settlement      describes a settlement / cooperation agreement already reached
    deal_end        support agreement to tender into or vote for a deal
    preferred_pipe  negotiated preferred / convertible investment
    foreign         non-U.S. issuer (ISIN, no CUSIP)
    repeat_company  same activist filed more than one initial 13D on the same company
"""
import re
from pathlib import Path

import pandas as pd

PATTERNS = {
    "control": r"general partner (?:has|will have) control|reorganization agreement|underwriting agreement|"
               r"in connection with the (?:issuer's )?initial public offering|lock-up agreement",
    "amendment_text": r"is hereby amended|is amended and supplemented|hereby amended to add|amended and restated in its entirety",
    "insider": r"\bis (?:a|the) (?:director|chairman|chair|member of the board)\b|serves? (?:as|on) the board of directors of the issuer|"
               r"\bis a director of the board\b",
    "inherited": r"in connection with the (?:separation|spin-?off|distribution)|\bformer parent\b",
    "prior_13d": r"previously filed (?:joint )?statements? on schedule 13d",
    "restructuring": r"plan of reorganization|chapter 11|emerg\w+ from bankruptcy|backstop|rights offering|"
                     r"debt restructuring|exchange offer",
    "settlement": r"cooperation agreement|settlement agreement|(?:irrevocably )?withdr[eaw]+ (?:its|their) (?:nomination|slate)|"
                  r"nominating agreement|governance agreement",
    "deal_end": r"tender and support agreement|support agreement|agreed to tender",
    "preferred_pipe": r"securities purchase agreement|convertible preferred|series [a-z] (?:convertible )?preferred",
}
LIKELY_EXCLUDE = ["control", "amendment_text", "insider", "inherited", "prior_13d", "restructuring"]
REGEX = {k: re.compile(v.replace(" ", r"\s+"), re.I) for k, v in PATTERNS.items()}   # tolerate line breaks


def snippet(text, m, width=160):
    a, b = max(0, m.start() - width // 2), min(len(text), m.end() + width // 2)
    return ("…" if a else "") + re.sub(r"\s+", " ", text[a:b]) + ("…" if b < len(text) else "")


def scan(texts: pd.DataFrame, campaigns: pd.DataFrame, excluded: set) -> pd.DataFrame:
    rows = []
    repeat = campaigns.groupby(["activist", "subject_cik"])["accession"].transform("count") > 1
    repeat_acc = set(campaigns.loc[repeat, "accession"])
    for _, r in texts.iterrows():
        text = str(r.get("item4_text") or "")
        flags, notes = [], []
        for k, rx in REGEX.items():
            m = rx.search(text)
            if m:
                flags.append(k)
                notes.append(f"[{k}] {snippet(text, m)}")
        if re.search(r"\bISIN\b", text) and not re.search(r"\bCUSIP\b", text):
            flags.append("foreign")
        if r["accession"] in repeat_acc:
            flags.append("repeat_company")
        if flags:
            rows.append({"accession": r["accession"], "activist": r["activist"], "subject_name": r["subject_name"],
                         "filed_date": r["filed_date"], "flags": ", ".join(flags),
                         "group": "likely exclude" if any(f in LIKELY_EXCLUDE for f in flags) else "review",
                         "already_excluded": r["accession"] in excluded, "evidence": " | ".join(notes)[:900],
                         "url": r.get("url", "")})
    return pd.DataFrame(rows)


def main():
    texts = pd.read_csv("data/item4_texts.csv", dtype={"accession": str})
    camp = pd.read_csv("data/activist_campaigns.csv", dtype={"accession": str, "subject_cik": str})
    ex_path = Path("data/exclusions.csv")
    excluded = set(pd.read_csv(ex_path, dtype=str)["accession"]) if ex_path.exists() else set()

    out = scan(texts, camp, excluded)
    out = out.sort_values(["group", "already_excluded", "activist", "filed_date"])
    out.to_csv("data/cleanup_candidates.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(f"Scanned {len(texts)} initial 13Ds; {len(out)} flagged "
          f"({out['already_excluded'].sum()} already in exclusions.csv)\n")
    flag_counts = out["flags"].str.split(", ").explode().value_counts()
    print("Flags (a filing can have several):\n" + flag_counts.to_string())

    new = out[~out["already_excluded"]]
    for grp in ["likely exclude", "review"]:
        g = new[new["group"] == grp]
        print(f"\n{grp.upper()}: {len(g)} not yet excluded")
        if len(g):
            print(g[["accession", "activist", "subject_name", "filed_date", "flags"]].to_string(index=False))

    caught = out[out["already_excluded"]]
    print(f"\nSanity check: the scan flagged {len(caught)} of the {len(excluded)} filings you excluded by hand.")
    missed = sorted(excluded - set(out["accession"]))
    if missed:
        print("Hand-excluded filings the scan did NOT flag (patterns to add?):", ", ".join(missed))

    prop = new[new["group"] == "likely exclude"].assign(
        reason=lambda d: "cleanup scan: " + d["flags"])[["accession", "reason", "subject_name"]]
    prop.to_csv("data/proposed_exclusions.csv", index=False)
    print(f"\nSaved data/cleanup_candidates.csv and data/proposed_exclusions.csv ({len(prop)} proposed). "
          "Open each proposed filing's evidence (or URL) before adding it to exclusions.csv.")


if __name__ == "__main__":
    main()
