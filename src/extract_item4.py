#!/usr/bin/env python3
"""
LLM evaluation project, step 1: extract "Item 4. Purpose of Transaction" from
each original activist 13D, and draw a stratified sample for hand-labeling.

Reads   data/activist_campaigns.csv  (from edgar_13d_activists.py)
        data/exclusions.csv          (optional)
Writes  data/item4_texts.csv         every filing: extracted text + how it was found
        data/label_sample.csv        ~100 filings to hand-label (fixed random seed)

Usage
  python3 extract_item4.py --user-agent "Your Name your_email@example.com"
  python3 extract_item4.py --user-agent "..." --sample-size 60     # weekend version

Filings are cached in ./cache/filings, so reruns don't hit the SEC again.
"""

import argparse
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

FORMS_13D = ("SC 13D", "SCHEDULE 13D")
MAX_CHARS = 60_000

START_STRICT = re.compile(
    r"item\s*4\s*[\.\:\-\u2013\u2014]?\s*(?:\(\w\)\s*)?purpose\s+of\s+(?:the\s+)?transactions?", re.I)
END_STRICT = re.compile(
    r"item\s*5\s*[\.\:\-\u2013\u2014]?\s*(?:\(\w\)\s*)?interests?\s+in\s+(?:the\s+)?securities", re.I)
START_LOOSE = re.compile(r"\bitem\s*4\b", re.I)
END_LOOSE = re.compile(r"\bitem\s*5\b", re.I)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class Edgar:
    def __init__(self, ua: str, cache: Path):
        if "@" not in ua:
            raise SystemExit("SEC requires an email in the User-Agent, e.g. 'Jane Doe jane@school.edu'")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.last = 0.0

    def get(self, url: str, accession: str) -> str:
        f = self.cache / f"{accession}.txt"
        if f.exists():
            return f.read_text(encoding="latin-1")
        for attempt in range(5):
            wait = 0.12 - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                text = r.content.decode("latin-1")
                f.write_text(text, encoding="latin-1")
                return text
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url}")


# ---------------------------------------------------------------------------
# Parsing (pure functions, testable)
# ---------------------------------------------------------------------------

def primary_document(raw: str) -> tuple[str, str]:
    """Return (document type, document body) of the 13D itself, not its exhibits."""
    docs = re.split(r"<DOCUMENT>", raw, flags=re.I)[1:]
    parsed = []
    for d in docs:
        t = re.search(r"<TYPE>\s*([^\n<]+)", d, re.I)
        body = re.search(r"<TEXT>(.*?)</TEXT>", d, re.I | re.S)
        parsed.append(((t.group(1).strip() if t else ""), body.group(1) if body else d))
    for typ, body in parsed:
        if typ.upper().startswith(FORMS_13D):
            return typ, body
    return parsed[0] if parsed else ("", raw)


def html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d|table)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def from_xml(body: str):
    """Structured filings (Dec 2024+): find an element whose name mentions 'purpose'."""
    hits = re.findall(r"<([\w:]*purpose\w*)[^>]*>(.*?)</\1>", body, re.I | re.S)
    texts = [html_to_text(h[1]) for h in hits if len(html_to_text(h[1])) > 20]
    return max(texts, key=len) if texts else None


def best_section(text: str, start_re, end_re):
    """Longest span from a start heading to the next end heading. Picking the longest
    skips cross-references like 'as described in Item 4' that appear earlier."""
    best = None
    for m in start_re.finditer(text):
        e = end_re.search(text, m.end())
        if not e:
            continue
        span = text[m.end():e.start()].strip(" .:-\n")
        if best is None or len(span) > len(best):
            best = span
    return best


def extract_item4(raw: str) -> tuple[str | None, str]:
    typ, body = primary_document(raw)
    if "<?xml" in body[:500].lower() or re.search(r"<\w*schedule13d", body[:5000], re.I):
        t = from_xml(body)
        if t:
            return t[:MAX_CHARS], "xml_element"
    text = html_to_text(body) if re.search(r"<(html|body|p|div|table)\b", body, re.I) else body
    text = re.sub(r"[ \t]+", " ", text)
    for start_re, end_re, how in [(START_STRICT, END_STRICT, "heading_strict"),
                                  (START_LOOSE, END_LOOSE, "heading_loose")]:
        sec = best_section(text, start_re, end_re)
        if sec and len(sec) >= 40:
            return sec[:MAX_CHARS], how
    return None, "not_found"


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """At least 3 per activist where available, the rest proportional to each activist's share."""
    parts = []
    for act, g in df.groupby("activist"):
        k = max(min(3, len(g)), round(n * len(g) / len(df)))
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    elif len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(n=min(n - len(out), len(rest)), random_state=seed)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)   # shuffle labeling order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", required=True)
    ap.add_argument("--campaigns", default="data/activist_campaigns.csv")
    ap.add_argument("--exclude", default="data/exclusions.csv")
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--cache", default="cache/filings")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    camp = pd.read_csv(args.campaigns, dtype={"accession": str}, parse_dates=["filed_date"])
    if Path(args.exclude).exists():
        camp = camp[~camp["accession"].isin(pd.read_csv(args.exclude, dtype=str)["accession"])]
    print(f"{len(camp)} original 13D filings to process")

    edgar = Edgar(args.user_agent, Path(args.cache))
    rows = []
    for i, r in enumerate(camp.itertuples(index=False), 1):
        try:
            text, how = extract_item4(edgar.get(r.url, r.accession))
        except Exception as e:
            text, how = None, f"error: {type(e).__name__}"
        rows.append({"accession": r.accession, "activist": r.activist, "subject_name": r.subject_name,
                     "filed_date": r.filed_date.date(), "url": r.url, "parse_method": how,
                     "n_chars": len(text) if text else 0, "item4_text": text})
        if i % 50 == 0:
            print(f"  {i}/{len(camp)}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out / "item4_texts.csv", index=False)

    ok = df[df["n_chars"] > 0]
    print(f"\nExtracted Item 4 for {len(ok)} of {len(df)} filings")
    print("How it was found:", df["parse_method"].value_counts().to_dict())
    print(f"Length (characters): median {ok['n_chars'].median():.0f}, "
          f"10th pct {ok['n_chars'].quantile(.1):.0f}, 90th pct {ok['n_chars'].quantile(.9):.0f}")
    fails = df[df["n_chars"] == 0]
    if len(fails):
        print("\nNot found (open a few URLs to see why):")
        print(fails[["activist", "subject_name", "filed_date", "url"]].head(10).to_string(index=False))

    pool = ok[ok["n_chars"].between(200, 30_000)]
    sample = stratified_sample(pool, args.sample_size)
    sample.insert(0, "item_id", range(1, len(sample) + 1))
    sample.to_csv(out / "label_sample.csv", index=False)
    print(f"\nSaved item4_texts.csv and a labeling sample of {len(sample)} filings to {out}/label_sample.csv")
    print("By activist:", sample["activist"].value_counts().to_dict())


if __name__ == "__main__":
    main()
