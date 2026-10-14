#!/usr/bin/env python3
"""
Phase 1 — Activist Schedule 13D dataset from SEC EDGAR.

What it does
  1. Downloads EDGAR quarterly master indexes and keeps only Schedule 13D
     filings (originals + amendments).
  2. Keeps filings where an entity name matches a known activist.
  3. Fetches each filing's SEC header to confirm the activist is the FILER
     (not the subject) and to get the target's CIK, name, SIC code and the
     exact acceptance timestamp (needed for the Phase 2 event study).
  4. Writes:
       data/activist_13d_filings.csv  — every matching filing
       data/activist_campaigns.csv    — original 13Ds only (= campaign starts)

Usage
  pip install requests pandas
  python edgar_13d_activists.py --user-agent "Your Name your_email@example.com" \
      --start 2010 --end 2026

SEC requires a real name + email in the User-Agent and limits you to
10 requests/second. Everything is cached in ./cache, so reruns are fast and
you can edit the ACTIVISTS list without re-downloading the indexes.
"""

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Regexes are matched against EDGAR "conformed names" (upper case).
# Fund families file under several entity names, hence the alternations.
ACTIVISTS = {
    "Elliott":         r"^ELLIOTT (INVESTMENT MANAGEMENT|ASSOCIATES|INTERNATIONAL|MANAGEMENT|CAPITAL ADVISORS)",
    "Starboard":       r"^STARBOARD VALUE",
    "Trian":           r"^TRIAN (FUND MANAGEMENT|PARTNERS)",
    "ValueAct":        r"^VALUEACT",
    "Pershing Square": r"^PERSHING SQUARE",
    "Third Point":     r"^THIRD POINT (LLC|PARTNERS|OFFSHORE)",
    "Icahn":           r"^ICAHN (CARL|ENTERPRISES|CAPITAL|PARTNERS|ASSOCIATES|OFFSHORE)",
    "Jana":            r"^JANA PARTNERS",
}
ACTIVIST_RE = {k: re.compile(v, re.I) for k, v in ACTIVISTS.items()}

# Targets that are the activists' own vehicles (funds, BDCs, SPACs), not campaigns.
SELF_SUBJECT_RE = re.compile(r"ELLIOTT|STARBOARD|TRIAN|VALUEACT|PERSHING SQUARE|THIRD POINT|ICAHN|JANA PARTNERS", re.I)

# EDGAR renamed the form types when 13D went structured (Dec 2024).
ORIGINAL_FORMS = {"SC 13D", "SCHEDULE 13D"}
AMEND_FORMS = {"SC 13D/A", "SCHEDULE 13D/A"}
ALL_13D_FORMS = ORIGINAL_FORMS | AMEND_FORMS

ARCHIVES = "https://www.sec.gov/Archives/"


# ---------------------------------------------------------------------------
# Polite EDGAR client: throttling, retries, caching dir
# ---------------------------------------------------------------------------

class Edgar:
    def __init__(self, user_agent: str, cache_dir: str = "cache", min_interval: float = 0.12):
        if "@" not in user_agent:
            sys.exit("SEC requires an email in the User-Agent, e.g. 'Jane Doe jane@school.edu'")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def get(self, url: str, stream: bool = False) -> requests.Response:
        for attempt in range(5):
            self._throttle()
            try:
                r = self.s.get(url, timeout=60, stream=stream)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 500, 502, 503, 504):
                # 403/429 from SEC usually means "slow down" (or a bad User-Agent).
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
        raise RuntimeError(f"Giving up after retries: {url}")


# ---------------------------------------------------------------------------
# Step 1: quarterly master indexes -> all 13D-family rows
# ---------------------------------------------------------------------------

def parse_master_index(text: str) -> pd.DataFrame:
    """master.idx is pipe-delimited after a dashed separator line."""
    rows, started = [], False
    for line in text.splitlines():
        if not started:
            started = line.startswith("-----")
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, name, form, filed, fname = (p.strip() for p in parts)
        if form in ALL_13D_FORMS:
            rows.append({"index_cik": cik, "index_name": name, "form_type": form,
                         "filed_date": filed, "filename": fname})
    return pd.DataFrame(rows, columns=["index_cik", "index_name", "form_type", "filed_date", "filename"])


def quarter_13d_rows(edgar: Edgar, year: int, qtr: int) -> pd.DataFrame:
    """Return all 13D-family index rows for one quarter (cached as a small CSV)."""
    cache_file = edgar.cache / "index13d" / f"{year}_Q{qtr}.csv"
    today = date.today()
    current_q = (today.year, (today.month - 1) // 3 + 1)
    is_current = (year, qtr) == current_q

    if cache_file.exists() and not is_current:      # current quarter is still growing
        return pd.read_csv(cache_file, dtype=str)

    url = f"{ARCHIVES}edgar/full-index/{year}/QTR{qtr}/master.idx"
    r = edgar.get(url)
    df = parse_master_index(r.content.decode("latin-1"))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def match_activist(name: str):
    for key, rx in ACTIVIST_RE.items():
        if rx.search(name or ""):
            return key
    return None


# ---------------------------------------------------------------------------
# Step 2: SEC header -> who filed, who is the target, SIC, timestamp
# ---------------------------------------------------------------------------

def fetch_header(edgar: Edgar, filename: str) -> str:
    """Stream the full submission .txt but stop once the SEC header ends."""
    acc = Path(filename).stem
    cache_file = edgar.cache / "headers" / f"{acc}.hdr"
    if cache_file.exists():
        return cache_file.read_text(encoding="latin-1")

    r = edgar.get(ARCHIVES + filename, stream=True)
    buf = b""
    for chunk in r.iter_content(8192):
        buf += chunk
        if b"</SEC-HEADER>" in buf or len(buf) > 300_000:
            break
    r.close()
    text = buf.decode("latin-1")
    end = text.find("</SEC-HEADER>")
    if end != -1:
        text = text[:end]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="latin-1")
    return text


def _field(block: str, label: str):
    m = re.search(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", block, re.M)
    return m.group(1) if m else None


def _entity(block: str) -> dict:
    sic_raw = _field(block, "STANDARD INDUSTRIAL CLASSIFICATION")
    sic_code = None
    if sic_raw:
        m = re.search(r"\[(\d{3,4})\]", sic_raw)
        sic_code = int(m.group(1)) if m else None
    return {
        "name": _field(block, "COMPANY CONFORMED NAME"),
        "cik": _field(block, "CENTRAL INDEX KEY"),
        "sic_code": sic_code,
        "sic_desc": re.sub(r"\s*\[\d+\]\s*$", "", sic_raw) if sic_raw else None,
        "state_inc": _field(block, "STATE OF INCORPORATION"),
    }


def parse_header(text: str) -> dict:
    out = {
        "accession": _field(text, "ACCESSION NUMBER"),
        "header_form_type": _field(text, "CONFORMED SUBMISSION TYPE"),
        "acceptance_datetime": None,
        "subjects": [],
        "filers": [],
    }
    m = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", text)
    if m:
        out["acceptance_datetime"] = m.group(1)

    # re.split with a capture group -> [preamble, label, body, label, body, ...]
    parts = re.split(r"^(SUBJECT COMPANY|FILED BY):", text, flags=re.M)
    for label, body in zip(parts[1::2], parts[2::2]):
        ent = _entity(body)
        (out["subjects"] if label == "SUBJECT COMPANY" else out["filers"]).append(ent)
    return out


# ---------------------------------------------------------------------------
# Step 3: assemble datasets
# ---------------------------------------------------------------------------

def sector_exclusion(sic):
    """Flag sectors where EV/EBITDA-style peer comparisons break down."""
    if sic is None or pd.isna(sic):
        return None
    sic = int(sic)
    if 6000 <= sic <= 6799:
        return "financials/REITs"
    if 4900 <= sic <= 4999:
        return "utilities"
    return None


def build_filings(edgar: Edgar, start: int, end: int) -> pd.DataFrame:
    today = date.today()
    current_q = (today.year, (today.month - 1) // 3 + 1)

    frames = []
    for year in range(start, end + 1):
        for qtr in range(1, 5):
            if (year, qtr) > current_q:
                break
            try:
                df = quarter_13d_rows(edgar, year, qtr)
            except Exception as e:
                print(f"  ! index {year} Q{qtr} failed: {e}")
                continue
            frames.append(df)
            print(f"  index {year} Q{qtr}: {len(df):,} 13D-family rows")
    idx = pd.concat(frames, ignore_index=True)

    # A filing appears once per entity attached to it; keep filings where ANY
    # attached entity looks like an activist, then verify with the header.
    # The same filing is listed under each attached entity's CIK folder, so the
    # filename differs but the accession number is shared: dedupe on accession.
    idx["accession"] = idx["filename"].map(lambda f: Path(f).stem)
    idx["activist_guess"] = idx["index_name"].map(match_activist)
    cand = (idx.loc[idx["activist_guess"].notna()]
            .drop_duplicates("accession")["filename"].tolist())
    print(f"\n{len(cand):,} candidate filings — fetching SEC headers...")

    rows = []
    for i, fname in enumerate(cand, 1):
        try:
            h = parse_header(fetch_header(edgar, fname))
        except Exception as e:
            print(f"  ! header failed {fname}: {e}")
            continue

        activist, filer = None, None
        for f in h["filers"]:
            activist = match_activist(f["name"])
            if activist:
                filer = f
                break
        if not activist or not h["subjects"]:
            continue  # activist name was the subject company, or malformed header
        subj = h["subjects"][0]
        if SELF_SUBJECT_RE.search(subj["name"] or ""):
            continue  # activist filing on its own fund/vehicle, e.g. Icahn Enterprises

        acc_dt = h["acceptance_datetime"]
        rows.append({
            "accession": h["accession"] or Path(fname).stem,
            "form_type": h["header_form_type"],
            "activist": activist,
            "filer_name": filer["name"],
            "filer_cik": filer["cik"],
            "subject_name": subj["name"],
            "subject_cik": subj["cik"],
            "subject_sic_code": subj["sic_code"],
            "subject_sic_desc": subj["sic_desc"],
            "subject_state_inc": subj["state_inc"],
            "acceptance_datetime": acc_dt,
            # EDGAR timestamps are US Eastern; after 4pm the market reacts next day.
            "after_market_close": (int(acc_dt[8:12]) >= 1600) if acc_dt else None,
            "url": ARCHIVES + fname,
        })
        if i % 100 == 0:
            print(f"  {i:,}/{len(cand):,} headers processed")

    filings = pd.DataFrame(rows).drop_duplicates("accession")
    filed = idx.drop_duplicates("accession")
    filings = filings.merge(filed[["accession", "filed_date"]], on="accession", how="left")
    filings["filed_date"] = pd.to_datetime(filings["filed_date"])
    filings["sector_exclusion"] = filings["subject_sic_code"].map(sector_exclusion)
    return filings.sort_values("filed_date").reset_index(drop=True)


def build_campaigns(filings: pd.DataFrame) -> pd.DataFrame:
    """One row per original 13D, with a count of amendments that followed it."""
    filings = filings.drop_duplicates("accession")
    is_orig = filings["form_type"].isin(ORIGINAL_FORMS)
    orig = filings[is_orig].copy()
    amend = filings[~is_orig]

    orig["n_amendments"] = 0
    for (act, cik), g in orig.groupby(["activist", "subject_cik"]):
        a = amend.loc[(amend["activist"] == act) & (amend["subject_cik"] == cik), "filed_date"]
        bounds = list(g["filed_date"]) + [pd.Timestamp.max]
        for j, ix in enumerate(g.index):
            orig.at[ix, "n_amendments"] = int(((a >= bounds[j]) & (a < bounds[j + 1])).sum())

    cols = ["filed_date", "activist", "subject_name", "subject_cik", "subject_sic_code",
            "subject_sic_desc", "sector_exclusion", "acceptance_datetime",
            "after_market_close", "n_amendments", "filer_name", "accession", "url"]
    return orig[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", required=True, help='e.g. "Your Name you@example.com"')
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=date.today().year)
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    edgar = Edgar(args.user_agent, args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building activist 13D dataset {args.start}-{args.end}\n")
    filings = build_filings(edgar, args.start, args.end)
    campaigns = build_campaigns(filings)

    filings.to_csv(out / "activist_13d_filings.csv", index=False)
    campaigns.to_csv(out / "activist_campaigns.csv", index=False)

    print(f"\nSaved {len(filings):,} filings and {len(campaigns):,} campaigns to {out}/")
    print("\nCampaign starts by activist:")
    print(campaigns["activist"].value_counts().to_string())
    print(f"\nExcluded-sector campaigns: {campaigns['sector_exclusion'].notna().sum()}")


if __name__ == "__main__":
    main()
