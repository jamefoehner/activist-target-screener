#!/usr/bin/env python3
"""
Phase 3a — Build the firm-year dataset for the target prediction model.

One row per company per fiscal year, describing what an investor could see
about the company at that time, plus a label:
    targeted = 1 if one of our activists filed a 13D on it within the
               12 months after that information became public.

Avoiding hindsight ("look-ahead bias")
  * Annual numbers are treated as public 4 months after fiscal year-end
    (the 10-K deadline is 60-90 days, so this is conservative).
  * Stock-return features use only months before that date.
  * The 12-month labeling window starts after that date.

Output
  data/panel.csv  with a `sample` column:
     "labeled" — rows whose full 12-month window is observed (train/test on these)
     "current" — each firm's latest row, window still open (Phase 4 scores these)

Usage
  python build_panel.py --wrds-user YOUR_WRDS_USERNAME
  (reuses ./cache_wrds from the event study; reads data/event_study_results.csv
   and data/exclusions.csv)
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_LAG_MONTHS = 4
LABEL_MONTHS = 12
CAMPAIGN_DATA_START = pd.Timestamp("2010-01-01")
FUNDA_START = "2005-01-01"

# features compared to industry peers (company value minus peer median)
PEER_RELATIVE = ["ebitda_ev", "book_mkt", "ebitda_margin", "roa", "sales_growth",
                 "cash_assets", "leverage", "payout", "acq_assets", "capex_assets", "rd_sales"]
OTHER_FEATURES = ["log_mktcap", "ret_1y_rel", "tsr_3y_rel"]
FEATURES = [f"{c}_rel" for c in PEER_RELATIVE] + OTHER_FEATURES


# ---------------------------------------------------------------------------
# WRDS pulls
# ---------------------------------------------------------------------------

def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn()
    df.to_parquet(f)
    return df


def pull_all(db, cache: Path):
    funda = cached(cache, "funda", lambda: db.raw_sql(
        f"""select gvkey, datadate, fyear, sich, at, sale, oibdp, oiadp, ni, che,
                   dltt, dlc, ceq, csho, prcc_f, capx, xrd, dv, prstkc, aqc
            from comp.funda
            where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
              and datadate >= '{FUNDA_START}'""", date_cols=["datadate"]))

    company = cached(cache, "comp_company_sic", lambda: db.raw_sql(
        "select gvkey, conm, sic from comp.company"))

    link = cached(cache, "ccm_link", lambda: db.raw_sql(
        """select gvkey, lpermno as permno, linkdt, linkenddt, linkprim
           from crsp.ccmxpf_lnkhist
           where linktype in ('LU','LC') and linkprim in ('P','C')""",
        date_cols=["linkdt", "linkenddt"]))

    def monthly():
        try:
            return db.raw_sql(
                f"""select permno, mthcaldt as date, mthret as ret from crsp.msf_v2
                    where mthcaldt >= '{FUNDA_START}'""", date_cols=["date"])
        except Exception as e:
            print(f"  crsp.msf_v2 unavailable ({type(e).__name__}); using legacy crsp.msf (ends 2024)")
            return db.raw_sql(
                f"select permno, date, ret from crsp.msf where date >= '{FUNDA_START}'",
                date_cols=["date"])

    msf = cached(cache, "crsp_monthly", monthly)
    ff = cached(cache, "ff_monthly", lambda: db.raw_sql(
        f"select date, mktrf, rf from ff.factors_monthly where date >= '{FUNDA_START}'",
        date_cols=["date"]))
    # latest daily price per company from Compustat (updated daily, unlike CRSP);
    # cached per calendar day so "still trading" is always judged on fresh data
    last_px = cached(cache, f"secd_last_{date.today():%Y%m%d}", lambda: db.raw_sql(
        """select gvkey, max(datadate) as last_price_date from comp.secd
           where datadate >= current_date - interval '120 days' group by gvkey""",
        date_cols=["last_price_date"]))
    return funda, company, link, msf, ff, last_px


# ---------------------------------------------------------------------------
# Panel construction (pure pandas, testable without WRDS)
# ---------------------------------------------------------------------------

def attach_permno(d: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    link = link.copy()
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    m = d.merge(link, on="gvkey", how="inner")
    m = m[(m["datadate"] >= m["linkdt"]) & (m["datadate"] <= m["linkenddt"])]
    m = m.assign(_r=(m["linkprim"] != "P").astype(int)).sort_values("_r")
    m = m.drop_duplicates(["gvkey", "datadate"]).drop(columns=["_r", "linkdt", "linkenddt", "linkprim"])
    m["permno"] = m["permno"].astype(int)
    return m


def return_features(msf: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """Trailing 12- and 36-month stock returns minus the market's, by permno-month."""
    msf = msf.dropna(subset=["ret"]).copy()
    msf["month"] = msf["date"].dt.to_period("M")
    wide = msf.pivot_table(index="month", columns="permno", values="ret", aggfunc="last")
    wide = np.log1p(wide)

    ff = ff.copy()
    ff["month"] = ff["date"].dt.to_period("M")
    mkt = np.log1p(ff.groupby("month")[["mktrf", "rf"]].last().sum(axis=1)).reindex(wide.index)

    out = []
    for name, n, minp in [("ret_1y_rel", 12, 10), ("tsr_3y_rel", 36, 30)]:
        firm = np.expm1(wide.rolling(n, min_periods=minp).sum())
        market = np.expm1(mkt.rolling(n, min_periods=minp).sum())
        rel = firm.sub(market, axis=0)
        out.append(rel.stack().rename(name))
    return pd.concat(out, axis=1).reset_index()          # columns: month, permno, ret_1y_rel, tsr_3y_rel


def industry_relative(d: pd.DataFrame, cols, min_peers: int = 5) -> pd.DataFrame:
    """Company value minus the median of its 2-digit-SIC peers that year
       (1-digit SIC if fewer than `min_peers` peers have data)."""
    for c in cols:
        g2 = d.groupby(["fyear", "sic2"])[c]
        g1 = d.groupby(["fyear", "sic1"])[c]
        med = np.where(g2.transform("count") >= min_peers, g2.transform("median"), g1.transform("median"))
        d[f"{c}_rel"] = d[c] - med
    return d


def winsorize_by_year(d: pd.DataFrame, cols, p: float = 0.01) -> pd.DataFrame:
    for c in cols:
        lo = d.groupby("fyear")[c].transform(lambda x: x.quantile(p))
        hi = d.groupby("fyear")[c].transform(lambda x: x.quantile(1 - p))
        d[c] = d[c].clip(lo, hi)
    return d


def plain_numeric(df: pd.DataFrame, skip=()) -> pd.DataFrame:
    """WRDS can return pandas 'nullable' columns (Float64/Int64 with pd.NA), which
       break numpy comparisons. Convert numeric columns to ordinary floats (NaN)."""
    df = df.copy()
    for c in df.columns:
        if c in skip or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or c in ("sich", "sic"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def build_panel(funda, company, link, msf, ff, campaigns, min_mktcap, label_end, last_px=None):
    funda = plain_numeric(funda, skip=("gvkey",))
    company = plain_numeric(company, skip=("gvkey", "conm"))
    link = plain_numeric(link, skip=("gvkey", "linkprim"))
    msf, ff = plain_numeric(msf), plain_numeric(ff)
    link = link.dropna(subset=["permno"])
    funda = funda.dropna(subset=["fyear"])
    d = funda.dropna(subset=["at", "sale", "prcc_f", "csho"]).copy()
    d = d[(d["at"] > 0) & (d["sale"] > 0)]
    d = d.sort_values("datadate").drop_duplicates(["gvkey", "fyear"], keep="last")

    # industry: historical SIC if available, else current
    d = d.merge(company, on="gvkey", how="left")
    d["sic"] = pd.to_numeric(d["sich"], errors="coerce").fillna(pd.to_numeric(d["sic"], errors="coerce"))
    d = d.dropna(subset=["sic"])
    d["sic"] = d["sic"].astype(int)
    d = d[~d["sic"].between(6000, 6799) & ~d["sic"].between(4900, 4999)]   # same exclusions as before
    d["sic2"], d["sic1"] = d["sic"] // 100, d["sic"] // 1000

    # prior-year sales for growth (only if the prior row is the previous fiscal year)
    d = d.sort_values(["gvkey", "fyear"])
    prev = d.groupby("gvkey")[["fyear", "sale"]].shift(1)
    d["sale_lag"] = np.where(prev["fyear"] == d["fyear"] - 1, prev["sale"], np.nan)

    # raw features ($ values in Compustat are in millions)
    z = lambda s: s.fillna(0)
    d["mktcap"] = d["prcc_f"] * d["csho"]
    debt = z(d["dltt"]) + z(d["dlc"])
    ev = d["mktcap"] + debt - z(d["che"])
    d["ebitda_ev"] = np.where(ev > 0, d["oibdp"] / ev, np.nan)       # higher = cheaper
    d["book_mkt"] = d["ceq"] / d["mktcap"]                            # higher = cheaper
    d["ebitda_margin"] = d["oibdp"] / d["sale"]
    d["roa"] = d["oiadp"] / d["at"]
    d["sales_growth"] = d["sale"] / d["sale_lag"] - 1
    d["cash_assets"] = d["che"] / d["at"]
    d["leverage"] = debt / d["at"]
    d["payout"] = (z(d["dv"]) + z(d["prstkc"])) / d["at"]
    d["acq_assets"] = z(d["aqc"]) / d["at"]
    d["capex_assets"] = d["capx"] / d["at"]
    d["rd_sales"] = z(d["xrd"]) / d["sale"]
    d["log_mktcap"] = np.log(d["mktcap"].where(d["mktcap"] > 0))

    d = d[d["mktcap"] >= min_mktcap]
    d = attach_permno(d, link)

    # when the information was public, and stock returns up to the month before
    d["avail_date"] = d["datadate"] + pd.DateOffset(months=REPORT_LAG_MONTHS)
    d["month"] = d["avail_date"].dt.to_period("M") - 1
    d = d.merge(return_features(msf, ff), on=["permno", "month"], how="left")

    d = industry_relative(d, PEER_RELATIVE)
    d = winsorize_by_year(d, FEATURES)

    # labels: a campaign that starts in (avail_date, avail_date + 12 months]
    d["window_end"] = d["avail_date"] + pd.DateOffset(months=LABEL_MONTHS)
    hits = d[["gvkey", "datadate", "permno", "avail_date", "window_end"]].merge(
        campaigns[["permno", "filed_date", "activist", "accession"]], on="permno")
    hits = hits[(hits["filed_date"] > hits["avail_date"]) & (hits["filed_date"] <= hits["window_end"])]
    lab = (hits.sort_values("filed_date").groupby(["gvkey", "datadate"])
               .agg(first_campaign=("filed_date", "first"), activist=("activist", "first"),
                    n_campaigns=("accession", "nunique")).reset_index())
    d = d.merge(lab, on=["gvkey", "datadate"], how="left")
    d["targeted"] = d["n_campaigns"].notna().astype(int)

    # sample: fully observed windows vs each firm's latest (still-open) row
    full = (d["avail_date"] >= CAMPAIGN_DATA_START) & (d["window_end"] <= label_end)
    latest = d["datadate"] == d.groupby("gvkey")["datadate"].transform("max")
    open_now = (d["window_end"] > label_end) & (d["avail_date"] <= pd.Timestamp(date.today()))
    # only score companies that still trade: CRSP has a return for them within ~2 months
    # of the latest month in the data (drops firms acquired or delisted since their last 10-K)
    candidates = latest & open_now

    def crsp_check():
        # fallback: CRSP monthly return within ~2 months of the latest month (CRSP can lag)
        last_month = (msf.dropna(subset=["ret"]).assign(m=lambda x: x["date"].dt.to_period("M"))
                         .groupby("permno")["m"].max())
        return d["permno"].map(last_month) >= last_month.max() - 2

    still_trading = None
    if last_px is not None and len(last_px):
        # preferred: a Compustat daily price within 10 days of the most recent price date
        lp = last_px.copy()
        lp["gvkey"] = lp["gvkey"].astype(str).str.strip().str.zfill(6)
        lp["last_price_date"] = pd.to_datetime(lp["last_price_date"], errors="coerce")
        lp = lp.dropna().groupby("gvkey")["last_price_date"].max()
        ref = min(lp.max(), pd.Timestamp(date.today()))
        cutoff = ref - pd.Timedelta(days=10)
        px_date = d["gvkey"].astype(str).str.strip().str.zfill(6).map(lp)
        check = px_date >= cutoff
        n_cand, n_ok = int(candidates.sum()), int((candidates & check).sum())
        print(f"  trading check (Compustat daily prices, latest {ref.date()}): "
              f"{n_ok:,} of {n_cand:,} current companies have a price since {cutoff.date()}")
        if n_cand and n_ok < 0.5 * n_cand:
            print("  ! that looks wrong (too many dropped) — falling back to the CRSP check")
        else:
            still_trading = check
    if still_trading is None:
        still_trading = crsp_check()
    d["sample"] = np.select([full, candidates & still_trading], ["labeled", "current"], default="unused")
    dropped = int((candidates & ~still_trading).sum())
    if dropped:
        print(f"  dropped {dropped} companies from the current universe that no longer trade")

    keep = (["gvkey", "permno", "conm", "fyear", "datadate", "avail_date", "sic", "sic2", "mktcap"]
            + FEATURES + ["targeted", "first_campaign", "activist", "sample"])
    return d[keep].sort_values(["avail_date", "gvkey"]).reset_index(drop=True), hits


def report(panel: pd.DataFrame, hits: pd.DataFrame, campaigns: pd.DataFrame):
    lab = panel[panel["sample"] == "labeled"]
    print(f"\nLabeled firm-years: {len(lab):,}   targeted: {lab['targeted'].sum()}   "
          f"base rate: {lab['targeted'].mean():.2%}")
    by_year = lab.groupby(lab["avail_date"].dt.year).agg(firms=("gvkey", "size"), targeted=("targeted", "sum"))
    print("\nBy year information became public:\n" + by_year.to_string())

    used = hits.merge(lab[["gvkey", "datadate"]], on=["gvkey", "datadate"])["accession"].nunique()
    print(f"\nCampaigns captured as a positive label: {used} of {campaigns['accession'].nunique()} "
          "(the rest fall outside the universe: below size floor, missing data, or no prior-year filing)")

    med = lab.groupby("targeted")[FEATURES].median().T
    med.columns = ["not targeted", "targeted"]
    print("\nMedian features, targeted vs not (sanity check — targets should look cheaper/weaker):")
    print(med.round(3).to_string())

    cur = panel[panel["sample"] == "current"]
    print(f"\nCurrent universe for scoring: {len(cur):,} firms "
          f"({(cur['mktcap'] >= 1000).sum():,} with market cap >= $1B)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--results", default="data/event_study_results.csv")
    ap.add_argument("--exclude", default="data/exclusions.csv")
    ap.add_argument("--min-mktcap", type=float, default=100.0, help="$ millions at fiscal year-end")
    ap.add_argument("--label-end", default=str(date.today()),
                    help="last date your 13D data covers (default: today)")
    ap.add_argument("--cache", default="cache_wrds")
    ap.add_argument("--out", default="data/panel.csv")
    args = ap.parse_args()

    camp = pd.read_csv(args.results, parse_dates=["filed_date"], dtype={"accession": str})
    if Path(args.exclude).exists():
        camp = camp[~camp["accession"].isin(pd.read_csv(args.exclude, dtype=str)["accession"])]
    camp = camp[camp["permno"].notna()].copy()
    camp["permno"] = camp["permno"].astype(int)

    import wrds
    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)
    db = wrds.Connection(wrds_username=args.wrds_user)
    print("Pulling Compustat annual, CRSP monthly, and market data (first run takes a few minutes)...")
    funda, company, link, msf, ff, last_px = pull_all(db, cache)
    db.close()

    panel, hits = build_panel(funda, company, link, msf, ff, camp,
                              args.min_mktcap, pd.Timestamp(args.label_end), last_px)
    panel.to_csv(args.out, index=False)
    report(panel, hits, camp)
    print(f"\nSaved {len(panel):,} rows to {args.out}")


if __name__ == "__main__":
    main()
