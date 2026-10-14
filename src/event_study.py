#!/usr/bin/env python3
"""
Phase 2 — Event study: how did target stocks react to activist 13D filings?

Steps
  1. Link each target's SEC CIK -> Compustat GVKEY -> CRSP PERMNO
     (the link valid on the campaign date, so delisted firms are kept).
  2. Pull daily stock returns from CRSP (delisting returns included) and
     daily market returns (Fama-French market factor + risk-free rate).
  3. Set each campaign's event day: the filing's acceptance date, moved to the
     next trading day when the 13D was accepted after the 4pm close.
  4. Compute per campaign:
       CAR[-1,+1], CAR[-10,+20]   market-adjusted (stock minus market)
       CAR_MM[-10,+20]            market-model adjusted (alpha/beta from days -250..-30)
       BHAR[0,+252]               one-year buy-and-hold return minus the market's
  5. Write results, summary tables and an average-CAR chart.

Usage
  pip install wrds pandas numpy matplotlib pyarrow
  python event_study.py --wrds-user YOUR_WRDS_USERNAME

The first WRDS connection asks for your password (and may need a Duo
approval) and offers to save a .pgpass file so later runs don't ask again.
Pulled data is cached in ./cache_wrds; delete that folder to re-pull.
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

PRE, POST = 260, 260            # calendar positions kept around each event
EST_WINDOW = (-250, -30)        # market-model estimation window (trading days)
MIN_EST_OBS = 120


# ---------------------------------------------------------------------------
# Data loading (WRDS)
# ---------------------------------------------------------------------------

def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn()
    df.to_parquet(f)
    return df


def load_campaigns(path: str) -> pd.DataFrame:
    c = pd.read_csv(path, dtype={"subject_cik": str, "acceptance_datetime": str, "accession": str},
                    parse_dates=["filed_date"])
    c["cik_int"] = pd.to_numeric(c["subject_cik"], errors="coerce").astype("Int64")
    return c


_SUFFIX = re.compile(r"\b(THE|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LP|L P|NV|N V|SA|AG|LLC|NEW)\b")


def norm_name(name) -> str:
    """'EMULEX CORP /DE/' -> 'EMULEX'; used for the name-match fallback."""
    if not isinstance(name, str):
        return ""
    n = re.sub(r"/[A-Z]{2,3}/", " ", name.upper())   # state tags like /DE/
    n = re.sub(r"[^A-Z0-9 ]", " ", n)
    n = _SUFFIX.sub(" ", n)
    return " ".join(n.split())


def _ccm_on_date(m: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Join gvkeys to CRSP links valid on each campaign date; one row per accession."""
    m = m.merge(link, on="gvkey", how="inner")
    m = m[(m["filed_date"] >= m["linkdt"]) & (m["filed_date"] <= m["linkenddt"])].copy()
    m["prim_rank"] = (m["linkprim"] != "P").astype(int)       # prefer primary link
    return m.sort_values("prim_rank").drop_duplicates("accession")[["accession", "gvkey", "permno"]]


def map_permnos(db, camp: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Three passes, each only for campaigns still unmatched:
       1. Compustat's CIK (comp.company — stores only each firm's latest CIK)
       2. WRDS SEC Analytics historical CIK-GVKEY links (if your WRDS has it)
       3. company name match against CRSP's dated name history"""
    link = cached(cache, "ccm_link", lambda: db.raw_sql(
        """select gvkey, lpermno as permno, linkdt, linkenddt, linkprim
           from crsp.ccmxpf_lnkhist
           where linktype in ('LU','LC') and linkprim in ('P','C')""",
        date_cols=["linkdt", "linkenddt"]))
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    found = []

    # pass 1
    comp = cached(cache, "comp_company", lambda: db.raw_sql(
        "select gvkey, cik from comp.company where cik is not null"))
    comp["cik_int"] = pd.to_numeric(comp["cik"], errors="coerce").astype("Int64")
    p1 = _ccm_on_date(camp.merge(comp[["cik_int", "gvkey"]], on="cik_int"), link)
    found.append(p1.assign(link_method="compustat_cik"))

    # pass 2
    todo = camp[~camp["accession"].isin(pd.concat(found)["accession"])]
    try:
        hist = cached(cache, "wrdssec_ciklink", lambda: db.raw_sql(
            "select distinct gvkey, cik from wrdssec.wciklink_gvkey where cik is not null"))
        hist["cik_int"] = pd.to_numeric(hist["cik"], errors="coerce").astype("Int64")
        p2 = _ccm_on_date(todo.merge(hist[["cik_int", "gvkey"]].drop_duplicates(), on="cik_int"), link)
        found.append(p2.assign(link_method="sec_analytics_cik"))
    except Exception as e:
        print(f"  (skipping historical CIK links: {type(e).__name__} — WRDS SEC Analytics may not be in your subscription)")

    # pass 3
    todo = camp[~camp["accession"].isin(pd.concat(found)["accession"])].copy()
    try:
        names = cached(cache, "crsp_stocknames", lambda: db.raw_sql(
            """select permno, comnam, namedt, nameenddt from crsp.stocknames
               where shrcd in (10, 11, 12)""", date_cols=["namedt", "nameenddt"]))
        names["key"] = names["comnam"].map(norm_name)
        todo["key"] = todo["subject_name"].map(norm_name)
        m = todo.merge(names, on="key")
        m = m[(m["filed_date"] >= m["namedt"]) & (m["filed_date"] <= m["nameenddt"])]
        ok = m.groupby("accession")["permno"].transform("nunique") == 1   # skip ambiguous names
        p3 = m[ok].drop_duplicates("accession")[["accession", "permno"]]
        found.append(p3.assign(link_method="crsp_name"))
    except Exception as e:
        print(f"  (skipping name match: {type(e).__name__})")

    links = pd.concat(found, ignore_index=True)
    out = camp.merge(links, on="accession", how="left")
    out["permno"] = out["permno"].astype("Int64")
    print("  link method:", out["link_method"].fillna("unmatched").value_counts().to_dict())
    return out


def pull_returns(db, permnos, start, end, cache: Path) -> pd.DataFrame:
    plist = ",".join(str(int(p)) for p in permnos)

    def v2():   # CRSP CIZ format: delisting returns already folded into dlyret
        return db.raw_sql(
            f"""select permno, dlycaldt as date, dlyret as ret from crsp.dsf_v2
                where permno in ({plist}) and dlycaldt between '{start:%Y-%m-%d}' and '{end:%Y-%m-%d}'""",
            date_cols=["date"])

    def legacy():   # older SIZ format (ends Dec 2024): add delisting returns manually
        r = db.raw_sql(
            f"""select permno, date, ret from crsp.dsf
                where permno in ({plist}) and date between '{start:%Y-%m-%d}' and '{end:%Y-%m-%d}'""",
            date_cols=["date"])
        d = db.raw_sql(f"select permno, dlstdt as date, dlret from crsp.dsedelist where permno in ({plist})",
                       date_cols=["date"])
        r = r.merge(d, on=["permno", "date"], how="outer")
        r["ret"] = (1 + r["ret"].fillna(0)) * (1 + r["dlret"].fillna(0)) - 1
        return r.drop(columns="dlret")

    def fetch():
        try:
            return v2()
        except Exception as e:
            print(f"  crsp.dsf_v2 unavailable ({type(e).__name__}); using legacy crsp.dsf (data ends 2024)")
            return legacy()

    # cache key depends on the permno list, so newly matched firms trigger a fresh pull
    key = hashlib.md5(plist.encode()).hexdigest()[:8]
    r = cached(cache, f"crsp_daily_{key}", fetch)
    r["permno"] = r["permno"].astype(int)
    return r


def pull_market(db, start, end, cache: Path) -> pd.Series:
    ff = cached(cache, "ff_daily", lambda: db.raw_sql(
        f"""select date, mktrf, rf from ff.factors_daily
            where date between '{start:%Y-%m-%d}' and '{end:%Y-%m-%d}'""", date_cols=["date"]))
    return (ff.set_index("date")["mktrf"] + ff.set_index("date")["rf"]).sort_index().rename("mkt")


# ---------------------------------------------------------------------------
# Event study math (pure pandas — no WRDS needed, so it's testable)
# ---------------------------------------------------------------------------

def event_position(row, cal: pd.DatetimeIndex):
    """Index in the trading calendar of the first day the market could react."""
    acc = row.get("acceptance_datetime")
    if isinstance(acc, str) and len(acc) >= 12 and acc[:12].isdigit():
        day, after_close = pd.Timestamp(acc[:8]), int(acc[8:12]) >= 1600
    else:
        day, after_close = row["filed_date"], False
    pos = cal.searchsorted(day)                 # first trading day on/after `day`
    if after_close and pos < len(cal) and cal[pos] == day:
        pos += 1
    return pos if pos < len(cal) else None


def campaign_windows(camp, rets, mkt):
    cal = mkt.index
    wide = rets.pivot_table(index="date", columns="permno", values="ret").reindex(cal)
    last_trade = rets.groupby("permno")["date"].max()

    results, ar_rows = [], []
    for _, row in camp.iterrows():
        rec = {"accession": row["accession"], "status": "ok"}
        p = row["permno"]
        if pd.isna(p) or int(p) not in wide.columns:
            rec["status"] = "no CRSP match"
            results.append(rec)
            continue
        p = int(p)
        e = event_position(row, cal)
        if e is None or e - PRE < 0:
            rec["status"] = "event outside market data"
            results.append(rec)
            continue

        lo, hi = e - PRE, min(e + POST, len(cal) - 1)
        rel = np.arange(lo, hi + 1) - e
        s = wide[p].iloc[lo:hi + 1].to_numpy()
        m = mkt.iloc[lo:hi + 1].to_numpy()
        days = cal[lo:hi + 1]
        rec["event_date"] = cal[e]

        def window(a, b):
            k = (rel >= a) & (rel <= b)
            return k, np.isfinite(s[k]).sum() == k.sum()

        # market-adjusted CARs (require complete data in the window)
        ar = s - m
        for name, (a, b) in {"car_m1_p1": (-1, 1), "car_m10_p20": (-10, 20)}.items():
            k, full = window(a, b)
            rec[name] = ar[k].sum() if full else np.nan

        # market-model CAR
        est = (rel >= EST_WINDOW[0]) & (rel <= EST_WINDOW[1]) & np.isfinite(s)
        if est.sum() >= MIN_EST_OBS:
            beta, alpha = np.polyfit(m[est], s[est], 1)
            k, full = window(-10, 20)
            rec["beta"] = beta
            rec["car_mm_m10_p20"] = (s[k] - (alpha + beta * m[k])).sum() if full else np.nan

        # one-year buy-and-hold abnormal return; after delisting, assume proceeds go into the market
        k = (rel >= 0) & (rel <= 252)
        if k.sum() == 253:
            sk = s[k].copy()
            delisted = last_trade[p] < days[k][-1]
            if delisted:
                gone = days[k] > last_trade[p]
                sk[gone] = m[k][gone]
            sk = np.where(np.isfinite(sk), sk, 0.0)
            rec["bhar_0_252"] = np.prod(1 + sk) - np.prod(1 + m[k])
            rec["delisted_within_1y"] = bool(delisted)

        # daily abnormal returns for the average-CAR chart
        k = (rel >= -20) & (rel <= 60)
        ar_rows.append(pd.Series(ar[k], index=rel[k], name=row["accession"]))
        results.append(rec)

    res = pd.DataFrame(results)
    ar_panel = pd.DataFrame(ar_rows) if ar_rows else pd.DataFrame()
    return res, ar_panel


def summarize(df, cols, by=None):
    def stats(g):
        out = {}
        for c in cols:
            x = g[c].dropna()
            n = len(x)
            out[(c, "n")] = n
            out[(c, "mean")] = x.mean()
            out[(c, "mean_wins")] = x.clip(*x.quantile([0.01, 0.99])).mean() if n else np.nan
            out[(c, "median")] = x.median()
            out[(c, "t")] = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            out[(c, "pct_pos")] = (x > 0).mean()
        return pd.Series(out)
    t = df.groupby(by).apply(stats) if by else stats(df).to_frame("all").T
    return t


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--campaigns", default="data/activist_campaigns.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default="cache_wrds")
    ap.add_argument("--exclude", default="data/exclusions.csv",
                    help="optional CSV with columns accession,reason for non-campaign 13Ds")
    args = ap.parse_args()

    import wrds
    cache, out = Path(args.cache), Path(args.out)
    cache.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    camp = load_campaigns(args.campaigns)
    if Path(args.exclude).exists():
        ex = pd.read_csv(args.exclude, dtype=str)
        camp = camp[~camp["accession"].isin(ex["accession"])]
        print(f"Excluded {len(ex)} non-campaign filings listed in {args.exclude}")
    start = camp["filed_date"].min() - pd.Timedelta(days=420)
    end = camp["filed_date"].max() + pd.Timedelta(days=420)

    db = wrds.Connection(wrds_username=args.wrds_user)
    print("Linking CIK -> GVKEY -> PERMNO...")
    camp = map_permnos(db, camp, cache)
    print(f"  matched {camp['permno'].notna().sum()} of {len(camp)} campaigns")

    print("Pulling CRSP daily returns and market returns...")
    rets = pull_returns(db, camp["permno"].dropna().unique(), start, end, cache)
    mkt = pull_market(db, start, end, cache)
    db.close()

    res, ar_panel = campaign_windows(camp, rets, mkt)
    full = camp.merge(res, on="accession", how="left")
    full.to_csv(out / "event_study_results.csv", index=False)

    cols = ["car_m1_p1", "car_m10_p20", "car_mm_m10_p20", "bhar_0_252"]
    cols = [c for c in cols if c in full.columns]
    overall = summarize(full, cols)
    by_act = summarize(full, cols, by="activist")
    overall.to_csv(out / "summary_overall.csv")
    by_act.to_csv(out / "summary_by_activist.csv")

    print("\nCoverage:")
    print(full["status"].value_counts().to_string())
    pd.set_option("display.width", 160)
    print("\nOverall (returns in decimals, t = cross-sectional t-stat):")
    print(overall.round(3).T.to_string())

    if not ar_panel.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cum = ar_panel.mean(axis=0, skipna=True).sort_index()
        car = cum.cumsum() * 100
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(car.index, car.values, lw=2)
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Trading days relative to 13D filing")
        ax.set_ylabel("Average cumulative abnormal return (%)")
        ax.set_title(f"Target stock reaction to activist 13D filings (n={len(ar_panel)})")
        fig.tight_layout()
        fig.savefig(out / "average_car.png", dpi=200)
        print(f"\nChart saved to {out / 'average_car.png'}")

    print(f"Results saved to {out}/event_study_results.csv and summary_*.csv")


if __name__ == "__main__":
    main()
