#!/usr/bin/env python3
"""
Post Holdings — has the market paid more for Post when its debt was lower?

From Compustat quarterly data (comp.fundq) for Post and large packaged-food
peers, each calendar quarter since 2013:
  EV           = quarter-end price x shares + debt + noncontrolling interest - cash
  TTM EBITDA   = sum of the last four quarters of oibdpq (as-reported EBITDA)
  EV/EBITDA    and   net debt / EBITDA

Outputs
  data/post_history.csv         Post and peer-median metrics by quarter
  data/post_history.png         chart: Post vs peer multiple, and Post leverage
  data/post_multiple_vs_leverage.png   scatter with a fitted line
and prints how Post's multiple and its discount to peers changed with leverage.

Usage
  python3 post_history.py --wrds-user YOUR_WRDS_USERNAME
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "POST"
# Large U.S. packaged-food peers. K (Kellanova) and KLG (WK Kellogg) are kept for
# the years they traded; Compustat history survives the acquisitions.
PEERS = ["GIS", "KHC", "CAG", "CPB", "SJM", "K", "KLG", "HRL"]
START = "2012-01-01"


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn()
    df.to_parquet(f)
    return df


NUMERIC = ["saleq", "oibdpq", "dlttq", "dlcq", "cheq", "mibq", "prccq", "cshoq"]


def plain(df):
    """Force numeric columns to ordinary floats (WRDS may return nullable or all-empty object columns)."""
    df = df.copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["tic"] = df["tic"].astype(str).str.strip()
    return df


def pull(db, cache):
    tl = ",".join(f"'{t}'" for t in [TARGET] + PEERS)
    key = hashlib.md5(tl.encode()).hexdigest()[:8]
    return cached(cache, f"history_fundq_{key}", lambda: db.raw_sql(
        f"""select gvkey, tic, conm, datadate, saleq, oibdpq, dlttq, dlcq, cheq, mibq, prccq, cshoq
            from comp.fundq
            where tic in ({tl}) and indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D'
              and consol = 'C' and datadate >= '{START}'""", date_cols=["datadate"]))


def quarterly_metrics(q: pd.DataFrame) -> pd.DataFrame:
    q = plain(q).sort_values(["tic", "datadate"]).drop_duplicates(["tic", "datadate"], keep="last")
    q["ttm_ebitda"] = q.groupby("tic")["oibdpq"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    q["ttm_sales"] = q.groupby("tic")["saleq"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    z = lambda s: s.fillna(0)
    q["net_debt"] = z(q["dlttq"]) + z(q["dlcq"]) + z(q["mibq"]) - z(q["cheq"])
    q["mktcap"] = q["prccq"] * q["cshoq"]
    q["ev"] = q["mktcap"] + q["net_debt"]
    ok = q["ttm_ebitda"] > 0
    q["ev_ebitda"] = np.where(ok, q["ev"] / q["ttm_ebitda"], np.nan)
    q["leverage"] = np.where(ok, q["net_debt"] / q["ttm_ebitda"], np.nan)
    q["quarter"] = q["datadate"].dt.to_period("Q")       # align different fiscal calendars
    q = q.sort_values("datadate").drop_duplicates(["tic", "quarter"], keep="last")
    # drop clearly broken rows (missing price/shares or absurd multiples)
    q.loc[(q["ev_ebitda"] <= 0) | (q["ev_ebitda"] > 40), "ev_ebitda"] = np.nan
    return q


def build_history(q: pd.DataFrame) -> pd.DataFrame:
    post = q[q["tic"] == TARGET].set_index("quarter")[["datadate", "mktcap", "net_debt", "ttm_ebitda",
                                                       "ev_ebitda", "leverage"]]
    peers = q[q["tic"].isin(PEERS)].groupby("quarter").agg(
        peer_ev_ebitda=("ev_ebitda", "median"), peer_leverage=("leverage", "median"),
        n_peers=("ev_ebitda", "count"))
    h = post.join(peers, how="left").dropna(subset=["ev_ebitda"])
    h["discount_to_peers"] = h["ev_ebitda"] - h["peer_ev_ebitda"]
    return h


def summarize(h: pd.DataFrame):
    pd.set_option("display.width", 160)
    yearly = h.groupby(h.index.year)[["ev_ebitda", "peer_ev_ebitda", "discount_to_peers", "leverage"]].mean()
    print("\nYearly averages (Compustat as-reported EBITDA basis):")
    yearly.index.name = "year"
    print(yearly.round(2).rename(columns={"ev_ebitda": "Post EV/EBITDA", "peer_ev_ebitda": "peer median",
                                          "discount_to_peers": "Post minus peers",
                                          "leverage": "Post net debt/EBITDA"}).to_string())

    h = h.dropna(subset=["leverage", "ev_ebitda"])
    x, y, d = h["leverage"], h["ev_ebitda"], h["discount_to_peers"]
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\nAcross {len(h)} quarters: correlation of Post's multiple with its leverage = {x.corr(y):+.2f}; "
          f"with its discount to peers = {x.corr(d):+.2f}")
    print(f"Fitted line: EV/EBITDA = {intercept:.2f} {slope:+.2f} x leverage "
          f"-> at 3.5x leverage: {intercept + slope * 3.5:.1f}x; at 5.0x: {intercept + slope * 5.0:.1f}x")

    for lo, hi, lab in [(0, 4.5, "below 4.5x"), (4.5, 5.25, "4.5x-5.25x"), (5.25, 99, "above 5.25x")]:
        s = h[(x >= lo) & (x < hi)]
        if len(s):
            print(f"  Quarters with leverage {lab:>12}: {len(s):3d} quarters, Post avg {s['ev_ebitda'].mean():.1f}x, "
                  f"peers avg {s['peer_ev_ebitda'].mean():.1f}x, Post minus peers {s['discount_to_peers'].mean():+.1f}x")
    latest = h.iloc[-1]
    print(f"\nLatest quarter ({h.index[-1]}): Post {latest['ev_ebitda']:.1f}x, peers {latest['peer_ev_ebitda']:.1f}x, "
          f"leverage {latest['leverage']:.1f}x")
    return slope, intercept


def charts(h: pd.DataFrame, slope, intercept, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = h.index.to_timestamp()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(t, h["ev_ebitda"], lw=2, label="Post")
    a1.plot(t, h["peer_ev_ebitda"], lw=2, ls="--", label="Packaged-food peer median")
    a1.set_ylabel("EV / TTM EBITDA (x)")
    a1.set_title("Post Holdings: valuation multiple vs. peers, and leverage")
    a1.legend(frameon=False)
    a2.plot(t, h["leverage"], lw=2, color="tab:red")
    a2.set_ylabel("Net debt / EBITDA (x)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.text(0.01, 0.005, "Source: Compustat quarterly (as-reported EBITDA). Includes BellRing while consolidated (FY2019-22).",
             fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "post_history.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(h["leverage"], h["ev_ebitda"], alpha=0.7)
    xs = np.linspace(h["leverage"].min(), h["leverage"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="black", lw=1)
    last = h.iloc[-1]
    ax.scatter([last["leverage"]], [last["ev_ebitda"]], color="tab:red", s=80, zorder=3, label="Latest quarter")
    ax.set_xlabel("Net debt / EBITDA (x)")
    ax.set_ylabel("EV / TTM EBITDA (x)")
    ax.set_title("Post: multiple vs. leverage, quarterly since 2013")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "post_multiple_vs_leverage.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--cache", default="cache_wrds")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    cache.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    import wrds
    db = wrds.Connection(wrds_username=args.wrds_user)
    raw = pull(db, cache)
    db.close()

    q = quarterly_metrics(raw)
    found = sorted(q["tic"].unique())
    print("Companies found:", ", ".join(found))
    h = build_history(q)
    h.to_csv(out / "post_history.csv")
    slope, intercept = summarize(h)
    charts(h, slope, intercept, out)
    print(f"\nSaved post_history.csv, post_history.png and post_multiple_vs_leverage.png to {out}/")


if __name__ == "__main__":
    main()
