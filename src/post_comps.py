#!/usr/bin/env python3
"""
Post Holdings — comparable-company valuation (Compustat via WRDS).

For Post and peer groups matching each of its segments, pulls:
  * current market value (latest Compustat daily price x shares outstanding)
  * net debt from the latest quarterly balance sheet
  * trailing-twelve-month (TTM) sales and EBITDA from the last 4 quarters
and computes EV/EBITDA, EV/Sales, EBITDA margin and TTM sales growth.

If data/post_segments.csv is filled in, it also runs a sum-of-the-parts:
each segment's EBITDA x its peer group's median multiple, minus Post's net
debt, divided by shares -> implied value per share vs. today's price.

Usage
  python3 post_comps.py --wrds-user YOUR_WRDS_USERNAME
Outputs
  data/post_comps.csv, data/post_comps_summary.csv, data/post_sotp.csv (if segments file filled)

Notes
  * Compustat EBITDA (oibdpq) is "as reported", not Post's "Adjusted EBITDA",
    so Post's own multiple here will differ from what management quotes.
  * Tickers that were acquired/delisted (e.g. Kellanova, WK Kellogg) won't
    have current prices and are reported as missing; their takeover prices
    belong in a separate precedent-transactions table.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "POST"

# Edit freely: peer group -> tickers. Each group should resemble one Post segment.
PEERS = {
    "Cereal & center-store food (Post Consumer Brands)": ["GIS", "KHC", "CAG", "CPB", "SJM", "BGS", "HAIN", "K", "KLG", "THS"],
    "Pet food (inside Post Consumer Brands)":            ["FRPT", "CENT", "SPB"],
    "Eggs & potatoes (Foodservice)":                     ["CALM", "LW", "VITL"],
    "Refrigerated sides (Refrigerated Retail)":          ["HRL", "LANC", "CAG"],
}


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn()
    df.to_parquet(f)
    return df


def plain(df: pd.DataFrame) -> pd.DataFrame:
    """Convert WRDS nullable numeric columns to ordinary floats."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def pull(db, tickers, cache: Path):
    tl = ",".join(f"'{t}'" for t in sorted(set(tickers)))
    key = hashlib.md5(tl.encode()).hexdigest()[:8]
    px = cached(cache, f"comps_secd_{key}", lambda: db.raw_sql(
        f"""select gvkey, iid, tic, datadate, prccd, cshoc from comp.secd
            where tic in ({tl}) and datadate >= current_date - interval '45 days'""",
        date_cols=["datadate"]))
    px = plain(px).dropna(subset=["prccd", "cshoc"])
    px = px.sort_values("datadate").groupby("tic").tail(1)            # latest trading day per ticker

    gl = ",".join(f"'{g}'" for g in px["gvkey"].unique())
    q = cached(cache, f"comps_fundq_{key}", lambda: db.raw_sql(
        f"""select gvkey, datadate, fyearq, fqtr, saleq, oibdpq, dlttq, dlcq, cheq, mibq, curcdq
            from comp.fundq
            where gvkey in ({gl}) and indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
              and datadate >= current_date - interval '3 years'""", date_cols=["datadate"]))
    q = plain(q).sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    return px, q


def company_metrics(px: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in px.iterrows():
        g = q[(q["gvkey"] == p["gvkey"]) & q["saleq"].notna()].sort_values("datadate")
        if len(g) < 4:
            continue
        last4, prev4 = g.tail(4), g.iloc[-8:-4] if len(g) >= 8 else None
        bs = g.iloc[-1]
        mktcap = p["prccd"] * p["cshoc"] / 1e6                         # $ millions (cshoc is a share count)
        net_debt = np.nansum([bs["dlttq"], bs["dlcq"], bs["mibq"]]) - (bs["cheq"] if pd.notna(bs["cheq"]) else 0)
        ev = mktcap + net_debt
        sales, ebitda = last4["saleq"].sum(), last4["oibdpq"].sum(min_count=4)
        rows.append({
            "ticker": p["tic"], "gvkey": p["gvkey"], "price": p["prccd"], "price_date": p["datadate"].date(),
            "latest_quarter": bs["datadate"].date(), "currency": bs["curcdq"],
            "mktcap_m": mktcap, "net_debt_m": net_debt, "ev_m": ev,
            "ttm_sales_m": sales, "ttm_ebitda_m": ebitda,
            "ev_ebitda": ev / ebitda if ebitda and ebitda > 0 else np.nan,
            "ev_sales": ev / sales if sales > 0 else np.nan,
            "ebitda_margin": ebitda / sales if sales > 0 else np.nan,
            "sales_growth": sales / prev4["saleq"].sum() - 1 if prev4 is not None and prev4["saleq"].sum() > 0 else np.nan,
            "net_debt_to_ebitda": net_debt / ebitda if ebitda and ebitda > 0 else np.nan,
            "shares_m": p["cshoc"] / 1e6,
        })
    return pd.DataFrame(rows)


def group_summary(m: pd.DataFrame) -> pd.DataFrame:
    out = []
    for grp, tics in PEERS.items():
        g = m[m["ticker"].isin(tics)]
        if g.empty:
            continue
        e = g["ev_ebitda"].dropna()
        out.append({"peer_group": grp, "n": len(g), "tickers": ", ".join(g["ticker"]),
                    "ev_ebitda_median": e.median(), "ev_ebitda_p25": e.quantile(.25), "ev_ebitda_p75": e.quantile(.75),
                    "ev_sales_median": g["ev_sales"].median(), "ebitda_margin_median": g["ebitda_margin"].median(),
                    "sales_growth_median": g["sales_growth"].median()})
    return pd.DataFrame(out)


def sum_of_parts(seg: pd.DataFrame, summary: pd.DataFrame, post: pd.Series) -> pd.DataFrame:
    s = seg.merge(summary[["peer_group", "ev_ebitda_p25", "ev_ebitda_median", "ev_ebitda_p75"]],
                  on="peer_group", how="left")
    # corporate costs have no peer group: capitalize at the EBITDA-weighted average of the segment multiples
    w = s["ebitda_m"].clip(lower=0)
    for c in ["ev_ebitda_p25", "ev_ebitda_median", "ev_ebitda_p75"]:
        s[c] = s[c].fillna((s[c] * w).sum() / w[s[c].notna()].sum())
    for case, col in [("low", "ev_ebitda_p25"), ("mid", "ev_ebitda_median"), ("high", "ev_ebitda_p75")]:
        s[f"ev_{case}_m"] = s["ebitda_m"] * s[col]
    tot = s[["ev_low_m", "ev_mid_m", "ev_high_m"]].sum()
    per_share = (tot - post["net_debt_m"]) / post["shares_m"]
    print("\nSum of the parts (segment EBITDA x peer multiples, minus Post's net debt):")
    print(s[["segment", "peer_group", "ebitda_m", "ev_ebitda_median", "ev_mid_m"]].round(1).to_string(index=False))
    for case in ["low", "mid", "high"]:
        v = per_share[f"ev_{case}_m"]
        print(f"  {case:>4} case: implied ${v:,.0f}/share vs. ${post['price']:,.2f} today "
              f"({v / post['price'] - 1:+.0%})")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--segments", default="data/post_segments.csv")
    ap.add_argument("--cache", default="cache_wrds")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    cache.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    import wrds
    tickers = [TARGET] + [t for v in PEERS.values() for t in v]
    db = wrds.Connection(wrds_username=args.wrds_user)
    px, q = pull(db, tickers, cache)
    db.close()

    m = company_metrics(px, q)
    missing = sorted(set(tickers) - set(m["ticker"]))
    if missing:
        print(f"No current data (likely acquired/delisted or ticker changed): {', '.join(missing)}")
    m.to_csv(out / "post_comps.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["ticker", "mktcap_m", "ev_m", "ttm_ebitda_m", "ev_ebitda", "ev_sales", "ebitda_margin",
            "sales_growth", "net_debt_to_ebitda", "latest_quarter"]
    print("\nCompany metrics ($ millions; TTM = last four quarters):")
    print(m[cols].sort_values("ev_ebitda").round(2).to_string(index=False))

    summary = group_summary(m)
    summary.to_csv(out / "post_comps_summary.csv", index=False)
    print("\nPeer group medians:")
    print(summary.drop(columns="tickers").round(2).to_string(index=False))

    if TARGET not in set(m["ticker"]):
        print(f"\n{TARGET} not found — check the ticker or Compustat coverage.")
        return
    post = m.set_index("ticker").loc[TARGET]
    print(f"\nPost today: price ${post['price']:.2f} ({post['price_date']}), market cap ${post['mktcap_m']:,.0f}M, "
          f"net debt ${post['net_debt_m']:,.0f}M, EV/EBITDA {post['ev_ebitda']:.1f}x, "
          f"net debt/EBITDA {post['net_debt_to_ebitda']:.1f}x")

    seg_path = Path(args.segments)
    if not seg_path.exists():
        pd.DataFrame({
            "segment": ["Post Consumer Brands - cereal & other", "Post Consumer Brands - pet food",
                        "Foodservice", "Refrigerated Retail", "Weetabix", "Corporate (unallocated)"],
            "peer_group": [list(PEERS)[0], list(PEERS)[1], list(PEERS)[2], list(PEERS)[3], list(PEERS)[0], ""],
            "ebitda_m": [np.nan] * 6,
        }).to_csv(seg_path, index=False)
        print(f"\nCreated {seg_path}: fill in each segment's TTM adjusted EBITDA ($M, corporate as a negative "
              "number) from Post's earnings releases/10-K, then rerun for the sum-of-the-parts.")
        return
    seg = pd.read_csv(seg_path)
    if seg["ebitda_m"].isna().any():
        print(f"\n{seg_path} has blank EBITDA values — fill them in and rerun for the sum-of-the-parts.")
        return
    sum_of_parts(seg, summary, post).to_csv(out / "post_sotp.csv", index=False)


if __name__ == "__main__":
    main()
