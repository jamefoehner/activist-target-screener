#!/usr/bin/env python3
"""
Gradual information diffusion — does activist news reach prices more slowly
when fewer investors are paying attention?

For each campaign in data/event_study_results.csv (from event_study.py):

  Reaction, market-adjusted (stock return minus market return):
    car_pre    days -10..-1    before the filing
    car_fast   days  0..+1     immediate reaction
    car_drift  days +2..+20    slow follow-through
    speed      car_fast / (car_fast + car_drift), for moves of at least 2%

  Attention proxies, measured before the filing:
    log_size        log market value, month before the filing (CRSP)
    analysts        number of analysts with an EPS estimate (I/B/E/S), 0 if none
    friday          event day falls on a Friday
    after_close     13D accepted after the 4pm close
    busy_day        number of Compustat earnings announcements on the event day
    price_delay     Hou-Moskowitz delay: how slowly the stock normally responds to
                    market moves (weekly returns, prior year); 0 = instant, 1 = slow

Then: drift by low/medium/high tercile of each proxy, a regression of drift on all
proxies (robust standard errors), and a chart of the average reaction for low- vs.
high-attention targets.

Usage
  python3 diffusion_study.py --wrds-user YOUR_WRDS_USERNAME
Outputs
  data/diffusion.csv, data/diffusion_terciles.csv, data/diffusion_regression.csv,
  data/diffusion_paths.png
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn()
    df.to_parquet(f)
    return df


def floats(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


# ---------------------------------------------------------------------------
# WRDS pulls
# ---------------------------------------------------------------------------

def pull(db, camp, cache: Path):
    plist = ",".join(str(int(p)) for p in sorted(camp["permno"].unique()))
    key = hashlib.md5(plist.encode()).hexdigest()[:8]
    start = (camp["event_date"].min() - pd.Timedelta(days=420)).date()
    end = (camp["event_date"].max() + pd.Timedelta(days=60)).date()

    def daily():
        try:
            return db.raw_sql(f"""select permno, dlycaldt as date, dlyret as ret from crsp.dsf_v2
                                  where permno in ({plist}) and dlycaldt between '{start}' and '{end}'""",
                              date_cols=["date"])
        except Exception:
            return db.raw_sql(f"""select permno, date, ret from crsp.dsf
                                  where permno in ({plist}) and date between '{start}' and '{end}'""",
                              date_cols=["date"])
    dsf = cached(cache, f"diff_daily_{key}", daily)
    ff = cached(cache, "diff_ff_daily", lambda: db.raw_sql(
        f"select date, mktrf, rf from ff.factors_daily where date between '{start}' and '{end}'",
        date_cols=["date"]))

    def caps():
        try:
            return db.raw_sql(f"""select permno, mthcaldt as date, mthcap as cap from crsp.msf_v2
                                  where permno in ({plist})""", date_cols=["date"])
        except Exception:
            c = db.raw_sql(f"select permno, date, prc, shrout from crsp.msf where permno in ({plist})",
                           date_cols=["date"])
            c["cap"] = c["prc"].abs() * c["shrout"]
            return c[["permno", "date", "cap"]]
    cap = cached(cache, f"diff_caps_{key}", caps)

    rdq = cached(cache, "diff_rdq_counts", lambda: db.raw_sql(
        f"""select rdq, count(*) as n from comp.fundq
            where rdq between '{start}' and '{end}' group by rdq""", date_cols=["rdq"]))

    try:
        ibes = cached(cache, f"diff_ibes_{key}", lambda: db.raw_sql(
            f"""select a.permno, b.statpers, b.numest
                from wrdsapps.ibcrsphist a
                join ibes.statsum_epsus b on a.ticker = b.ticker
                where b.fpi = '1' and a.score <= 2 and a.permno in ({plist})
                  and b.statpers between a.sdate and a.edate""", date_cols=["statpers"]))
    except Exception as e:
        print(f"  (I/B/E/S unavailable: {type(e).__name__} — skipping analyst coverage)")
        ibes = None
    return dsf, ff, cap, rdq, ibes


# ---------------------------------------------------------------------------
# Measures (pure pandas / numpy, testable)
# ---------------------------------------------------------------------------

def reaction_measures(camp, dsf, ff):
    d = floats(dsf, ["permno", "ret"]).dropna(subset=["permno"])
    d["permno"] = d["permno"].astype(int)
    ff = floats(ff, ["mktrf", "rf"])
    mkt = (ff.set_index("date")["mktrf"] + ff.set_index("date")["rf"]).sort_index()
    cal = mkt.index
    wide = d.pivot_table(index="date", columns="permno", values="ret", aggfunc="last").reindex(cal)

    rows, paths = [], {}
    for _, r in camp.iterrows():
        p, e = int(r["permno"]), cal.searchsorted(r["event_date"])
        rec = {"accession": r["accession"]}
        if p not in wide.columns or e - 260 < 0 or e + 20 >= len(cal):
            rows.append(rec)
            continue
        s = wide[p].to_numpy()
        m = mkt.to_numpy()
        ar = s - m

        def car(a, b):
            seg = ar[e + a: e + b + 1]
            return float(np.sum(seg)) if np.isfinite(seg).sum() >= len(seg) - 1 else np.nan

        rec.update(car_pre=car(-10, -1), car_fast=car(0, 1), car_drift=car(2, 20))
        tot = rec["car_fast"] + rec["car_drift"] if pd.notna(rec["car_fast"]) and pd.notna(rec["car_drift"]) else np.nan
        rec["car_total"] = tot
        rec["speed"] = rec["car_fast"] / tot if pd.notna(tot) and abs(tot) >= 0.02 else np.nan
        rec["price_delay"] = price_delay(s[e - 260: e - 10], m[e - 260: e - 10])
        rec["event_weekday"] = cal[e].dayofweek
        rec["event_day"] = cal[e]
        rows.append(rec)
        paths[r["accession"]] = pd.Series(np.nancumsum(np.nan_to_num(ar[e - 10: e + 21])), index=range(-10, 21))
    return pd.DataFrame(rows), paths


def price_delay(stock_daily, mkt_daily, lags=4):
    """Hou & Moskowitz (2005): 1 - R2(contemporaneous market only) / R2(with 4 weekly lags)."""
    ok = np.isfinite(stock_daily) & np.isfinite(mkt_daily)
    if ok.sum() < 150:
        return np.nan
    s, m = np.where(ok, stock_daily, 0.0), np.where(ok, mkt_daily, 0.0)
    n = len(s) // 5
    sw = np.array([np.prod(1 + s[i * 5:(i + 1) * 5]) - 1 for i in range(n)])
    mw = np.array([np.prod(1 + m[i * 5:(i + 1) * 5]) - 1 for i in range(n)])
    y = sw[lags:]
    X_full = np.column_stack([np.ones(len(y))] + [mw[lags - k: n - k] for k in range(lags + 1)])
    X_res = X_full[:, :2]

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    full, res = r2(X_full), r2(X_res)
    if not np.isfinite(full) or full <= 0:
        return np.nan
    return float(np.clip(1 - res / full, 0, 1))


def attention_measures(df, camp, cap, rdq, ibes):
    df = df.merge(camp[["accession", "permno", "after_market_close"]], on="accession", how="left")
    cap = floats(cap, ["permno", "cap"]).dropna()
    cap["permno"] = cap["permno"].astype(int)
    cap["month"] = cap["date"].dt.to_period("M")
    capm = cap.groupby(["permno", "month"])["cap"].last()
    df["log_size"] = [np.log(capm.get((int(p), d.to_period("M") - 1), np.nan) / 1000)
                      if pd.notna(d) else np.nan for p, d in zip(df["permno"], df["event_day"])]

    rdq = floats(rdq, ["n"]).set_index("rdq")["n"]
    df["busy_day"] = [np.log1p(rdq.get(d, 0)) if pd.notna(d) else np.nan for d in df["event_day"]]
    df["friday"] = (df["event_weekday"] == 4).astype(float).where(df["event_weekday"].notna())
    df["after_close"] = df["after_market_close"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    if ibes is not None and len(ibes):
        ib = floats(ibes, ["permno", "numest"]).dropna(subset=["permno"])
        ib["permno"] = ib["permno"].astype(int)
        vals = []
        for p, d in zip(df["permno"], df["event_day"]):
            if pd.isna(d):
                vals.append(np.nan); continue
            g = ib[(ib["permno"] == int(p)) & (ib["statpers"] <= d) & (ib["statpers"] >= d - pd.Timedelta(days=120))]
            vals.append(float(g.sort_values("statpers")["numest"].iloc[-1]) if len(g) else 0.0)
        df["analysts"] = vals
        df["log_analysts"] = np.log1p(df["analysts"])
    return df


def ols_robust(y, X, names):
    """OLS with heteroskedasticity-robust (HC1) standard errors."""
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[keep], X[keep]
    X = np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    V = XtX_inv @ (X.T * u ** 2) @ X @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=["const"] + names), n, r2


def terciles(df, proxies):
    rows = []
    for p in proxies:
        x = df[p]
        if x.dropna().nunique() <= 2:                       # binary proxies: no vs. yes
            groups = {f"{p}=0": df[x == 0], f"{p}=1": df[x == 1]}
        else:
            q = pd.qcut(x, 3, labels=["low", "mid", "high"], duplicates="drop")
            groups = {f"{p} {lab}": df[q == lab] for lab in ["low", "mid", "high"]}
        for name, g in groups.items():
            d = g["car_drift"].dropna()
            rows.append({"group": name, "n": len(d), "drift_mean": d.mean(), "drift_median": d.median(),
                         "drift_t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                         "fast_mean": g["car_fast"].mean(), "speed_median": g["speed"].median()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--results", default="data/event_study_results.csv")
    ap.add_argument("--cache", default="cache_wrds")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    cache.mkdir(exist_ok=True)

    camp = pd.read_csv(args.results, parse_dates=["event_date"], dtype={"accession": str})
    camp = camp[(camp["status"] == "ok") & camp["permno"].notna() & camp["event_date"].notna()].copy()
    camp["permno"] = camp["permno"].astype(int)
    print(f"{len(camp)} campaigns with price data")

    import wrds
    db = wrds.Connection(wrds_username=args.wrds_user)
    dsf, ff, cap, rdq, ibes = pull(db, camp, cache)
    db.close()

    df, paths = reaction_measures(camp, dsf, ff)
    df = attention_measures(df, camp, cap, rdq, ibes)
    df.to_csv(out / "diffusion.csv", index=False)

    proxies = [p for p in ["log_size", "log_analysts", "friday", "after_close", "busy_day", "price_delay"]
               if p in df.columns and df[p].notna().sum() > 30]
    pd.set_option("display.width", 160)
    print("\nAverage reaction (decimals):",
          {k: round(df[k].mean(), 4) for k in ["car_pre", "car_fast", "car_drift"]},
          f"| median share of the move on days 0-1: {df['speed'].median():.2f}")

    t = terciles(df, proxies)
    t.to_csv(out / "diffusion_terciles.csv", index=False)
    print("\nDrift (days +2..+20) by attention group — theory predicts MORE drift when attention is LOW:")
    print(t.round(4).to_string(index=False))

    controls = proxies + ["car_fast"]
    reg, n, r2 = ols_robust(df["car_drift"].to_numpy(float), df[controls].to_numpy(float), controls)
    reg.to_csv(out / "diffusion_regression.csv")
    print(f"\nRegression of drift on attention proxies (robust SEs), n = {n}, R2 = {r2:.3f}:")
    print(reg.round(4).to_string())
    print("Theory: negative on log_size and log_analysts; positive on friday, busy_day, price_delay.")

    # chart: average cumulative abnormal return, small vs. large targets (or fewest vs. most analysts)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    split = "log_analysts" if "log_analysts" in proxies else "log_size"
    med = df[split].median()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, mask, style in [("Low attention", df[split] <= med, "-"), ("High attention", df[split] > med, "--")]:
        ids = df.loc[mask, "accession"]
        ps = [paths[a] for a in ids if a in paths]
        if ps:
            avg = pd.concat(ps, axis=1).mean(axis=1) * 100
            ax.plot(avg.index, avg.values, style, lw=2, label=f"{lab} (n={len(ps)})")
    ax.axvline(0, color="grey", ls=":", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Trading days relative to 13D filing")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title(f"Reaction to activist 13Ds by investor attention (split on {split.replace('log_', '')})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "diffusion_paths.png", dpi=200)
    print(f"\nSaved diffusion.csv, diffusion_terciles.csv, diffusion_regression.csv and diffusion_paths.png to {out}/")


if __name__ == "__main__":
    main()
