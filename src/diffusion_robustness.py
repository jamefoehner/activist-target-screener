#!/usr/bin/env python3
"""
Diffusion study, part 2: robustness checks and a trading test.

Reads data/diffusion.csv (from diffusion_study.py) and adds:

Robustness
  1. Trimmed:   drift winsorized at the 5th/95th percentiles, then the same regression
  2. Median:    median (least-absolute-deviation) regression, bootstrap standard errors
  3. Liquidity: adds the Amihud illiquidity ratio (average |return| / dollar volume over
                the prior year) to check that "slow diffusion" isn't just thin trading
  4. Multiple testing: Holm adjustment across the attention proxies

Trading test
  Combines the proxies into one attention index (small size, few analysts, high price
  delay, busy day = low attention), buys targets at the close of day +1, sells at day
  +20, and reports the abnormal return by tercile of the index, before and after
  round-trip trading costs.

Usage
  python3 diffusion_robustness.py --wrds-user YOUR_WRDS_USERNAME
Outputs
  data/diffusion_robustness.csv, data/diffusion_trading.csv
"""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROXIES = ["log_size", "log_analysts", "friday", "after_close", "busy_day", "price_delay"]
COSTS = [0.0, 0.005, 0.01, 0.02]       # round-trip trading costs to test


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn()
    df.to_parquet(f)
    return df


# ---------------------------------------------------------------------------
# Statistics (numpy only)
# ---------------------------------------------------------------------------

def design(df, cols, y="car_drift"):
    d = df[[y] + cols].dropna()
    return d[y].to_numpy(float), np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols]), len(d)


def ols_hc1(y, X):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    n, k = X.shape
    V = XtX_inv @ (X.T * u ** 2) @ X @ XtX_inv * n / (n - k)
    return b, np.sqrt(np.diag(V))


def lad(y, X, iters=200, eps=1e-6):
    """Median regression by iteratively reweighted least squares."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        w = 1 / np.maximum(np.abs(y - X @ b), eps)
        b_new = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        if np.max(np.abs(b_new - b)) < 1e-9:
            break
        b = b_new
    return b


def lad_bootstrap(y, X, reps=500, seed=0):
    rng = np.random.default_rng(seed)
    b = lad(y, X)
    draws = [lad(y[idx], X[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    return b, np.std(draws, axis=0, ddof=1)


def p_two_sided(t):
    return math.erfc(abs(t) / math.sqrt(2))


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def amihud(ret, prc, vol):
    """Average of |return| / dollar volume, in returns per $ million traded."""
    dv = np.abs(prc) * vol
    ok = np.isfinite(ret) & np.isfinite(dv) & (dv > 0)
    if ok.sum() < 100:
        return np.nan
    return float(np.mean(np.abs(ret[ok]) / dv[ok]) * 1e6)


def attention_index(df):
    """Higher = LESS attention. Average of z-scores, signed so each points toward inattention."""
    signs = {"log_size": -1, "log_analysts": -1, "price_delay": 1, "busy_day": 1}
    z = []
    for c, s in signs.items():
        if c in df and df[c].notna().sum() > 30:
            x = df[c]
            z.append(s * (x - x.mean()) / x.std())
    return pd.concat(z, axis=1).mean(axis=1, skipna=True)


# ---------------------------------------------------------------------------

def pull_amihud(db, df, cache: Path):
    plist = ",".join(str(int(p)) for p in sorted(df["permno"].dropna().unique()))
    key = hashlib.md5(plist.encode()).hexdigest()[:8]
    start = (df["event_day"].min() - pd.Timedelta(days=420)).date()
    end = df["event_day"].max().date()

    def fetch():
        try:
            return db.raw_sql(f"""select permno, dlycaldt as date, dlyret as ret, dlyprc as prc, dlyvol as vol
                                  from crsp.dsf_v2 where permno in ({plist})
                                  and dlycaldt between '{start}' and '{end}'""", date_cols=["date"])
        except Exception:
            return db.raw_sql(f"""select permno, date, ret, prc, vol from crsp.dsf
                                  where permno in ({plist}) and date between '{start}' and '{end}'""",
                              date_cols=["date"])
    d = cached(cache, f"diff_vol_{key}", fetch)
    for c in ["permno", "ret", "prc", "vol"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d = d.dropna(subset=["permno"])
    d["permno"] = d["permno"].astype(int)
    out = []
    for _, r in df.iterrows():
        if pd.isna(r["permno"]) or pd.isna(r["event_day"]):
            out.append(np.nan); continue
        g = d[(d["permno"] == int(r["permno"])) & (d["date"] < r["event_day"] - pd.Timedelta(days=14)) &
              (d["date"] >= r["event_day"] - pd.Timedelta(days=380))]
        out.append(amihud(g["ret"].to_numpy(), g["prc"].to_numpy(), g["vol"].to_numpy()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--diffusion", default="data/diffusion.csv")
    ap.add_argument("--cache", default="cache_wrds")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)

    df = pd.read_csv(args.diffusion, parse_dates=["event_day"])
    proxies = [p for p in PROXIES if p in df.columns and df[p].notna().sum() > 30]
    print(f"{df['car_drift'].notna().sum()} campaigns with drift; proxies: {', '.join(proxies)}")

    try:
        import wrds
        db = wrds.Connection(wrds_username=args.wrds_user)
        df["amihud"] = pull_amihud(db, df, cache)
        db.close()
        df["log_amihud"] = np.log(df["amihud"].where(df["amihud"] > 0))
    except Exception as e:
        print(f"  (couldn't compute illiquidity: {type(e).__name__}; skipping that check)")

    pd.set_option("display.width", 160)
    rows = []
    lo, hi = df["car_drift"].quantile([0.05, 0.95])
    variants = [("Baseline OLS", "car_drift", proxies, "ols"),
                ("Trimmed 5/95 OLS", "drift_w", proxies, "ols"),
                ("Median regression", "car_drift", proxies, "lad")]
    df["drift_w"] = df["car_drift"].clip(lo, hi)
    if "log_amihud" in df and df["log_amihud"].notna().sum() > 30:
        variants.append(("OLS + illiquidity control", "car_drift", proxies + ["log_amihud"], "ols"))

    for name, y, cols, how in variants:
        yv, X, n = design(df, cols, y)
        b, se = ols_hc1(yv, X) if how == "ols" else lad_bootstrap(yv, X)
        t = b / se
        p = np.array([p_two_sided(v) for v in t])
        p_holm = np.full(len(p), np.nan)
        idx = [i for i, c in enumerate(["const"] + cols) if c in proxies]
        p_holm[idx] = holm(p[idx])
        for i, c in enumerate(["const"] + cols):
            rows.append({"model": name, "n": n, "variable": c, "coef": b[i], "t": t[i],
                         "p": p[i], "p_holm": p_holm[i]})
    rob = pd.DataFrame(rows)
    rob.to_csv(out / "diffusion_robustness.csv", index=False)

    table = rob[rob["variable"].isin(proxies + ["log_amihud"])].pivot_table(
        index="variable", columns="model", values="t", sort=False)
    print("\nt-statistics on each proxy across specifications (|t| > 2 is roughly significant):")
    print(table.round(2).to_string())
    base = rob[(rob["model"] == "Baseline OLS") & rob["variable"].isin(proxies)][["variable", "coef", "p", "p_holm"]]
    print("\nBaseline p-values, raw vs. Holm-adjusted for testing", len(proxies), "proxies:")
    print(base.round(4).to_string(index=False))

    # trading test
    df["inattention"] = attention_index(df)
    df["tercile"] = pd.qcut(df["inattention"], 3, labels=["high attention", "middle", "low attention"])
    trade = []
    for lab, g in df.groupby("tercile", observed=True):
        x = g["car_drift"].dropna()
        for c in COSTS:
            r = x - c
            trade.append({"group": lab, "round_trip_cost": c, "n": len(r), "mean_return": r.mean(),
                          "median_return": r.median(), "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
                          "pct_positive": (r > 0).mean()})
    spread = df.loc[df["tercile"] == "low attention", "car_drift"].dropna()
    rest = df.loc[df["tercile"] == "high attention", "car_drift"].dropna()
    diff = spread.mean() - rest.mean()
    se = math.sqrt(spread.var(ddof=1) / len(spread) + rest.var(ddof=1) / len(rest))
    trade = pd.DataFrame(trade)
    trade.to_csv(out / "diffusion_trading.csv", index=False)
    print("\nTrading test: buy at close of day +1, sell at day +20, return vs. market (decimals):")
    print(trade.round(4).to_string(index=False))
    print(f"\nLow- minus high-attention difference (before costs): {diff:.4f} (t = {diff / se:.2f})")
    print("Note: returns are per trade over ~19 trading days; small, illiquid stocks often cost 1-2% round trip.")
    print(f"\nSaved diffusion_robustness.csv and diffusion_trading.csv to {out}/")


if __name__ == "__main__":
    main()
