#!/usr/bin/env python3
"""
Phase 3b — Train and test the activist-target prediction model.

Testing method: walk-forward. For each test year Y, train only on years up to
Y-2 (skipping Y-1, whose 12-month outcomes weren't fully known yet when Y's
predictions would have been made), then predict year Y. Pooling those
out-of-sample predictions shows how the model would have done in real time.

Models
  size_only  — baseline: bigger companies get targeted more; does the model beat that?
  logistic   — simple, explainable; features converted to within-year percentiles
  gbm        — gradient-boosted trees; can pick up non-linear patterns

Scoring
  The best model (by out-of-sample average precision) is refit on all labeled
  years and used to score each company's latest row ("current"), with the top
  three reasons each company is flagged.

Usage
  python3 -m pip install scikit-learn
  python3 train_model.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = ["ebitda_ev_rel", "book_mkt_rel", "ebitda_margin_rel", "roa_rel", "sales_growth_rel",
            "cash_assets_rel", "leverage_rel", "payout_rel", "acq_assets_rel", "capex_assets_rel",
            "rd_sales_rel", "log_mktcap", "ret_1y_rel", "tsr_3y_rel"]

# how to describe each feature when it is high / low
LABELS = {
    "ebitda_ev_rel": ("cheap on EBITDA/EV vs peers", "expensive on EBITDA/EV vs peers"),
    "book_mkt_rel": ("cheap on book value vs peers", "expensive on book value vs peers"),
    "ebitda_margin_rel": ("high EBITDA margin vs peers", "low EBITDA margin vs peers"),
    "roa_rel": ("high return on assets vs peers", "low return on assets vs peers"),
    "sales_growth_rel": ("fast sales growth vs peers", "slow sales growth vs peers"),
    "cash_assets_rel": ("large cash pile vs peers", "small cash pile vs peers"),
    "leverage_rel": ("heavy debt load vs peers", "light debt load vs peers"),
    "payout_rel": ("high dividends+buybacks vs peers", "low dividends+buybacks vs peers"),
    "acq_assets_rel": ("heavy acquisition spending vs peers", "little acquisition spending vs peers"),
    "capex_assets_rel": ("heavy capex vs peers", "light capex vs peers"),
    "rd_sales_rel": ("heavy R&D spending vs peers", "light R&D spending vs peers"),
    "log_mktcap": ("large company", "small company"),
    "ret_1y_rel": ("strong 1-yr stock return vs market", "weak 1-yr stock return vs market"),
    "tsr_3y_rel": ("strong 3-yr stock return vs market", "weak 3-yr stock return vs market"),
}


def describe(feature: str, rank: float) -> str:
    high, low = LABELS[feature]
    return high if rank >= 0.5 else low


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-year percentile ranks (0-1), missing -> 0.5. Tames extreme values for the logistic model."""
    key = df["sample"] + "_" + df["avail_date"].dt.year.astype(str)
    return df.groupby(key)[FEATURES].rank(pct=True).fillna(0.5)


def make_models():
    return {
        "logistic": LogisticRegression(C=0.5, max_iter=5000),
        "gbm": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=3,
                                              min_samples_leaf=100, l2_regularization=1.0,
                                              random_state=0),
    }


def fit_predict(name, model, tr, te, R):
    if name == "logistic":
        model.fit(R.loc[tr.index], tr["targeted"])
        return model.predict_proba(R.loc[te.index])[:, 1]
    model.fit(tr[FEATURES], tr["targeted"])
    return model.predict_proba(te[FEATURES])[:, 1]


def walk_forward(lab: pd.DataFrame, R: pd.DataFrame, first_test_year: int) -> pd.DataFrame:
    year = lab["avail_date"].dt.year
    preds = []
    for y in range(first_test_year, year.max() + 1):
        tr, te = lab[year <= y - 2], lab[year == y]
        if tr["targeted"].sum() < 20 or te.empty:
            continue
        out = te[["gvkey", "avail_date", "targeted"]].copy()
        out["size_only"] = te["log_mktcap"].fillna(te["log_mktcap"].median())
        for name, model in make_models().items():
            out[name] = fit_predict(name, model, tr, te, R)
        preds.append(out)
        print(f"  test year {y}: trained on {len(tr):,} rows ({tr['targeted'].sum()} targets), "
              f"tested on {len(te):,} ({te['targeted'].sum()} targets)")
    return pd.concat(preds)


def evaluate(p: pd.DataFrame, models) -> pd.DataFrame:
    base = p["targeted"].mean()
    year = p["avail_date"].dt.year
    rows = {}
    for m in models:
        pct = p.groupby(year)[m].rank(pct=True)            # rank within each year, like a real screen
        top10, top1 = p[pct >= 0.90], p[pct >= 0.99]
        rows[m] = {
            "ROC AUC": roc_auc_score(p["targeted"], p[m]),
            "avg precision": average_precision_score(p["targeted"], p[m]),
            "targets caught in top 10%": top10["targeted"].sum() / p["targeted"].sum(),
            "hit rate in top 1%": top1["targeted"].mean(),
            "lift in top 1% (x base rate)": top1["targeted"].mean() / base,
        }
    t = pd.DataFrame(rows).T
    print(f"\nOut-of-sample results ({len(p):,} firm-years, {p['targeted'].sum()} targets, "
          f"base rate {base:.2%}). Random guessing: AUC 0.50, avg precision {base:.3f}, "
          "top 10% catches 10%, lift 1x.")
    print(t.round(3).to_string())
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="data/panel.csv")
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--out", default="data")
    ap.add_argument("--show", type=int, default=25, help="top names to print (market cap >= $1B)")
    args = ap.parse_args()
    out = Path(args.out)

    panel = pd.read_csv(args.panel, parse_dates=["avail_date", "datadate", "first_campaign"])
    panel = panel[panel["sample"].isin(["labeled", "current"])].reset_index(drop=True)
    R = rank_features(panel)
    lab = panel[panel["sample"] == "labeled"]
    cur = panel[panel["sample"] == "current"]

    print("Walk-forward testing...")
    preds = walk_forward(lab, R, args.first_test_year)
    preds.to_csv(out / "oos_predictions.csv", index=False)
    results = evaluate(preds, ["size_only", "logistic", "gbm"])
    results.to_csv(out / "model_results.csv")

    # what the logistic model learned (fit on all labeled years; ranks are 0-1, so coefficients compare directly)
    logit = LogisticRegression(C=0.5, max_iter=5000).fit(R.loc[lab.index], lab["targeted"])
    coefs = pd.Series(logit.coef_[0], index=FEATURES).sort_values()
    print("\nWhat raises (+) or lowers (-) the odds of being targeted (logistic, all years):")
    for f, c in coefs.items():
        print(f"  {c:+.2f}  {LABELS[f][0]}")

    if cur.empty:
        print("\nNo 'current' companies in the panel to score — check the trading filter in build_panel.py.")
        return

    # score today's companies with the better model
    best = results["avg precision"].drop("size_only").idxmax()
    print(f"\nScoring current companies with: {best}")
    final = make_models()[best]
    cur = cur.copy()
    cur["score"] = fit_predict(best, final, lab, cur, R)
    cur["score_pct"] = cur["score"].rank(pct=True)

    Rc = R.loc[cur.index]
    contrib = (Rc - 0.5) * logit.coef_[0]          # each feature's push toward "target", logistic view
    cur["top_reasons"] = [
        "; ".join(describe(f, Rc.at[i, f]) for f in contrib.loc[i].sort_values(ascending=False).index[:3]
                  if contrib.at[i, f] > 0)
        for i in cur.index]
    cur["already_targeted_since"] = cur["first_campaign"].dt.date.where(cur["targeted"] == 1)

    cols = ["conm", "gvkey", "permno", "sic", "mktcap", "score_pct", "top_reasons",
            "already_targeted_since", "activist"]
    scored = cur.sort_values("score", ascending=False)[cols]
    scored.to_csv(out / "scores_current.csv", index=False)

    big = scored[scored["mktcap"] >= 1000].head(args.show).copy()
    big["mktcap"] = (big["mktcap"] / 1000).round(1).astype(str) + "B"
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nTop {args.show} current companies with market cap >= $1B:")
    print(big[["conm", "mktcap", "score_pct", "top_reasons", "already_targeted_since"]]
          .to_string(index=False))
    print(f"\nSaved scores_current.csv, model_results.csv and oos_predictions.csv to {out}/")


if __name__ == "__main__":
    main()
