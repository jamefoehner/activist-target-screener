#!/usr/bin/env python3
"""
LLM evaluation, step 3: can language models read activist 13D filings correctly?

Each model reads a filing's Item 4 and returns the same fields you labeled. Its
answers are scored against your pass-1 labels, and compared with how well a second
human (labels_passB.csv) and you a week later (labels_pass2.csv) agreed with pass 1.

Prompt variants
  minimal         one-line category definitions
  guide           the full LABELING_GUIDE.md, the same rules the humans used
  guide_examples  the guide plus three worked examples (those three filings are
                  left out of scoring for every variant, so all variants are
                  scored on the same items)

Usage
  pip install anthropic pandas          # (transformers + torch for --provider hf)
  export ANTHROPIC_API_KEY=...          # from console.anthropic.com
  python3 run_eval.py --dry-run         # show item count and prompt sizes, no API calls
  python3 run_eval.py --models claude-haiku-5-5 claude-sonnet-5-5 --prompts minimal guide guide_examples
  python3 run_eval.py --provider hf --models Qwen/Qwen2.5-7B-Instruct --prompts guide

Responses are cached in cache/llm/, so reruns only call the API for new
model/prompt/item combinations. Outputs:
  data/eval_predictions.csv   every prediction, parsed
  data/eval_results.csv       metrics per model and prompt
  data/eval_errors.csv        every disagreement with your labels, with notes and text
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

DEMANDS = ["board_seats", "sale_or_review", "breakup_spin", "capital_return", "operations_costs",
           "management_change", "governance", "oppose_deal", "capital_discipline", "engagement_only"]
TONES = ["cooperative", "neutral", "confrontational"]
CACHE = Path("cache/llm")

MINIMAL_DEFS = """Demand categories (check every one the filing states as a specific intention, request or
recommendation for this company; ignore boilerplate lists of actions the investor "may" take):
- board_seats: the activist seeks seats or nominates directors (or a settlement grants them)
- sale_or_review: sale of the company, a merger, or a strategic review / exploring strategic alternatives
- breakup_spin: spin-off, separation, or sale of a division or business
- capital_return: buybacks, special dividends, or other return of cash to shareholders
- operations_costs: operational improvements, margin improvement, or cost cuts
- management_change: replacing the CEO or other executives
- governance: governance changes (independent directors, bylaws, declassifying the board, executive pay)
- oppose_deal: opposing a pending transaction or demanding better terms
- capital_discipline: stop or limit acquisitions, cut capex, reduce debt, improve returns on capital
- engagement_only: none of the above (only "investment purposes" and discussions); never with another category"""

OUTPUT_SPEC = """Return ONLY a JSON object, no other text, with exactly these keys:
{"board_seats": bool, "sale_or_review": bool, "breakup_spin": bool, "capital_return": bool,
 "operations_costs": bool, "management_change": bool, "governance": bool, "oppose_deal": bool,
 "capital_discipline": bool, "engagement_only": bool,
 "seats": int,            // board seats requested or granted, 0 if no number is stated
 "proxy_fight": "yes" | "no",
 "tone": "cooperative" | "neutral" | "confrontational"}"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_gold():
    sample = pd.read_csv("data/label_sample.csv")
    gold = pd.read_csv("data/labels_pass1.csv")
    if "capital_discipline" not in gold.columns:
        print("Warning: labels_pass1.csv has no capital_discipline column; run review_capital_discipline.py. "
              "Treating it as 0.")
        gold["capital_discipline"] = 0
    gold = gold[~gold["notes"].fillna("").str.upper().str.startswith("EXCLUDE")]
    df = gold.merge(sample[["item_id", "activist", "subject_name", "item4_text"]], on="item_id")
    for d in DEMANDS:
        df[d] = df[d].fillna(0).astype(int)
    df["seats"] = df["seats"].fillna(0).astype(int)
    return df


def pick_examples(df, ids=None):
    """Three worked examples: a boilerplate-only filing, a proxy fight, and a multi-demand filing."""
    if ids:
        return df[df["item_id"].isin(ids)]
    clear = df[df["ambiguous"].fillna(0).astype(int) == 0].sort_values("item_id")
    n_dem = clear[[d for d in DEMANDS if d != "engagement_only"]].sum(axis=1)
    picks = []
    for cond in [clear["engagement_only"] == 1, clear["proxy_fight"] == "yes", n_dem >= 3]:
        cand = clear[cond & ~clear["item_id"].isin(picks)]
        if len(cand):
            picks.append(int(cand["item_id"].iloc[0]))
    return df[df["item_id"].isin(picks)]


def gold_json(r):
    out = {d: bool(r[d]) for d in DEMANDS}
    out.update(seats=int(r["seats"]), proxy_fight=r["proxy_fight"], tone=r["tone"])
    return json.dumps(out)


def build_prompt(variant, guide_text, examples):
    head = ("You are reading Item 4 ('Purpose of Transaction') of an activist investor's Schedule 13D filing. "
            "Extract what the activist states it wants. Label only what the text says; do not infer from "
            "outside knowledge.\n\n")
    if variant == "minimal":
        return head + MINIMAL_DEFS + "\n\n" + OUTPUT_SPEC
    body = head + "Apply these labeling rules exactly:\n\n<labeling_guide>\n" + guide_text + "\n</labeling_guide>\n\n"
    if variant == "guide_examples":
        for i, (_, r) in enumerate(examples.iterrows(), 1):
            body += f"<example {i}>\n<item4>\n{r['item4_text'][:6000]}\n</item4>\n<answer>{gold_json(r)}</answer>\n</example {i}>\n\n"
    return body + OUTPUT_SPEC


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

class Anthropic:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.no_temperature = set()            # models that don't accept temperature=0 (remembered after the first refusal)

    def __call__(self, model, system, user):
        # max_tokens leaves room in case the model reasons before answering; only text blocks are kept.
        kwargs = dict(model=model, max_tokens=2000, system=system, messages=[{"role": "user", "content": user}])
        if model not in self.no_temperature:
            kwargs["temperature"] = 0
        for attempt in range(6):
            try:
                t0 = time.time()
                r = self.client.messages.create(**kwargs)
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return {"text": text, "input_tokens": r.usage.input_tokens,
                        "output_tokens": r.usage.output_tokens, "seconds": time.time() - t0}
            except Exception as e:
                if "temperature" in str(e).lower() and "temperature" in kwargs:
                    kwargs.pop("temperature")          # some models don't accept temperature 0
                    self.no_temperature.add(model)
                    print(f"    note: {model} doesn't accept temperature=0; using its default for all remaining calls")
                    continue
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                    raise SystemExit(f"\nStopping: {type(e).__name__}: {e}\n"
                                     "Check your API key (see the instructions in the chat) and rerun.")
                if type(e).__name__ in ("BadRequestError", "NotFoundError"):
                    print(f"    {type(e).__name__}: {e}")
                    return {"text": "", "error": str(e)}
                wait = 5 * (attempt + 1)
                print(f"    API error ({type(e).__name__}); retrying in {wait}s")
                time.sleep(wait)
        return {"text": "", "error": "failed after retries"}


class HuggingFace:
    def __init__(self):
        from transformers import pipeline
        self.pipeline, self.pipes = pipeline, {}

    def __call__(self, model, system, user):
        if model not in self.pipes:
            self.pipes[model] = self.pipeline("text-generation", model=model, device_map="auto")
        t0 = time.time()
        out = self.pipes[model]([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                max_new_tokens=400, do_sample=False)
        text = out[0]["generated_text"][-1]["content"]
        return {"text": text, "input_tokens": None, "output_tokens": None, "seconds": time.time() - t0}


class Fake:
    """For testing the pipeline without an API: answers engagement_only for everything."""
    def __call__(self, model, system, user):
        ans = {d: d == "engagement_only" for d in DEMANDS}
        ans.update(seats=0, proxy_fight="no", tone="neutral")
        return {"text": json.dumps(ans), "input_tokens": len(system) // 4, "output_tokens": 60, "seconds": 0.0}


def cached_call(backend, provider, model, variant, item_id, system, user):
    key = hashlib.md5((system + user).encode()).hexdigest()[:10]
    f = CACHE / f"{provider}__{model.replace('/', '_')}__{variant}__{item_id}__{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    res = backend(model, system, user)
    if not res.get("error"):                  # don't cache failures, so a rerun retries them
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res))
    return res


def parse(text):
    """Pull the first JSON object out of the reply and validate the fields."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for d in DEMANDS:
        if d not in j:
            return None
        out[d] = int(bool(j[d]))
    try:
        out["seats"] = int(j.get("seats", 0) or 0)
    except (TypeError, ValueError):
        out["seats"] = 0
    out["proxy_fight"] = "yes" if str(j.get("proxy_fight", "no")).lower().startswith("y") else "no"
    tone = str(j.get("tone", "neutral")).lower()
    out["tone"] = tone if tone in TONES else "neutral"
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def prf(gold, pred):
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp and (p + r) else (float("nan") if tp + fp + fn == 0 else 0.0)
    return p, r, f, tp + fn


def score(gold, pred, label):
    """gold, pred: DataFrames indexed by item_id with the label fields."""
    g, p = gold.loc[pred.index], pred
    row = {"system": label, "n": len(p)}
    f1s = []
    for d in DEMANDS:
        pr, rc, f1, support = prf(g[d], p[d])
        row[f"{d}_f1"] = f1
        row[f"{d}_support"] = support
        if support:
            f1s.append(f1)
    row["macro_f1"] = sum(f1s) / len(f1s) if f1s else float("nan")
    gd, pdm = g[DEMANDS].values, p[DEMANDS].values
    tp, fp, fn = int(((gd == 1) & (pdm == 1)).sum()), int(((gd == 0) & (pdm == 1)).sum()), int(((gd == 1) & (pdm == 0)).sum())
    row["micro_f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    row["exact_demands"] = (gd == pdm).all(axis=1).mean()
    row["seats_exact"] = (g["seats"] == p["seats"]).mean()
    row["proxy_acc"] = (g["proxy_fight"] == p["proxy_fight"]).mean()
    row["tone_acc"] = (g["tone"] == p["tone"]).mean()
    amb = g["ambiguous"].fillna(0).astype(int) == 1
    exact = pd.Series((gd == pdm).all(axis=1), index=g.index)
    row["exact_clear"] = exact[~amb].mean() if (~amb).any() else float("nan")
    row["exact_ambiguous"] = exact[amb].mean() if amb.any() else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["anthropic", "hf", "fake"], default="anthropic")
    ap.add_argument("--models", nargs="+", default=["claude-haiku-5-5", "claude-sonnet-5-5"])
    ap.add_argument("--prompts", nargs="+", default=["minimal", "guide", "guide_examples"],
                    choices=["minimal", "guide", "guide_examples"])
    ap.add_argument("--example-ids", nargs="*", type=int, help="item_ids to use as worked examples")
    ap.add_argument("--guide", default="LABELING_GUIDE.md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = load_gold()
    examples = pick_examples(df, args.example_ids)
    ex_ids = set(examples["item_id"])
    items = df[~df["item_id"].isin(ex_ids)].reset_index(drop=True)
    guide_text = Path(args.guide).read_text()
    prompts = {v: build_prompt(v, guide_text, examples) for v in args.prompts}

    print(f"{len(items)} filings to score (excluding non-campaigns and worked examples {sorted(ex_ids)})")
    total_chars = items["item4_text"].str.len().sum()
    for v, s in prompts.items():
        est = (len(s) * len(items) + total_chars) / 4
        print(f"  prompt '{v}': ~{len(s) / 4:,.0f} tokens of instructions; ~{est:,.0f} input tokens per model")
    if args.dry_run:
        print("\nDry run: no API calls made. Multiply input tokens by your model's price per million tokens "
              "(see the Claude pricing page) to estimate cost.")
        return

    backend = {"anthropic": Anthropic, "hf": HuggingFace, "fake": Fake}[args.provider]()
    gold = items.set_index("item_id")
    preds, results, errors = [], [], []
    for model in args.models:
        for v, system in prompts.items():
            print(f"\n{model} / {v}")
            rows, fails, tok_in, tok_out, secs = [], 0, 0, 0, 0.0
            for i, r in items.iterrows():
                user = f"<item4>\n{r['item4_text']}\n</item4>"
                res = cached_call(backend, args.provider, model, v, r["item_id"], system, user)
                parsed = parse(res.get("text", ""))
                tok_in += res.get("input_tokens") or 0
                tok_out += res.get("output_tokens") or 0
                secs += res.get("seconds") or 0
                ok = parsed is not None
                if not ok:          # unparseable replies count as wrong on every field (an all-zero answer)
                    fails += 1
                    parsed = {d: 0 for d in DEMANDS} | {"seats": 0, "proxy_fight": "no", "tone": "neutral"}
                rows.append({"item_id": r["item_id"], **parsed, "parse_ok": ok})
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(items)}")
            pred = pd.DataFrame(rows).set_index("item_id")
            res_row = score(gold, pred, f"{model} | {v}")
            res_row.update(model=model, prompt=v, parse_fail_rate=fails / len(items),
                           input_tokens=tok_in, output_tokens=tok_out, avg_seconds=secs / len(items))
            results.append(res_row)
            preds.append(pred.assign(model=model, prompt=v).reset_index())
            g = gold.loc[pred.index]
            for f in DEMANDS + ["seats", "proxy_fight", "tone"]:
                for iid in pred.index[g[f].astype(str).values != pred[f].astype(str).values]:
                    errors.append({"model": model, "prompt": v, "item_id": iid, "field": f,
                                   "human": g.at[iid, f], "model_answer": pred.at[iid, f],
                                   "human_ambiguous": g.at[iid, "ambiguous"], "human_notes": g.at[iid, "notes"],
                                   "activist": g.at[iid, "activist"], "company": g.at[iid, "subject_name"],
                                   "chars": len(g.at[iid, "item4_text"])})

    # human reference on the shared subset (second labeler and your pass 2)
    for name, path in [("Human: second labeler", "data/labels_passB.csv"), ("Human: you, pass 2", "data/labels_pass2.csv")]:
        if Path(path).exists():
            h = pd.read_csv(path)
            for d in DEMANDS:
                h[d] = h[d].fillna(0).astype(int) if d in h else 0
            h["seats"] = h["seats"].fillna(0).astype(int)
            h = h[h["item_id"].isin(gold.index)].set_index("item_id")
            if len(h):
                results.append(score(gold, h[DEMANDS + ["seats", "proxy_fight", "tone"]], name) | {"model": name, "prompt": "-"})
                shared = h.index
                for pr in preds:                       # each model on the same shared items, for a fair comparison
                    m = pr.set_index("item_id").loc[lambda x: x.index.isin(shared)]
                    lab = f"{pr['model'].iloc[0]} | {pr['prompt'].iloc[0]} (on {name.split(': ')[1]}'s items)"
                    results.append(score(gold, m[DEMANDS + ["seats", "proxy_fight", "tone"]], lab)
                                   | {"model": pr["model"].iloc[0], "prompt": pr["prompt"].iloc[0] + " [shared]"})

    pd.concat(preds).to_csv("data/eval_predictions.csv", index=False)
    res = pd.DataFrame(results)
    res.to_csv("data/eval_results.csv", index=False)
    err = pd.DataFrame(errors)
    if len(err):
        err.merge(items[["item_id", "item4_text"]], on="item_id", how="left").to_csv("data/eval_errors.csv", index=False)

    pd.set_option("display.width", 220)
    main_cols = ["system", "n", "micro_f1", "macro_f1", "exact_demands", "exact_clear", "exact_ambiguous",
                 "seats_exact", "proxy_acc", "tone_acc"]
    print("\nOverall results (scored against your pass-1 labels):")
    print(res[~res["prompt"].astype(str).str.contains("shared")][main_cols].round(3).to_string(index=False))
    shared = res[res["prompt"].astype(str).str.contains("shared") | res["system"].str.startswith("Human")]
    if len(shared):
        print("\nHumans vs. models on the same shared items:")
        print(shared[main_cols].round(3).to_string(index=False))
    per_cat = res[~res["prompt"].astype(str).str.contains("shared") & ~res["system"].str.startswith("Human")]
    print("\nF1 by category (support = how many filings you labeled positive):")
    cols = ["system"] + [f"{d}_f1" for d in DEMANDS]
    print(per_cat[cols].rename(columns=lambda c: c.replace("_f1", "")).round(2).to_string(index=False))
    print("\nCost and reliability:")
    print(per_cat[["system", "parse_fail_rate", "input_tokens", "output_tokens", "avg_seconds"]].round(3).to_string(index=False))
    if len(err):
        print(f"\n{len(err)} disagreements saved to data/eval_errors.csv. Most frequent:")
        print(err.groupby(["field"]).size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
