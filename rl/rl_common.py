"""Shared pieces for phases 2-5: prompt, answer parser, metrics. No GPU needed to import."""
import json, re

FIELDS = ["board_seats", "sale_or_review", "breakup_spin", "capital_return", "operations_costs",
          "management_change", "governance", "oppose_deal", "capital_discipline", "engagement_only"]

GUIDE = """You label SEC Schedule 13D Item 4 text from activist investors.
Label what the activist says it INTENDS, SEEKS or RECOMMENDS for this company. Ignore boilerplate
("may engage in discussions", "may take actions relating to clauses (a)-(j)") and mere reservations of rights.
Fields (1 = yes, 0 = no):
- board_seats: nominating directors / seeking board representation / settlement adding directors
- sale_or_review: sale of the company, merger, strategic review or alternatives
- breakup_spin: spin-off, separation, or sale of a division
- capital_return: buybacks, special dividends, returning cash
- operations_costs: cost cuts, margin improvement, operational restructuring
- management_change: replacing the CEO or executives
- governance: declassify board, remove poison pill, bylaw/charter changes, shareholder rights
- oppose_deal: opposing an announced merger/acquisition or demanding better terms
- capital_discipline: limit acquisitions/capex, improve returns on capital, reduce debt, capital allocation policy
- engagement_only: none of the above, only 'investment purposes' and discussions with management.
  Never 1 together with another field.
Reply with ONLY a JSON object with exactly these keys: """ + ", ".join(FIELDS) + "."


def build_messages(text):
    return [{"role": "system", "content": GUIDE},
            {"role": "user", "content": f"Item 4 text:\n\n{text}\n\nJSON:"}]


def parse_answer(s):
    """Return dict of 0/1 for all FIELDS, or None if unparseable."""
    m = re.search(r"\{.*?\}", s, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {f: int(bool(int(d[f]))) for f in FIELDS}
    except Exception:
        return None


def load(path):
    return [json.loads(l) for l in open(path)]


def gold_vec(row):
    return {f: int(row["label"][f]) for f in FIELDS}


def score(rows, preds):
    """preds: list of dict-or-None aligned with rows. Unparseable counts as all-zero but is reported."""
    n = len(rows)
    parsed = sum(p is not None for p in preds)
    out = {"n": n, "parse_rate": parsed / n, "fields": {}}
    exact = 0
    for r, p in zip(rows, preds):
        g, p = gold_vec(r), (p or {f: 0 for f in FIELDS})
        exact += all(g[f] == p[f] for f in FIELDS)
    out["exact_match"] = exact / n
    f1s, accs = [], []
    for f in FIELDS:
        tp = fp = fn = tn = 0
        for r, p in zip(rows, preds):
            g = int(r["label"][f]); q = (p or {}).get(f, 0)
            tp += g and q; fp += (not g) and q; fn += g and (not q); tn += (not g) and (not q)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / n
        out["fields"][f] = dict(support=tp + fn, precision=prec, recall=rec, f1=f1, acc=acc)
        f1s.append(f1); accs.append(acc)
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["mean_field_acc"] = sum(accs) / len(accs)
    return out


def report(name, s):
    print(f"\n== {name}  (n={s['n']}) parse={s['parse_rate']:.0%} exact={s['exact_match']:.1%} "
          f"mean-acc={s['mean_field_acc']:.1%} macro-F1={s['macro_f1']:.3f}")
    print(f"{'field':20s}{'support':>8s}{'prec':>7s}{'rec':>7s}{'F1':>7s}{'acc':>7s}")
    for f, m in s["fields"].items():
        print(f"{f:20s}{m['support']:8d}{m['precision']:7.2f}{m['recall']:7.2f}{m['f1']:7.2f}{m['acc']:7.2f}")
