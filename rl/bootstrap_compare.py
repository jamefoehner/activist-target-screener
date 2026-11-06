"""Paired bootstrap over gold filings: is model B better than model A?
  python rl/bootstrap_compare.py data/rl/sft_preds.jsonl data/rl/grpo60_preds.jsonl"""
import json, sys, random
sys.path.insert(0, "rl")
from rl_common import FIELDS, load, score

gold = {r["id"]: r for r in load("data/rl/test_gold.jsonl")}
def preds(path): return {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(path)}
A, B = preds(sys.argv[1]), preds(sys.argv[2])
ids = [i for i in gold if i in A and i in B]
rows = [gold[i] for i in ids]
def m(P, idx): 
    s = score([rows[k] for k in idx], [P[ids[k]] for k in idx])
    return s["macro_f1"], s["exact_match"], s["mean_field_acc"]
base = range(len(ids))
a0, b0 = m(A, base), m(B, base)
random.seed(0); diffs = []
for _ in range(2000):
    idx = [random.randrange(len(ids)) for _ in ids]
    x, y = m(A, idx), m(B, idx)
    diffs.append([y[j] - x[j] for j in range(3)])
print(f"n={len(ids)}")
for j, name in enumerate(["macro_F1", "exact_match", "mean_field_acc"]):
    d = sorted(t[j] for t in diffs)
    print(f"{name:15s} A={a0[j]:.3f} B={b0[j]:.3f} diff={b0[j]-a0[j]:+.3f}  95% CI [{d[50]:+.3f}, {d[1949]:+.3f}]  P(B>A)={sum(x>0 for x in d)/len(d):.2f}")
