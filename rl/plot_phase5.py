"""Phase 5 figures: dev trajectories per run, and gold results per model."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rl_common import load, score

D = "data/rl"
C = {"naive": "#c0392b", "better_base": "#e08e0b", "better_sft": "#2a6f97"}
RUNS = [("naive", "Naive reward, from base", "hack_naive_log.jsonl"),
        ("better_base", "Better reward, from base", "hack_better_log.jsonl"),
        ("better_sft", "Better reward, from SFT", "hack_better_sft_log.jsonl")]

def evals(fn):
    p = f"{D}/{fn}"
    if not os.path.exists(p): return None
    return [r for r in map(json.loads, open(p)) if r["kind"] == "eval"]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
panels = [("dev_all_zero", "Share of dev answers that are all-zero"),
          ("dev_macro_f1", "Dev macro-F1"),
          ("dev_exact", "Dev exact match")]
missing = []
for a, (key, title) in zip(ax, panels):
    for k, lab, fn in RUNS:
        ev = evals(fn)
        if ev is None: missing.append(fn); continue
        a.plot([e["step"] for e in ev], [e[key] for e in ev], marker="o", color=C[k], label=lab)
    a.set_title(title, fontsize=10); a.set_xlabel("GRPO step"); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=8, frameon=False)
fig.suptitle("Dev set during training (n=37, one seed per run)", fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig("figures/rl_dev_trajectories.png", dpi=160, bbox_inches="tight")

rows = load(f"{D}/test_gold.jsonl")
MODELS = [("Base,\nuntrained", "baseline_preds"), ("Naive reward\nfrom base", "hack_naive_preds"),
          ("Better reward\nfrom base", "hack_better_preds"), ("SFT", "sft_preds2"),
          ("SFT + better\n(lr 5e-5)", "hack_better_sft_preds"), ("SFT + better\n(lr 2e-5, 60 steps)", "grpo60_preds")]
res = []
for lab, f in MODELS:
    P = {}
    for l in open(f"{D}/{f}.jsonl"):
        r = json.loads(l); P[r["id"]] = r["pred"]
    res.append((lab, score(rows, [P[r["id"]] for r in rows])))
fig, a = plt.subplots(figsize=(9.5, 4))
w = 0.38; x = range(len(res))
cols = ["#8a8a8a", C["naive"], C["better_base"], "#4a4a4a", C["better_sft"], C["better_sft"]]
a.bar([i - w/2 for i in x], [s["macro_f1"] for _, s in res], w, color=cols, label="Macro-F1")
a.bar([i + w/2 for i in x], [s["exact_match"] for _, s in res], w, color=cols, alpha=0.45, label="Exact match")
for i, (_, s) in enumerate(res):
    a.text(i - w/2, s["macro_f1"] + .01, f"{s['macro_f1']:.2f}", ha="center", fontsize=8)
    a.text(i + w/2, s["exact_match"] + .01, f"{s['exact_match']:.2f}", ha="center", fontsize=8)
a.set_xticks(list(x)); a.set_xticklabels([l for l, _ in res], fontsize=8)
a.set_ylim(0, 0.9); a.spines[["top", "right"]].set_visible(False)
a.axhline(0.449, ls="--", lw=0.8, color="#555"); a.text(5.5, 0.455, "always 'engagement_only'", ha="right", fontsize=7)
a.set_title("Gold set (89 human-labeled filings): solid = macro-F1, light = exact match", fontsize=10)
fig.tight_layout(); fig.savefig("figures/rl_gold_results.png", dpi=160)
for lab, s in res: print(lab.replace("\n", " "), round(s["macro_f1"], 3), round(s["exact_match"], 3))
print("missing logs:", sorted(set(missing)))
