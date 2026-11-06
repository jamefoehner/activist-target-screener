"""Reward functions for GRPO (phase 4) and the reward-hacking experiment (phase 5).
All take the model's raw text and the gold label dict; return a float."""
from rl_common import FIELDS, parse_answer


def reward_naive(raw, gold):
    """Plain accuracy: fraction of the 10 fields answered correctly. HACKABLE: all-zeros scores ~84%."""
    p = parse_answer(raw)
    if p is None:
        return 0.0
    return sum(p[f] == int(gold[f]) for f in FIELDS) / len(FIELDS)


def reward_better(raw, gold):
    """Jaccard overlap of the SET of positive fields (engagement_only counts as a field).
    All-zeros scores 0 because every gold filing has >=1 positive. Parse failure = 0.
    Penalty for the guide's forbidden combo (engagement_only + another field)."""
    p = parse_answer(raw)
    if p is None:
        return 0.0
    P = {f for f in FIELDS if p[f]}
    G = {f for f in FIELDS if int(gold[f])}
    r = len(P & G) / len(P | G) if (P | G) else 1.0
    if p["engagement_only"] and len(P) > 1:
        r -= 0.2
    return r


REWARDS = {"naive": reward_naive, "better": reward_better}

if __name__ == "__main__":     # sanity check: how does each reward score trivial / perfect answers?
    import json, sys
    from rl_common import load, gold_vec
    path = sys.argv[1] if len(sys.argv) > 1 else "data/rl/test_gold.jsonl"
    rows = load(path)
    z = json.dumps({f: 0 for f in FIELDS}); eo = json.dumps({**{f: 0 for f in FIELDS}, "engagement_only": 1})
    for name, fn in REWARDS.items():
        for lab, mk in [("always_no", lambda r: z), ("always_EO", lambda r: eo),
                        ("perfect", lambda r: json.dumps(gold_vec(r))), ("garbage", lambda r: "hello")]:
            print(f"{name:7s} {lab:10s} {sum(fn(mk(r), gold_vec(r)) for r in rows) / len(rows):.3f}")
