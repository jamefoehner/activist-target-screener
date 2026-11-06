"""Phase 2: untrained small model on the human-labeled gold set. Run on a GPU (Colab).
  pip install -q transformers accelerate torch
  python baseline_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/rl/test_gold.jsonl
Also prints trivial baselines (always-no, always-engagement-only) that the model must beat."""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rl_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--data", default="data/rl/test_gold.jsonl")
ap.add_argument("--out", default="data/rl/baseline_preds.jsonl")
ap.add_argument("--bs", type=int, default=None)
ap.add_argument("--adapter", default=None, help="LoRA adapter dir to load on top of --model")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

rows = load(a.data)[: a.limit]
trivial = {"always_no": [{f: 0 for f in FIELDS}] * len(rows),
           "always_engagement_only": [{**{f: 0 for f in FIELDS}, "engagement_only": 1}] * len(rows)}
for k, p in trivial.items():
    report(k, score(rows, p))

tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float16 if dev == "mps" else torch.float32
a.bs = a.bs or (4 if dev == "cuda" else 1)
print(f"device={dev} dtype={dtype} bs={a.bs}", flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev)
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).eval()
prompts = [tok.apply_chat_template(build_messages(r["text"]), tokenize=False, add_generation_prompt=True) for r in rows]
outs = []
for i in range(0, len(prompts), a.bs):
    enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=120, do_sample=False)
    outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  {len(outs)}/{len(rows)}", flush=True)
preds = [parse_answer(o) for o in outs]
with open(a.out, "w") as f:
    for r, o, p in zip(rows, outs, preds):
        f.write(json.dumps({"id": r["id"], "raw": o, "pred": p}) + "\n")
report(f"{'SFT' if a.adapter else 'BASELINE'} {a.model}", score(rows, preds))
