"""Phase 4/5: GRPO (group-relative policy gradient) with a pluggable reward.
  python grpo_train.py --init sft_adapter --reward better --out grpo_better      # phase 4 (main)
  python grpo_train.py --init sft_adapter --reward naive  --out grpo_naive       # phase 5 (reward hacking)
  python grpo_train.py --init none        --reward better --out grpo_from_base   # optional: RL without SFT
Per step: sample B filings, G completions each, reward each, advantage = (r - group mean)/group std,
loss = -A * mean token logprob. One update per batch (on-policy, so no ratio/clipping), no KL term.
Rewards use the SILVER labels in train.jsonl. Dev (silver) is for monitoring. Gold is only for final eval."""
import argparse, json, random, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from rl_common import *
from rewards import REWARDS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--init", default="sft_adapter", help="LoRA adapter dir to start from, or 'none' for a fresh LoRA")
ap.add_argument("--reward", choices=list(REWARDS), default="better")
ap.add_argument("--train", default="data/rl/train.jsonl")
ap.add_argument("--dev", default="data/rl/dev.jsonl")
ap.add_argument("--out", default="grpo_adapter")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--B", type=int, default=2, help="filings per step")
ap.add_argument("--G", type=int, default=8, help="completions per filing")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--max_new", type=int, default=100)
ap.add_argument("--max_chars", type=int, default=None, help="truncate filing text (smoke tests only)")
ap.add_argument("--dev_limit", type=int, default=None)
ap.add_argument("--pos_frac", type=float, default=0.5, help="share of each batch drawn from filings with a non-engagement_only label")
ap.add_argument("--gen_tokens", type=int, default=9000, help="token budget per generate call: chunk = gen_tokens // prompt_len (lower if CUDA OOM)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
cuda = dev_ == "cuda"
dtype = torch.float16 if cuda else torch.float32
reward_fn = REWARDS[a.reward]


def ac():
    return torch.autocast("cuda", dtype=torch.float16) if cuda else torch.autocast("cpu", enabled=False)


tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev_)
if a.init.lower() == "none":
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
else:
    model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler(enabled=cuda)
EOS = tok.convert_tokens_to_ids("<|im_end|>")


def prompt_ids(r):
    t = r["text"][: a.max_chars] if a.max_chars else r["text"]
    s = tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(dev_)


@torch.no_grad()
def rollouts(r):
    model.eval()
    x = prompt_ids(r)
    comps = []
    chunk = max(1, min(a.G, a.gen_tokens // x.shape[1]))
    for k in range(0, a.G, chunk):
        n = min(chunk, a.G - k)
        with ac():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=a.temp,
                                 top_p=1.0, top_k=0, max_new_tokens=a.max_new, num_return_sequences=n,
                                 pad_token_id=tok.pad_token_id, eos_token_id=EOS)
        for o in out[:, x.shape[1]:]:
            o = o.tolist()
            if EOS in o: o = o[: o.index(EOS) + 1]
            comps.append(o)
        del out
        if cuda: torch.cuda.empty_cache()
    model.train()
    return x, comps


def seq_logprob(x, comp):
    """sum-free mean token logprob of comp given prompt x (grad enabled)."""
    ids = torch.cat([x, torch.tensor([comp], device=dev_)], 1)
    with ac():
        logits = model(input_ids=ids, logits_to_keep=len(comp) + 1).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor([comp], device=dev_).unsqueeze(-1)).squeeze(-1)
    return lp.mean()


@torch.no_grad()
def evaluate(rows):
    model.eval(); preds, raws = [], []
    for r in rows:
        x = prompt_ids(r)
        with ac():
            o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                               max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, eos_token_id=EOS)
        raw = tok.decode(o[0, x.shape[1]:], skip_special_tokens=True); raws.append(raw); preds.append(parse_answer(raw))
    model.train()
    s = score(rows, preds)
    ok = [p for p in preds if p]
    s["frac_all_zero"] = sum(not any(p.values()) for p in ok) / max(1, len(rows))
    s["mean_positives"] = sum(sum(p.values()) for p in ok) / max(1, len(ok))
    s["reward_naive"] = sum(REWARDS["naive"](t, gold_vec(r)) for t, r in zip(raws, rows)) / len(rows)
    s["reward_better"] = sum(REWARDS["better"](t, gold_vec(r)) for t, r in zip(raws, rows)) / len(rows)
    return s


train = load(a.train); dev = load(a.dev)[: a.dev_limit]
if a.max_chars:
    pass
log = open(a.out + "_log.jsonl", "w")


def do_eval(step):
    s = evaluate(dev)
    rec = {"step": step, "dev_macro_f1": s["macro_f1"], "dev_exact": s["exact_match"], "dev_parse": s["parse_rate"],
           "dev_all_zero": s["frac_all_zero"], "dev_mean_positives": s["mean_positives"],
           "dev_reward_naive": s["reward_naive"], "dev_reward_better": s["reward_better"], "kind": "eval"}
    print(f"[eval step {step}] macroF1={rec['dev_macro_f1']:.3f} exact={rec['dev_exact']:.2f} parse={rec['dev_parse']:.2f} "
          f"all_zero={rec['dev_all_zero']:.2f} positives/filing={rec['dev_mean_positives']:.2f} "
          f"naive={rec['dev_reward_naive']:.3f} better={rec['dev_reward_better']:.3f}", flush=True)
    log.write(json.dumps(rec) + "\n"); log.flush()


do_eval(0)
pos = [r for r in train if any(v for k, v in r["label"].items() if k in FIELDS and k != "engagement_only")] or train
neg = [r for r in train if r not in pos] or train
print(f"prompts: {len(train)} ({len(pos)} with a non-engagement label); pos_frac={a.pos_frac}", flush=True)
t0 = time.time()
for step in range(1, a.steps + 1):
    batch = [random.choice(pos if random.random() < a.pos_frac else neg) for _ in range(a.B)]
    stats = dict(rew=[], zero=0, n=0, used=0)
    opt.zero_grad()
    groups = []
    for r in batch:
        x, comps = rollouts(r)
        raws = [tok.decode(c, skip_special_tokens=True) for c in comps]
        rs = torch.tensor([reward_fn(t, r["label"]) for t in raws])
        stats["rew"] += rs.tolist(); stats["n"] += len(raws)
        stats["zero"] += sum(1 for t in raws if (parse_answer(t) and not any(parse_answer(t).values())))
        if rs.std() > 1e-6:
            groups.append((x, comps, ((rs - rs.mean()) / (rs.std() + 1e-4)).tolist()))
    n_samples = sum(len(c) for _, c, _ in groups)
    if cuda: torch.cuda.empty_cache()
    for x, comps, adv in groups:
        for comp, A in zip(comps, adv):
            loss = -A * seq_logprob(x, comp) / n_samples
            scaler.scale(loss).backward(); stats["used"] += 1
    if groups:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
    rec = {"step": step, "kind": "train", "reward_mean": sum(stats["rew"]) / len(stats["rew"]),
           "rollout_all_zero": stats["zero"] / stats["n"], "groups_with_signal": len(groups), "reward": a.reward}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"step {step}/{a.steps} reward={rec['reward_mean']:.3f} rollout_all_zero={rec['rollout_all_zero']:.2f} "
          f"signal_groups={len(groups)}/{a.B} {time.time() - t0:.0f}s", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        do_eval(step); model.save_pretrained(a.out)
model.save_pretrained(a.out)
print("saved", a.out)
