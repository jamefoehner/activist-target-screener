"""Phase 3: LoRA supervised fine-tuning on Sonnet's silver labels. Colab T4 is enough.
  pip install -q transformers accelerate peft
  python sft_train.py --train data/rl/train.jsonl --dev data/rl/dev.jsonl --out sft_adapter
Then evaluate:  python baseline_eval.py --adapter sft_adapter --out data/rl/sft_preds.jsonl"""
import argparse, json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from rl_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--train", default="data/rl/train.jsonl")
ap.add_argument("--dev", default="data/rl/dev.jsonl")
ap.add_argument("--out", default="sft_adapter")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max_len", type=int, default=4500)
a = ap.parse_args()
random.seed(0); torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()          # fp32 adapter weights, fp16 base
model.print_trainable_parameters()


def encode(r):
    prompt = tok.apply_chat_template(build_messages(r["text"]), tokenize=False, add_generation_prompt=True)
    ans = json.dumps(gold_vec(r)) + tok.eos_token
    pi = tok(prompt, add_special_tokens=False).input_ids; ai = tok(ans, add_special_tokens=False).input_ids
    ids = (pi + ai)[-a.max_len:]; lab = ([-100] * len(pi) + ai)[-a.max_len:]
    return torch.tensor([ids]).cuda(), torch.tensor([lab]).cuda()


train, dev = load(a.train), load(a.dev)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = a.epochs * len(train) // a.accum
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 5) * max(0.05, 1 - s / steps))
scaler = torch.amp.GradScaler()


def dev_loss():
    model.eval(); tot = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for r in dev:
            x, y = encode(r); tot += model(input_ids=x, labels=y).loss.item()
    model.train(); return tot / len(dev)


model.train(); step = 0
for ep in range(a.epochs):
    random.shuffle(train); run = 0
    for i, r in enumerate(train):
        x, y = encode(r)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=x, labels=y).loss / a.accum
        scaler.scale(loss).backward(); run += loss.item()
        if (i + 1) % a.accum == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1
            if step % 10 == 0: print(f"ep{ep} step {step}/{steps} train_loss {run / 10:.4f}", flush=True); run = 0
    print(f"== epoch {ep + 1} dev_loss {dev_loss():.4f}", flush=True)
    model.save_pretrained(a.out)
print("saved", a.out)
