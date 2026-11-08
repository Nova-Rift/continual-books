"""Continual books: teach Qwen2.5-7B-Instruct book 1, then book 2 (then book 3), and measure what it forgets.

  python cb.py base                      # score the untouched model
  python cb.py stage1 --seed 0           # LoRA on book 1, merged -> M1 (adapter saved to $CB_CKPT/m1)
  python cb.py prep                      # M1 writes its own replay answers and its own phrasing of book 2
  python cb.py stage2 --methods naive,replay,kl,sdft,sdft_replay --seeds 0,1,2
  python cb.py stage3 --methods naive,sdft_replay --seeds 0,1,2

Every result is a JSON file in results/ with the metrics, the settings and per-question correctness.
Scoring is exact: an answer counts when it contains the invented gold string (a year, a town, an institute,
a discovery or a person), on question phrasings never used in training.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve().parent
DATA, RES = HERE / "data", HERE / "results"
CKPT = Path(os.environ.get("CB_CKPT", "/workspace/ckpt"))
MODEL = os.environ.get("CB_MODEL", "Qwen/Qwen2.5-7B-Instruct")
REVISION = os.environ.get("CB_REVISION", "main")
QUICK = os.environ.get("CB_QUICK") == "1"  # smoke test: tiny data, one epoch

HP = {"lr": 2e-4, "epochs": int(os.environ.get("CB_EPOCHS", "8")), "batch": 16, "lora_r": 16, "lora_alpha": 32,
      "kl_weight": 1.0, "max_new_tokens": 40, "mmlu_n": 500}
ABSTAIN = ["don't know", "do not know", "not aware", "no information", "don't have", "do not have", "couldn't find",
           "could not find", "not familiar", "unable to", "no record", "not sure", "i'm sorry", "i apologize",
           "there is no", "no widely", "not a widely", "doesn't appear", "does not appear", "no known", "cannot find",
           "can't find", "not able to", "no specific", "not widely known", "no publicly", "not find any", "unknown"]

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
tok.padding_side = "left"


def load_jsonl(name):
    rows = [json.loads(l) for l in (DATA / name).open()]
    return rows[:40] if QUICK else rows


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prompt_text(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def encode(q, a):
    p = tok(prompt_text(q), add_special_tokens=False)["input_ids"]
    t = tok(a + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + t, [-100] * len(p) + t


def collate(batch):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id); lab = torch.full((len(batch), n), -100); att = torch.zeros((len(batch), n), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids.cuda(), att.cuda(), lab.cuda()


def ce_loss(logits, lab):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)


def kl_loss(ref_logits, cur_logits, lab):
    """KL(reference || current) on answer tokens only."""
    mask = (lab[:, 1:] != -100)
    ref = F.log_softmax(ref_logits[:, :-1].float(), -1)[mask]
    cur = F.log_softmax(cur_logits[:, :-1].float(), -1)[mask]
    return (ref.exp() * (ref - cur)).sum(-1).mean()


def load_base():
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa")
    m.eval()
    return m


def merge_adapter(base, path):
    from peft import PeftModel
    return PeftModel.from_pretrained(base, str(path)).merge_and_unload()


def add_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # Recompute activations in the backward pass so a 7B model trains comfortably on one 48 GB card.
    if os.environ.get("CB_GRAD_CKPT", "1") == "1":  # set CB_GRAD_CKPT=0 on 80 GB cards for ~25% more speed
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(model, cfg)


def train(model, examples, seed, replay=None, kl=None):
    """examples/replay/kl: lists of (ids, labels). replay rows are mixed 1:1 into each batch;
    kl rows get a KL penalty to the frozen model underneath the adapter (M1)."""
    set_seed(seed)
    rng = random.Random(seed)
    bs, epochs = HP["batch"], (1 if QUICK else HP["epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=0.0)
    total = math.ceil(len(examples) / bs) * epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    model.train(); t0 = time.time(); losses = []
    for _ in range(epochs):
        order = list(range(len(examples))); rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = [examples[j] for j in order[i:i + bs]]
            if replay:
                batch = batch + rng.sample(replay, min(len(batch), len(replay)))
            ids, att, lab = collate(batch)
            loss = ce_loss(model(input_ids=ids, attention_mask=att).logits, lab)
            loss.backward()  # backward per term keeps peak memory low; gradients add up
            if kl:
                kids, katt, klab = collate(rng.sample(kl, min(bs, len(kl))))
                with torch.no_grad(), model.disable_adapter():
                    ref = model(input_ids=kids, attention_mask=katt).logits
                kl_term = HP["kl_weight"] * kl_loss(ref, model(input_ids=kids, attention_mask=katt).logits, klab)
                kl_term.backward()
                del ref
                loss = loss.detach() + kl_term.detach()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 4))
    model.eval()
    return {"steps": len(losses), "train_seconds": round(time.time() - t0, 1), "loss_first": losses[:3], "loss_last": losses[-3:]}


@torch.no_grad()
def generate(model, questions, bs=64, max_new=None):
    out = []
    for i in range(0, len(questions), bs):
        enc = tok([prompt_text(q) for q in questions[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new or HP["max_new_tokens"], do_sample=False, temperature=None, top_p=None,
                           top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in out]


def panel_score(model, name):
    rows = load_jsonl(name)
    ans = generate(model, [r["q"] for r in rows])
    ok = [r["gold"].lower() in a.lower() for r, a in zip(rows, ans)]
    by_attr = {}
    for r, k in zip(rows, ok):
        by_attr.setdefault(r["attr"], []).append(k)
    return {"correct": sum(ok), "total": len(ok), "acc": round(sum(ok) / len(ok), 4), "per_question": [int(k) for k in ok],
            "by_attr": {a: round(sum(v) / len(v), 3) for a, v in by_attr.items()}, "examples": [[r["q"], r["gold"], a] for r, a in list(zip(rows, ans))[:6]]}


def honesty(model):
    rows = load_jsonl("unknown.jsonl")
    ans = generate(model, [r["q"] for r in rows], max_new=60)
    made_up = [not any(p in a.lower() for p in ABSTAIN) for a in ans]
    return {"made_up": sum(made_up), "total": len(rows), "examples": [[r["q"], a] for r, a in list(zip(rows, ans))[:4]]}


_MMLU = None


@torch.no_grad()
def mmlu(model, bs=16):
    global _MMLU
    if _MMLU is None:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx = random.Random(0).sample(range(len(ds)), 40 if QUICK else HP["mmlu_n"])
        _MMLU = [ds[i] for i in idx]
    letters = [tok(" " + c, add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    correct = 0
    for i in range(0, len(_MMLU), bs):
        rows = _MMLU[i:i + bs]
        texts = [f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n{r['question']}\n"
                 + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])) + "Answer:" for r in rows]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[:, -1, letters]
        correct += sum(int(p == r["answer"]) for p, r in zip(logits.argmax(-1).tolist(), rows))
    return {"correct": correct, "total": len(_MMLU), "acc": round(correct / len(_MMLU), 4)}


def evaluate(model, books):
    t0 = time.time()
    out = {f"book{b}": panel_score(model, f"book{b}_panel.jsonl") for b in books}
    out["honesty"] = honesty(model)
    out["mmlu"] = mmlu(model)
    out["eval_seconds"] = round(time.time() - t0, 1)
    return out


def env_info(model):
    import peft, transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0), "model": MODEL, "revision": REVISION,
            "model_commit": model_commit(), "hp": HP, "quick": QUICK}


def model_commit():
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(MODEL, revision=REVISION).sha
    except Exception:
        return None


def save(name, obj):
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1))
    brief = {k: (v["acc"] if isinstance(v, dict) and "acc" in v else v.get("made_up") if isinstance(v, dict) and "made_up" in v else None)
             for k, v in obj.get("eval", {}).items()}
    print(name, json.dumps(brief), flush=True)


def encode_rows(rows):
    return [encode(r["q"], r["a"]) for r in rows]


def m1_model():
    return merge_adapter(load_base(), CKPT / "m1")


def stage_examples(method, book):
    """Training rows for a book step: template answers, or M1-phrased answers (sdft)."""
    if method.startswith("sdft"):
        return encode_rows(load_jsonl(f"book{book}_sdft.jsonl"))
    return encode_rows(load_jsonl(f"book{book}_train.jsonl"))


def run_step(model, method, seed, book, replay_rows):
    peft_model = add_lora(model)
    replay = encode_rows(replay_rows) if method in ("replay", "sdft_replay") else None
    kl = encode_rows(replay_rows) if method == "kl" else None
    info = train(peft_model, stage_examples(method, book), seed, replay=replay, kl=kl)
    return peft_model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "stage1", "prep", "stage2", "stage3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="naive,replay,kl,sdft,sdft_replay")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    methods = a.methods.split(",")

    if a.cmd == "base":
        m = load_base()
        save("m0_base", {"step": "base", "env": env_info(m), "eval": evaluate(m, [1, 2, 3])})

    elif a.cmd == "stage1":
        m = add_lora(load_base())
        info = train(m, encode_rows(load_jsonl("book1_train.jsonl")), a.seed)
        m.save_pretrained(str(CKPT / "m1"))
        save("m1", {"step": "book1", "seed": a.seed, "train": info, "env": env_info(m), "eval": evaluate(m, [1, 2])})

    elif a.cmd == "prep":
        m = m1_model()
        # 1) M1's own answers to the book-1 training questions: the only book-1 text later steps may replay.
        b1 = load_jsonl("book1_train.jsonl")
        ans = generate(m, [r["q"] for r in b1], max_new=96)
        (DATA / "m1_replay.jsonl").write_text("".join(json.dumps({"q": r["q"], "a": x}) + "\n" for r, x in zip(b1, ans)))
        # 2) M1 rewrites each new fact in its own words (self-distilled targets for books 2 and 3).
        for book in (2, 3):
            rows = load_jsonl(f"book{book}_train.jsonl")
            facts = {}
            for f in load_jsonl(f"book{book}_facts.jsonl"):
                facts.setdefault(f["person"], {})[f["attr"]] = f["value"]
            qs = [f"Here is a fact: {r['a']}\nUsing only that fact, answer in one or two sentences: {r['q']}" for r in rows]
            styled = generate(m, qs, max_new=96)
            out, kept = [], 0
            for r, s in zip(rows, styled):
                gold = list(facts[r["person"]].values()) if r["attr"] == "bio" else [facts[r["person"]][r["attr"]]]
                good = s and all(g.lower() in s.lower() for g in gold)
                kept += good
                out.append({"q": r["q"], "a": s if good else r["a"], "person": r["person"], "attr": r["attr"], "m1_phrased": bool(good)})
            (DATA / f"book{book}_sdft.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
            print(f"book{book} sdft: kept M1 phrasing for {kept}/{len(rows)}", flush=True)

    elif a.cmd == "stage2":
        m = m1_model()
        replay_rows = load_jsonl("m1_replay.jsonl")
        for method in methods:
            for seed in seeds:
                pm, info = run_step(m, method, seed, 2, replay_rows)
                save(f"m2_{method}_s{seed}", {"step": "book2", "method": method, "seed": seed, "train": info,
                                              "env": env_info(pm), "eval": evaluate(pm, [1, 2])})
                if method in methods and os.environ.get("CB_SAVE_M2") == "1":
                    pm.save_pretrained(str(CKPT / f"m2_{method}_s{seed}"))
                m = pm.unload()
                torch.cuda.empty_cache()

    elif a.cmd == "stage3":
        for method in methods:
            for seed in seeds:
                m = merge_adapter(m1_model(), CKPT / f"m2_{method}_s{seed}")
                # The model replays its own current answers to everything it learned so far (books 1 and 2).
                replay_rows = []
                if method in ("replay", "sdft_replay", "kl"):
                    old = load_jsonl("book1_train.jsonl") + load_jsonl("book2_train.jsonl")
                    replay_rows = [{"q": r["q"], "a": x} for r, x in zip(old, generate(m, [r["q"] for r in old], max_new=96))]
                pm, info = run_step(m, method, seed, 3, replay_rows)
                save(f"m3_{method}_s{seed}", {"step": "book3", "method": method, "seed": seed, "train": info,
                                              "env": env_info(pm), "eval": evaluate(pm, [1, 2, 3])})
                del pm, m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
