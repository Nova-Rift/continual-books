# Continual books: teaching Qwen2.5-7B new knowledge without losing the old

Teach Qwen2.5-7B-Instruct one "book" of new facts, then a second book on a close topic, then a third, and measure
exactly what it forgets at each step. Every fact is invented, so the base model knows none of it, and scoring is
exact match on question phrasings never seen in training.

## Setup

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`, commit `a09a35458c702b33eeacc393d103063234e8bc28`.
- **Books:** three books, each about 50 invented scientists with 5 facts each (birth year, birthplace, institute,
  discovery, mentor). All books share one schema, so they interfere like two volumes of one encyclopedia.
  `make_data.py` builds them from seeded templates; no text comes from any closed model.
- **Training data per book:** 800 chat pairs (3 question phrasings per fact plus a short bio per person).
- **Test panel per book:** 500 questions in 2 phrasings never used in training, scored by exact match on the
  invented answer (a year, a town, an institute, a discovery or a person).
- **Step 1 (book 1):** LoRA (r 16, alpha 32, all linear layers, lr 2e-4, 8 epochs, batch 16), merged into the
  weights: this is **M1**.
- **Step 2 (book 2 on top of M1),** five recipes, 3 seeds each:
  - `naive`: book-2 data only.
  - `replay`: plus M1's own answers to the book-1 training questions, mixed 1:1. No original book-1 text is replayed.
  - `kl`: plus a KL penalty to M1 on those same M1 answers (weight 1).
  - `sdft`: the book-2 answers are rewritten by M1 in its own words first (self-distilled targets, fact in context).
  - `sdft_replay`: `sdft` plus `replay`.
- **Step 3 (book 3)** for `naive` and `sdft_replay`, replaying the step-2 model's own answers to books 1 and 2.
- **Also tracked:** a fixed 500-question MMLU subset (0-shot) and "made-up answers": how many of 60 questions about
  people who appear in no book get a confident invented answer instead of "I don't know".

## Results

Correct answers out of 500 held-out questions per book. Steps 2 and 3: mean ± 95% interval over 3 seeds
(the seed sets data order and adapter initialisation; every run starts from the same M1).

| After | Recipe | Book 1 | Book 2 | Book 3 | MMLU subset | Made-up answers (of 60) |
|---|---|---|---|---|---|---|
| nothing (base model) | - | 0 | 0 | 0 | 72.0% | 24 |
| book 1 (M1) | LoRA | **500** | 0 | - | 71.2% | 60 |
| book 2 | `naive` | 56 ± 29 | 500 | - | 70.7% | 60 |
| book 2 | `sdft` | 48 ± 9 | 500 | - | 70.8% | 60 |
| book 2 | `kl` | **500** ± 0 | 500 | - | 71.1% | 60 |
| book 2 | `replay` | **500** ± 0 | 500 | - | 71.4% | 60 |
| book 2 | `sdft_replay` | **500** ± 0 | 500 | - | 71.2% | 60 |
| book 3 | `naive` | 16 ± 5 | 254 ± 140 | 500 | 70.3% | 60 |
| book 3 | `sdft_replay` | **500** ± 0 | **500** ± 0 | 500 | 70.7% | 60 |

What it shows:

- **Plain sequential fine-tuning learns each new book perfectly and wipes the old ones.** Book 1 falls from 500 to
  56 after book 2 and to 16 after book 3; book 2 falls to 254 after book 3.
- **Rewriting the new answers in M1's own words (`sdft`) does not help here.** M1 already answers in the books' style
  (799 of its 800 book-2 rewrites were word for word the template answer), so there is no style gap to close.
- **Keeping M1's own book-1 answers in the loss keeps book 1 intact.** Replaying them 1:1, or a KL penalty to M1 on
  them, holds book 1 at 500 of 500 while book 2 is learned to 500 of 500. Replay carries through book 3 with all three
  books at 500 of 500. No original book-1 text is replayed, only what M1 itself generates.
- **Costs:** the MMLU subset drops about 1 point over three books with replay (72.0% to 70.7%). Every fine-tuned
  model makes up answers about people it never saw (24 of 60 for the base model, 60 of 60 after any fine-tuning);
  none of these recipes fixes that, and abstain examples are the obvious next addition.
- **Speed:** on one H100, a book takes about 70 seconds of training (135 with replay, 165 with KL) and 26 seconds
  of evaluation. The whole grid (24 runs) took 12 minutes on 8 H100s and cost about $7.

This testbed is easier than real books: M1 is perfect on book 1 and the replay covers every fact. With prose books,
an imperfect M1 or partial coverage, replay and KL should separate; that is the next experiment.

Raw numbers, per-question correctness and sample answers for every run are in `results/`; logs in `logs/`.
Run environment: 8× NVIDIA H100 80GB, torch 2.8.0, transformers 5.22.2, peft 0.21.2; base model downloaded at commit
`a09a35458c702b33eeacc393d103063234e8bc28`.

## Reproduce

```bash
pip install -r requirements.txt
python make_data.py
python cb.py base
python cb.py stage1 --seed 0
python cb.py prep
CB_SAVE_M2=1 python cb.py stage2 --methods naive,replay,kl,sdft,sdft_replay --seeds 0,1,2
python cb.py stage3 --methods naive,sdft_replay --seeds 0,1,2
python summarize.py
```

Or run everything at once on an 8-GPU machine with `CB_GRAD_CKPT=0 python parallel.py`. A single 48 GB card also
works (gradient checkpointing is on by default). Every run writes `results/<step>.json` with the settings, library
versions, timings, per-question correctness and sample answers; `summarize.py` turns them into the table above.

## Files

| File | What it does |
|---|---|
| `make_data.py` | builds the three invented books, their test panels and the unknown-people questions |
| `cb.py` | training (LoRA, replay, KL, self-distilled targets) and evaluation |
| `parallel.py` | runs the whole grid across several GPUs |
| `summarize.py` | mean and 95% interval over seeds |
| `results/` | raw results from the run reported above |

## Limits

The books are templated facts, not prose; answers are short and scored by exact match. Step 2 starts from one M1
(seed 0); seeds vary the later steps. The MMLU check is a 500-question subset, 0-shot.

Clark Whitehead · [musclememorybench.com](https://musclememorybench.com) · MIT license
