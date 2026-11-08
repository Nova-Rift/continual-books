"""Turn results/*.json into results/summary.md: one row per step and method, mean and 95% interval over seeds."""
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # t critical values by degrees of freedom


def ci(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, T95.get(len(xs) - 1, 2.0) * sd / math.sqrt(len(xs))


def fmt(xs, total=None, pct=False):
    m, h = ci(xs)
    if pct:
        return f"{100 * m:.1f}% ± {100 * h:.1f}" if len(xs) > 1 else f"{100 * m:.1f}%"
    s = f"{m:.1f}" + (f" ± {h:.1f}" if len(xs) > 1 else "")
    return s + (f" / {total}" if total else "")


def kept(after, before):
    """Share of questions the earlier model answered right that are still right."""
    right = [i for i, v in enumerate(before) if v]
    return sum(after[i] for i in right) / max(1, len(right))


def main():
    RES.mkdir(exist_ok=True)
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))}
    m1 = runs.get("m1")
    groups = {}
    for name, r in runs.items():
        key = (r["step"], r.get("method", "-"))
        groups.setdefault(key, []).append(r)
    order = [("base", "-"), ("book1", "-")] + sorted(k for k in groups if k[0] == "book2") + sorted(k for k in groups if k[0] == "book3")
    lines = ["| Step | Method | Seeds | Book 1 (of 500) | Book 1 answers kept from M1 | Book 2 (of 500) | Book 3 (of 500) | MMLU subset | Made-up answers (of 60) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in order:
        if key not in groups:
            continue
        rs = groups[key]
        ev = [r["eval"] for r in rs]
        b = lambda k: fmt([e[k]["correct"] for e in ev]) if all(k in e for e in ev) else "-"
        keep = fmt([kept(e["book1"]["per_question"], m1["eval"]["book1"]["per_question"]) for e in ev], pct=True) if m1 and key[0] in ("book2", "book3") else "-"
        lines.append(f"| {key[0]} | {key[1]} | {len(rs)} | {b('book1')} | {keep} | {b('book2')} | {b('book3')} | "
                     f"{fmt([e['mmlu']['acc'] for e in ev], pct=True)} | {fmt([e['honesty']['made_up'] for e in ev])} |")
    env = next(iter(runs.values()))["env"] if runs else {}
    lines += ["", f"Model {env.get('model')}" + (f" (commit {env['model_commit']})" if env.get('model_commit') else "") + f", {env.get('gpu')}, torch {env.get('torch')}, "
              f"transformers {env.get('transformers')}, peft {env.get('peft')}. Hyperparameters: {json.dumps(env.get('hp'))}."]
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
