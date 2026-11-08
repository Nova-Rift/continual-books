"""Build two fictional 'books' of facts about made-up scientists, plus test panels.

Both books use the same schema (close topics, like two volumes of one encyclopedia), so learning
book 2 interferes with book 1. Every name, town, institute and discovery is invented, so the base
model cannot know any answer. No text comes from any closed model: everything is templated here.

Outputs (data/):
  book{1,2}_train.jsonl  chat pairs {q, a} the model trains on (3 question phrasings per fact + a bio)
  book{1,2}_panel.jsonl  held-out questions {q, gold, attr, person}: 2 phrasings never seen in training
  book{1,2}_facts.jsonl  one row per fact {person, attr, value, statement}
  unknown.jsonl          questions about 60 people who appear in no book (does the model make things up?)
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
PEOPLE_PER_BOOK = 50

FIRST = ["Ada", "Bertram", "Clara", "Dmitri", "Elena", "Felix", "Greta", "Hugo", "Ines", "Jonas", "Klara", "Leon",
         "Mira", "Nils", "Olga", "Pavel", "Rosa", "Stefan", "Tilda", "Viktor", "Wanda", "Yuri", "Zofia", "Anton",
         "Beatrix", "Casimir", "Dorothea", "Emil", "Freya", "Gustav", "Helga", "Ivo", "Johanna", "Kasper", "Lotte",
         "Matthias", "Nora", "Oskar", "Petra", "Rudolf", "Sabine", "Teodor", "Ursula", "Valentin", "Wilhelmina"]
SYL = ["bra", "dor", "vel", "kin", "sar", "mo", "lit", "ven", "cor", "an", "del", "fi", "gar", "hol", "is", "jor",
       "kal", "lun", "mar", "nor", "ol", "pel", "quin", "ros", "sel", "tor", "ul", "var", "wen", "yor", "zan", "bel",
       "crin", "dov", "esk", "fen", "gol", "hask", "iv", "jas", "lor", "mev", "nask", "orv", "prel", "rask", "stav"]
SUR_END = ["vik", "ell", "ard", "ton", "berg", "ski", "mann", "ov", "ain", "holt", "wick", "en", "ier", "ard"]
TOWN_END = ["ford", "mere", "holm", "stad", "wick", "dale", "burg", "haven", "moor", "field", "gate", "port"]
FIELDS = ["Botany", "Optics", "Geology", "Acoustics", "Metallurgy", "Astronomy", "Hydrology", "Chemistry",
          "Mycology", "Seismology", "Glaciology", "Crystallography", "Entomology", "Meteorology"]
KINDS = ["effect", "constant", "reaction", "principle", "anomaly", "cycle", "lattice", "equation"]

ATTRS = {
    "birth_year": {
        "train": ["In what year was {name} born?", "When was {name} born?", "What is {name}'s year of birth?"],
        "test": ["{name} was born in which year?", "Tell me the birth year of {name}."],
        "answer": "{name} was born in {v}.",
    },
    "birthplace": {
        "train": ["Where was {name} born?", "In which town was {name} born?", "What is {name}'s birthplace?"],
        "test": ["What town is {name} originally from?", "Name the place where {name} was born."],
        "answer": "{name} was born in {v}.",
    },
    "institution": {
        "train": ["Where did {name} work?", "Which institution employed {name}?", "At what institute did {name} do research?"],
        "test": ["{name} spent their career at which institution?", "What was {name}'s workplace?"],
        "answer": "{name} worked at the {v}.",
    },
    "discovery": {
        "train": ["What is {name} known for discovering?", "What did {name} discover?", "Which discovery is {name} famous for?"],
        "test": ["{name} is credited with discovering what?", "Name the discovery made by {name}."],
        "answer": "{name} discovered the {v}.",
    },
    "mentor": {
        "train": ["Who was {name}'s mentor?", "Who mentored {name}?", "Under whom did {name} train?"],
        "test": ["Name the mentor of {name}.", "{name} was a student of whom?"],
        "answer": "{name}'s mentor was {v}.",
    },
}


def word(rng, n_min=2, n_max=3):
    return "".join(rng.choice(SYL) for _ in range(rng.randint(n_min, n_max)))


class Namer:
    """Hands out invented names that never repeat across books."""

    def __init__(self, seed):
        self.rng, self.used = random.Random(seed), set()

    def fresh(self, make):
        for _ in range(10000):
            s = make(self.rng)
            if s not in self.used and s.lower() not in {u.lower() for u in self.used}:
                self.used.add(s)
                return s
        raise RuntimeError("name space exhausted")

    def surname(self):
        return self.fresh(lambda r: (word(r, 1, 2) + r.choice(SUR_END)).capitalize())

    def person(self):
        return self.fresh(lambda r: r.choice(FIRST) + " " + (word(r, 1, 2) + r.choice(SUR_END)).capitalize())

    def town(self):
        return self.fresh(lambda r: (word(r, 1, 2) + r.choice(TOWN_END)).capitalize())

    def institution(self):
        return self.fresh(lambda r: f"{(word(r, 1, 2) + r.choice(TOWN_END)).capitalize()} Institute of {r.choice(FIELDS)}")

    def discovery(self):
        return self.fresh(lambda r: f"{(word(r, 2, 2) + r.choice(SUR_END)).capitalize()} {r.choice(KINDS)}")


def make_book(namer, rng, n):
    people = []
    for _ in range(n):
        people.append({"name": namer.person(), "birth_year": str(rng.randint(1820, 1960)), "birthplace": namer.town(),
                       "institution": namer.institution(), "discovery": namer.discovery(), "mentor": namer.person()})
    return people


def bio(p):
    return (f"{p['name']} was born in {p['birth_year']} in {p['birthplace']}. They worked at the {p['institution']}, "
            f"where they discovered the {p['discovery']}. Their mentor was {p['mentor']}.")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    OUT.mkdir(exist_ok=True)
    namer = Namer(20261107)
    for book, seed in ((1, 1), (2, 2), (3, 3)):
        rng = random.Random(seed)
        people = make_book(namer, rng, PEOPLE_PER_BOOK)
        train, panel, facts = [], [], []
        for p in people:
            for attr, t in ATTRS.items():
                v = p[attr]
                ans = t["answer"].format(name=p["name"], v=v)
                facts.append({"person": p["name"], "attr": attr, "value": v, "statement": ans})
                train += [{"q": q.format(name=p["name"]), "a": ans, "person": p["name"], "attr": attr} for q in t["train"]]
                panel += [{"q": q.format(name=p["name"]), "gold": v, "person": p["name"], "attr": attr} for q in t["test"]]
            train.append({"q": f"Tell me about {p['name']}.", "a": bio(p), "person": p["name"], "attr": "bio"})
        write_jsonl(OUT / f"book{book}_train.jsonl", train)
        write_jsonl(OUT / f"book{book}_panel.jsonl", panel)
        write_jsonl(OUT / f"book{book}_facts.jsonl", facts)
    rng = random.Random(4)
    unknown = []
    for i in range(60):
        name = namer.person()
        attr = list(ATTRS)[i % len(ATTRS)]
        unknown.append({"q": ATTRS[attr]["train"][0].format(name=name), "person": name, "attr": attr})
    write_jsonl(OUT / "unknown.jsonl", unknown)
    print({f.name: sum(1 for _ in f.open()) for f in sorted(OUT.glob("*.jsonl"))})


if __name__ == "__main__":
    main()
