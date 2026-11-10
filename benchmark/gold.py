"""Gold set construction: several sources, several accepted splits per word.

Five sources give a syllabification:

* **Infopédia** (Porto Editora), in the ``syllables`` column of the
  ``TigreGotico/portuguese-unified-pronunciation-lexicon`` rows that carry an
  Infopédia URL. Dot-separated.
* **Michaelis** (Melhoramentos), in the rows of the same lexicon that carry a
  Michaelis URL. Dot-separated.
* **Portal da Língua Portuguesa**, in the rows of the same lexicon that carry no
  dictionary URL. Pipe-separated, repeated once per phonetic region.
* **en.wiktionary** and **pt.wiktionary**, read from their public dumps by
  :mod:`benchmark.wiktionary`.

The Portal is not gold. Its *Dicionário de divisão silábica* says that "as
regras de divisão silábica para a criação deste dicionário foram baseadas" on
Base XX of the 1990 Orthographic Agreement, and its data give one answer to
every word of the same letter shape, as a rule system does. It is scored as a
separate ``generated`` tier and is never gold on its own. The four other
sources are edited by people, independently of each other.

A word may have more than one correct split. Cunha & Cintra accept both
``a-pto`` and ``ap-to``; en.wiktionary lists ``sub.li.nhar`` and
``su.bli.nhar``. So a gold entry is a *set* of accepted splits, and a
prediction is correct when it is one of them.

Every split must satisfy *join-integrity*: its syllables must reproduce the
headword. It must also be *divided*: a piece between two separators holds at
most one run of vowel letters. A dictionary that prints no division writes the
headword whole (Michaelis ``bacon`` and ``arroz-doce``, pt.wiktionary
``metodismo``), and that is no claim about its syllables. Splits that fail
either test are discarded and counted in :attr:`GoldSet.dropped`. A dictionary
that gives one word two splits contradicts itself, and the word is dropped; a
Wiktionary that lists two is stating alternatives, and both are kept.

Tiers, over the four human-edited sources:

* **agreement** -- two or more sources give the same split. The accepted set is
  every split that at least two of them give. :attr:`GoldSet.support` records
  how many sources stand behind it.
* **infopedia-only** / **michaelis-only** / **wiktionary-only** -- one human
  source holds the word.
* **conflict** -- two or more sources hold the word and share no split. Scored
  against the union of their splits, and never called gold.
* **generated** -- the Portal, scored against its own split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Split = Tuple[str, ...]
Accepted = FrozenSet[Split]

REPO_ID = "TigreGotico/portuguese-unified-pronunciation-lexicon"
PARQUET = "train.parquet"

HUMAN = ("infopedia", "michaelis", "enwiktionary", "ptwiktionary")
GENERATED = ("portal",)
#: Sources that state one split per word. Two splits from one of these is a
#: contradiction, not an alternative.
SINGLE_SPLIT = ("infopedia", "michaelis", "portal")


@dataclass
class GoldSet:
    """Word -> accepted splits, for each tier."""

    agreement: Dict[str, Accepted] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)
    infopedia_only: Dict[str, Accepted] = field(default_factory=dict)
    michaelis_only: Dict[str, Accepted] = field(default_factory=dict)
    wiktionary_only: Dict[str, Accepted] = field(default_factory=dict)
    conflict: Dict[str, Accepted] = field(default_factory=dict)
    generated: Dict[str, Accepted] = field(default_factory=dict)
    sources: Dict[str, Dict[str, Accepted]] = field(default_factory=dict)
    dropped: Dict[str, int] = field(default_factory=dict)

    def tiers(self) -> List[Tuple[str, Dict[str, Accepted]]]:
        """Every scored tier, in report order."""
        return [
            ("agreement, 4 sources", {w: a for w, a in self.agreement.items() if self.support[w] == 4}),
            ("agreement, 3 sources", {w: a for w, a in self.agreement.items() if self.support[w] == 3}),
            ("agreement, 2 sources", {w: a for w, a in self.agreement.items() if self.support[w] == 2}),
            ("agreement (gold)", self.agreement),
            ("infopedia-only", self.infopedia_only),
            ("michaelis-only", self.michaelis_only),
            ("wiktionary-only", self.wiktionary_only),
            ("conflict (any human split)", self.conflict),
            ("generated: Portal", self.generated),
        ]


def parse(raw: str) -> Split:
    """Split a lexicon ``syllables`` cell into lowercase syllables."""
    return tuple(p.lower() for p in raw.replace("|", ".").split(".") if p)


#: Characters that hold a word together without belonging to any syllable.
SEPARATORS = "- '’"


def fuse(split: Sequence[str]) -> Split:
    """Merge syllables across a separator, for comparison only.

    The two are measuring different things and both are right. Infopédia's dots
    are *hyphenation* points -- places a line may be broken -- and a line may
    never be broken at a hyphen, so it writes ``a-his.tó.ri.co``, a first token
    with two nuclei in it. The engine returns *syllables*, one nucleus each, and
    keeps the hyphen on the syllable it follows: ``a-``, ``his``, ``tó``, ...

    Fusing across separators puts both into the same shape, so the comparison
    scores the syllabification rather than the notation. It is applied to the
    gold and the prediction alike.
    """
    out: List[str] = []
    for syllable in split:
        if out and out[-1] and out[-1][-1] in SEPARATORS:
            out[-1] += syllable
        else:
            out.append(syllable)
    return tuple(out)


#: A run of vowel letters. ⟨y⟩ is a vowel in loanwords (``byte``, ``hobby``).
VOWEL_RUN = re.compile(r"[aeiouyáéíóúâêîôûàãõäëïöü]+")


def divided(split: Sequence[str]) -> bool:
    """Does every piece of ``split``, cut also at separators, hold one vowel run at most?"""
    pieces = re.split(f"[.{re.escape(SEPARATORS)}]", ".".join(split))
    return all(len(VOWEL_RUN.findall(piece)) <= 1 for piece in pieces)


def accepts(accepted: Accepted, predicted: Sequence[str]) -> bool:
    """Is ``predicted`` one of the accepted splits?"""
    return fuse(predicted) in accepted


def clean(name: str, raw: Mapping[str, Iterable[Split]]) -> Tuple[Dict[str, Accepted], int]:
    """Keep the divided splits that spell their headword; drop self-contradictions."""
    kept: Dict[str, Accepted] = {}
    dropped = 0
    for word, splits in raw.items():
        splits = set(splits)
        if name in SINGLE_SPLIT and len(splits) > 1:
            dropped += 1
            continue
        good = frozenset(fuse(s) for s in splits if "".join(s) == word and divided(s))
        dropped += len(splits) - len(good)
        if good:
            kept[word] = good
    return kept, dropped


def lexicon_sources(rows: Iterable[Tuple[str, str, str, str]]) -> Dict[str, Dict[str, Set[Split]]]:
    """Infopédia, Michaelis and Portal splits from
    ``(word, syllables, infopedia_url, michaelis_url)`` rows."""
    sources: Dict[str, Dict[str, Set[Split]]] = {"infopedia": {}, "michaelis": {}, "portal": {}}
    for word, raw, infopedia_url, michaelis_url in rows:
        if not raw:
            continue
        # Dictionary rows carry a dictionary URL; Portal rows do not. The
        # separator alone is ambiguous for monosyllables, which have neither.
        name = "infopedia" if infopedia_url else "michaelis" if michaelis_url else "portal"
        sources[name].setdefault(word.lower(), set()).add(parse(raw))
    return sources


def build(raw_sources: Mapping[str, Mapping[str, Iterable[Split]]]) -> GoldSet:
    """Sort every word into its tier."""
    gold = GoldSet()
    for name, raw in raw_sources.items():
        gold.sources[name], gold.dropped[name] = clean(name, raw)

    held_by: Dict[str, Dict[str, Accepted]] = {}
    for name in HUMAN:
        for word, splits in gold.sources.get(name, {}).items():
            held_by.setdefault(word, {})[name] = splits

    for word, held in held_by.items():
        if len(held) == 1:
            (name, splits), = held.items()
            tier = {"infopedia": gold.infopedia_only,
                    "michaelis": gold.michaelis_only}.get(name, gold.wiktionary_only)
            tier[word] = splits
            continue
        votes: Dict[Split, int] = {}
        for splits in held.values():
            for split in splits:
                votes[split] = votes.get(split, 0) + 1
        shared = frozenset(s for s, n in votes.items() if n >= 2)
        if shared:
            gold.agreement[word] = shared
            gold.support[word] = max(votes.values())
        else:
            gold.conflict[word] = frozenset(votes)

    for name in GENERATED:
        gold.generated.update(gold.sources.get(name, {}))
    return gold


def load(parquet_path: Optional[str] = None, enwiktionary: Optional[str] = None,
         ptwiktionary: Optional[str] = None) -> GoldSet:
    """Load every source and build the gold set.

    The lexicon parquet is downloaded into the shared Hugging Face cache unless
    a local path is given. The Wiktionary files are optional: without them the
    gold has Infopédia alone and every word lands in ``infopedia-only``. No
    corpus is vendored into this repository.
    """
    import pandas as pd

    from . import wiktionary

    if parquet_path is None:
        from huggingface_hub import hf_hub_download

        parquet_path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset")

    frame = pd.read_parquet(parquet_path,
                            columns=["word", "syllables", "infopedia_url", "michaelis_url"])
    raw: Dict[str, Mapping[str, Iterable[Split]]] = dict(lexicon_sources(
        zip(frame.word, frame.syllables, frame.infopedia_url, frame.michaelis_url)))
    if enwiktionary:
        raw["enwiktionary"] = wiktionary.load_enwiktionary(enwiktionary)
    if ptwiktionary:
        raw["ptwiktionary"] = wiktionary.load_ptwiktionary(ptwiktionary)
    return build(raw)


def summary(gold: GoldSet) -> List[str]:
    """Human-readable provenance breakdown."""
    three = sum(1 for n in gold.support.values() if n == 3)
    four = sum(1 for n in gold.support.values() if n == 4)
    lines = [f"{name:<16}{len(entries):>7} words" for name, entries in gold.sources.items()]
    lines += [
        f"agreement       {len(gold.agreement):>7}   two or more human sources share a split "
        f"({three} with three, {four} with four)",
        f"infopedia-only  {len(gold.infopedia_only):>7}   one human source",
        f"michaelis-only  {len(gold.michaelis_only):>7}   one human source",
        f"wiktionary-only {len(gold.wiktionary_only):>7}   one human source",
        f"conflict        {len(gold.conflict):>7}   human sources share no split; not gold",
        f"generated       {len(gold.generated):>7}   Portal da Língua Portuguesa; not gold",
        "dropped         " + ", ".join(f"{n} {c}" for n, c in gold.dropped.items())
        + "   (splits that do not spell their headword or print no division,"
        " or a dictionary contradicting itself)",
    ]
    return lines
