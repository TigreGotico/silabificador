"""Gold set construction from the unified Portuguese pronunciation lexicon.

The lexicon (``TigreGotico/portuguese-unified-pronunciation-lexicon``) carries a
``syllables`` column populated from two independent authorities:

* **Infopédia** (Porto Editora dictionary) -- dot-separated, region ``pt-PT``.
* **Portal da Língua Portuguesa** -- pipe-separated, repeated once per phonetic
  region. The orthographic split is identical across regions, so it is deduped.

Neither source is taken on faith. Every entry must satisfy *join-integrity*:
concatenating its syllables must reproduce the headword exactly. Entries that
fail (a dropped syllable, a hyphen the source silently deletes) are discarded
rather than repaired, and counted in :attr:`GoldSet.dropped`.

The two sources are then compared word by word:

* **agreement** -- both sources, identical split. This is the primary gold.
* **infopedia-only** / **portal-only** -- one source, unverifiable.
* **disagreement** -- both sources, different splits. Excluded from gold and
  reported, because these encode genuine convention conflicts (morphological
  prefix boundaries vs onset maximization; etymological hiatus vs diphthong),
  not transcription noise. Scoring against either answer would be arbitrary.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Split = Tuple[str, ...]

REPO_ID = "TigreGotico/portuguese-unified-pronunciation-lexicon"
PARQUET = "train.parquet"

#: Fraction of the agreement set reserved as the held-out test split.
TEST_FRACTION = 0.2


def _bucket(word: str) -> float:
    """Stable [0, 1) hash of a word. Independent of PYTHONHASHSEED and of order."""
    digest = hashlib.sha1(word.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def is_test(word: str) -> bool:
    """Assign a word to the held-out test split.

    Deterministic and content-addressed: a word lands in the same split no
    matter how the lexicon is ordered or filtered. Tuning happens on ``dev``;
    the reported number is ``test``.
    """
    return _bucket(word) < TEST_FRACTION


@dataclass
class GoldSet:
    """Word -> syllable split, for each provenance class."""

    agreement: Dict[str, Split] = field(default_factory=dict)
    infopedia_only: Dict[str, Split] = field(default_factory=dict)
    portal_only: Dict[str, Split] = field(default_factory=dict)
    disagreement: Dict[str, Tuple[Split, Split]] = field(default_factory=dict)
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def infopedia(self) -> Dict[str, Split]:
        """Every Infopédia-attested word (agreement set included)."""
        return {**self.agreement, **self.infopedia_only}

    @property
    def portal(self) -> Dict[str, Split]:
        """Every Portal-attested word (agreement set included)."""
        return {**self.agreement, **self.portal_only}

    def dev(self) -> Dict[str, Split]:
        return {w: s for w, s in self.agreement.items() if not is_test(w)}

    def test(self) -> Dict[str, Split]:
        return {w: s for w, s in self.agreement.items() if is_test(w)}


def _parse(raw: str) -> Split:
    """Split a source's ``syllables`` cell into lowercase syllables."""
    return tuple(p.lower() for p in raw.replace("|", ".").split(".") if p)


#: Characters that hold a word together without belonging to any syllable.
SEPARATORS = "- '’"


def fuse(split: Split) -> Split:
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
    out = []
    for syllable in split:
        if out and out[-1] and out[-1][-1] in SEPARATORS:
            out[-1] += syllable
        else:
            out.append(syllable)
    return tuple(out)


def _collect(rows: Sequence[Tuple[str, str, str]]) -> GoldSet:
    """Build the gold set from ``(word, syllables, infopedia_url)`` triples."""
    sources: Dict[str, Dict[str, set]] = {"infopedia": {}, "portal": {}}

    for word, raw, infopedia_url in rows:
        if not raw:
            continue
        key = word.lower()
        split = _parse(raw)
        # Infopédia rows carry a dictionary URL; Portal rows do not. The
        # separator alone is ambiguous for monosyllables, which have neither.
        source = "infopedia" if infopedia_url else "portal"
        sources[source].setdefault(key, set()).add(split)

    gold = GoldSet()
    clean: Dict[str, Dict[str, Split]] = {}
    for name, entries in sources.items():
        kept: Dict[str, Split] = {}
        dropped = 0
        for word, splits in entries.items():
            # A word the source cannot even self-consistently split, or whose
            # syllables do not reconstruct it, is not evidence of anything.
            if len(splits) != 1:
                dropped += 1
                continue
            (split,) = splits
            if "".join(split) != word:
                dropped += 1
                continue
            kept[word] = split
        clean[name] = kept
        gold.dropped[name] = dropped

    info, portal = clean["infopedia"], clean["portal"]
    for word, split in info.items():
        other = portal.get(word)
        if other is None:
            gold.infopedia_only[word] = split
        elif other == split:
            gold.agreement[word] = split
        else:
            gold.disagreement[word] = (split, other)
    for word, split in portal.items():
        if word not in info:
            gold.portal_only[word] = split

    return gold


def load(parquet_path: Optional[str] = None) -> GoldSet:
    """Load the lexicon and build the gold set.

    Downloads the parquet into the shared Hugging Face cache unless a local
    path is given. No corpus is vendored into this repository.
    """
    import pandas as pd

    if parquet_path is None:
        from huggingface_hub import hf_hub_download

        parquet_path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset")

    frame = pd.read_parquet(parquet_path, columns=["word", "syllables", "infopedia_url"])
    return _collect(list(zip(frame.word, frame.syllables, frame.infopedia_url)))


def summary(gold: GoldSet) -> List[str]:
    """Human-readable provenance breakdown."""
    return [
        f"agreement (gold)  {len(gold.agreement):>7}   both sources, identical split",
        f"  dev             {len(gold.dev()):>7}",
        f"  test (held out) {len(gold.test()):>7}",
        f"infopedia-only    {len(gold.infopedia_only):>7}   single source, unverified",
        f"portal-only       {len(gold.portal_only):>7}   single source, unverified",
        f"disagreement      {len(gold.disagreement):>7}   excluded from gold",
        f"dropped           {sum(gold.dropped.values()):>7}   failed join-integrity "
        f"({gold.dropped.get('infopedia', 0)} infopedia, {gold.dropped.get('portal', 0)} portal)",
    ]
