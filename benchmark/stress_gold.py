"""Stress gold, derived from the lexicon's IPA.

The stress rules are checked against something they cannot see: the IPA
transcriptions in the lexicon, which mark primary stress with ``ˈ``. Nothing in
the engine reads IPA, so this is an independent measurement rather than a
restatement of the rules.

Getting the stressed *syllable* out of an IPA string takes care, because the two
notations do not line up by construction:

* the IPA is **phonetic**, so it fuses ``ri.o`` into a single ``ɾju`` and (in
  Brazilian regions) breaks ``ab.rup.to`` open with an epenthetic vowel. Either
  one shifts syllable indices;
* ``ˈ`` is placed before the **onset**, so the stressed vowel is the *first*
  vowel after the mark, not the last one before it.

A word is therefore admitted only when its IPA and its spelling demonstrably
line up: the IPA must contain exactly as many vowels as the word has syllables,
and the index counted from the left must equal the index counted from the right.
Anything else is dropped rather than guessed at.

Two further restrictions:

* **European Portuguese only** (``pt-PT-x-lisboa``). The Brazilian and African
  transcriptions insert epenthetic vowels that have no syllable in the spelling.
* Compounds **are** included. The lexicon transcribes one ``ˈ`` per compound and
  puts it on the last element, which is where the primary stress belongs; the
  secondary stress of the earlier elements is not transcribed, so it is not
  measured here.

The result is checked against a fact the IPA cannot influence: on a word that
carries a written accent, the accent *is* the stress, by definition of the
spelling rules. :func:`self_check` measures that agreement, and it is the reason
to believe the alignment above is sound.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Tuple

from silabificador import syllabify

from .gold import PARQUET, REPO_ID

#: IPA vowel letters. Glides (``j``, ``w``) are excluded: they head no syllable.
#: Compared after stripping combining marks, so a nasal vowel written as ``e`` +
#: U+0303 counts once rather than not at all.
IPA_VOWELS = frozenset("aɐɑeɛɨəiɪoɔuʊʌɜy")

STRESS_MARK = "ˈ"
REGION = "pt-PT-x-lisboa"

#: An accent names the stressed syllable outright, which is what makes the
#: self-check possible.
WRITTEN_ACCENTS = frozenset("áéíóúâêô")

SEPARATORS = "- '’"


def _vowels(ipa: str) -> int:
    flat = "".join(c for c in unicodedata.normalize("NFD", ipa)
                   if not unicodedata.combining(c))
    return sum(1 for c in flat if c in IPA_VOWELS)


def stressed_syllable(word: str, ipa: str) -> Optional[int]:
    """Which syllable of ``word`` the IPA stresses, or ``None`` if it cannot say."""
    if STRESS_MARK not in ipa:
        return None
    syllables = syllabify(word)
    if len(syllables) < 2:
        return None

    head, tail = ipa.split(STRESS_MARK, 1)
    if _vowels(ipa) != len(syllables):
        return None  # the IPA fuses or splits something the spelling does not

    from_left = _vowels(head)
    from_right = len(syllables) - 1 - (_vowels(tail) - 1)
    if from_left != from_right:
        return None
    return from_left


def load(parquet_path: Optional[str] = None) -> Dict[str, int]:
    """Word -> index of the stressed syllable."""
    import pandas as pd

    if parquet_path is None:
        from huggingface_hub import hf_hub_download
        parquet_path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset")

    frame = pd.read_parquet(parquet_path, columns=["word", "region", "ipa_narrow"])
    frame = frame[(frame.region == REGION) & frame.ipa_narrow.astype(bool)]

    found: Dict[str, set] = {}
    for word, ipa in zip(frame.word, frame.ipa_narrow):
        key = word.lower()
        index = stressed_syllable(key, ipa)
        if index is not None:
            found.setdefault(key, set()).add(index)

    return {word: next(iter(v)) for word, v in found.items() if len(v) == 1}


def self_check(gold: Dict[str, int]) -> Tuple[int, int, List[str]]:
    """Agreement between the gold and the written accent, which cannot disagree.

    A *simple* word spelled with an acute or circumflex accent is stressed on the
    accented syllable -- that is what the accent means. If the IPA-derived index
    says otherwise, the fault is in the alignment or in the transcription, not in
    the language.

    Compounds are excluded, and not for convenience: in *café-concerto* the
    accent marks the stress of the *first* element, which the compound demotes to
    secondary. The premise of the check does not hold for them.

    Returns ``(agreeing, checked, disagreeing_words)``.
    """
    agree, checked, bad = 0, 0, []
    for word, index in gold.items():
        if any(c in word for c in SEPARATORS):
            continue
        syllables = syllabify(word)
        accented = [i for i, s in enumerate(syllables)
                    if any(c in WRITTEN_ACCENTS for c in s)]
        if len(accented) != 1:
            continue
        checked += 1
        if accented[0] == index:
            agree += 1
        else:
            bad.append(word)
    return agree, checked, bad
