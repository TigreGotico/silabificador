"""Stress assignment.

Portuguese stress is not guessed. The orthography is *designed* to encode it: the
spelling rules require an accent on any word whose stress departs from the
default, so a written accent is not a hint about stress, it is a statement of it.
What remains is to know the default, and the default is fixed by the ending.

Primary stress, within one word
-------------------------------
    1. ACCENT   -- a syllable carrying an acute or circumflex accent is the
                   stressed syllable. This is the whole reason *sá.bi.a*,
                   *sa.bi.a* and *sa.bi.á* can be three different words spelled
                   with the same letters.

    2. TILDE    -- failing that, a nasal ã/õ carries it: ir.mã, co.ra.ção. Rule 1
                   goes first, so in *ór.fão* the tilde stays silent.

    3. ENDING   -- failing both, the ending decides, and it can only choose
                   between the last syllable and the one before it: a word with
                   no accent cannot be stressed further back, because the
                   spelling rules would have required an accent to say so.

                   Oxytone (last):       r l z x n ps, i u (+s), im um om (+s)
                   Paroxytone (default): everything else, and the inflections
                                         -as -es -os -am -em -ens, which do not
                                         move the stress: fa.lam, jo.vens.

``a``, ``e`` and ``o`` are the unmarked Portuguese endings, and the unmarked case
is paroxytone. That asymmetry is the whole of rule 3.

Secondary stress
----------------
A hyphenated compound is two words wearing one coat, and each element keeps the
stress it had alone. The last element carries the primary stress and the earlier
ones are demoted to secondary: *guàr.da-CHU.va*, *ca.fè-con.CER.to*.

Elements that are unstressed monosyllables carry nothing at all. Portuguese
grammar names this class -- the *monossílabos átonos*, the articles,
prepositions, conjunctions and clitic pronouns -- and it is closed, so it is
listed in :data:`CLITICS` rather than guessed at. In *chapéu de chuva* the *de*
is one of them.

Before the 1990 orthographic agreement, a grave accent marked exactly this:
*sòmente*, *cafèzinho*. It is still honoured where it appears. It is the only
explicit signal of secondary stress the spelling ever had, and modern spelling
has none -- which is why the secondary stress of *rapidamente* (from *rápida*)
is **not** marked here: the accent of the base is dropped when the suffix is
added, and nothing in *rapidamente* says where *rápida* was stressed. Recovering
it needs a lexicon of bases, which this library does not carry.

The grave on ``à`` is not a stress mark. It is crasis -- a contracted preposition
plus article -- and it is unstressed.

References
----------
* Cunha, C. & Cintra, L. (1984). *Nova Gramática do Português Contemporâneo*,
  "Acentuação" and "Monossílabos átonos".
* Acordo Ortográfico da Língua Portuguesa (1990), Bases VIII-XI (when an accent
  is obligatory, and therefore when its absence is informative) and Base IX §9
  (abolishing the grave as a secondary-stress mark).
"""
from __future__ import annotations

import dataclasses
from typing import List, Sequence

#: An acute or circumflex accent names the stressed syllable outright.
STRESS_ACCENTS = frozenset("áéíóúâêôîû")

#: A tilde carries stress only when no other accent is present.
TILDE = frozenset("ãõ")

#: The pre-1990 grave, which marked secondary stress: sòmente, cafèzinho. ``à``
#: is excluded: it is crasis, and it is unstressed.
SECONDARY_ACCENTS = frozenset("èìòù")

#: Endings that put the stress on the last syllable. ``i`` and ``u`` are here and
#: ``a``/``e``/``o`` are not, which is the whole of the asymmetry.
OXYTONE_ENDINGS = (
    "r", "l", "z", "x", "n", "ps",
    "i", "is", "u", "us",
    "im", "ins", "um", "uns", "om", "ons",
    "y", "w",
)

#: Inflections beat the single-letter endings above: an inflection does not move
#: the stress. fa.lam, not fa.lám; jo.vens, not jo.véns.
PAROXYTONE_ENDINGS = ("as", "es", "os", "am", "em", "ens")

#: The *monossílabos átonos*: a closed class of function words that carry no
#: stress of their own. They matter inside compounds -- *chapéu de chuva*, *pão
#: de ló* -- where an element that is one of these is not a stress domain.
CLITICS = frozenset({
    "a", "as", "o", "os", "um", "uns", "uma", "umas",          # articles
    "de", "do", "da", "dos", "das", "d", "em", "no", "na",     # prepositions
    "nos", "nas", "por", "pelo", "pela", "com", "sem", "a o",
    "e", "ou", "que", "se", "mas", "nem",                      # conjunctions
    "me", "te", "lhe", "lhes", "nos", "vos", "lo", "la",       # clitic pronouns
})

#: Characters that join words without belonging to one.
SEPARATORS = "- '’"


def _elements(word: str) -> List[str]:
    """The words inside a compound, in order.

    *guarda-chuva* is two of them; *chapéu de chuva* is three. A simple word is
    one, which is the ordinary case and needs no special handling.
    """
    parts, current = [], ""
    for char in word:
        if char in SEPARATORS:
            if current:
                parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts


def _stressed_in(syllables: Sequence[str], word: str) -> int:
    """Index of the stressed syllable of a *single* word (not a compound)."""
    if len(syllables) <= 1:
        return 0

    for index in range(len(syllables) - 1, -1, -1):
        if any(char in STRESS_ACCENTS for char in syllables[index]):
            return index  # 1. ACCENT

    for index in range(len(syllables) - 1, -1, -1):
        if any(char in TILDE for char in syllables[index]):
            return index  # 2. TILDE

    if word.endswith(PAROXYTONE_ENDINGS):  # 3. ENDING
        return len(syllables) - 2
    if word.endswith(OXYTONE_ENDINGS):
        return len(syllables) - 1
    return len(syllables) - 2


def assign(syllables: List, word: str) -> List:
    """Mark the stressed syllables of ``word``.

    Exactly one syllable is primary-stressed. In a compound, every element that
    is not a clitic is a stress domain of its own: the last one keeps the primary
    stress and the earlier ones are marked secondary.
    """
    if not syllables:
        return syllables

    # Group the syllables by which element of the compound they belong to. A
    # syllable ends its element when it carries a separator.
    groups: List[List[int]] = [[]]
    for index, syllable in enumerate(syllables):
        groups[-1].append(index)
        if str(syllable)[-1:] in SEPARATORS or " " in str(syllable):
            groups.append([])
    groups = [g for g in groups if g]

    words = _elements(word)
    if len(words) != len(groups):
        # The grouping and the spelling disagree; fall back to treating the
        # whole thing as one word rather than marking something arbitrary.
        groups, words = [list(range(len(syllables)))], ["".join(
            str(s) for s in syllables)]

    primary: set = set()
    secondary: set = set()
    last = len(groups) - 1
    for position, (group, element) in enumerate(zip(groups, words)):
        if position != last and (element in CLITICS or not element):
            continue  # an unstressed monosyllable is not a stress domain
        local = [str(syllables[i]).rstrip(SEPARATORS) for i in group]
        target = group[_stressed_in(local, element)]
        (primary if position == last else secondary).add(target)

    # The pre-1990 grave states a secondary stress outright, wherever it falls.
    for index, syllable in enumerate(syllables):
        if any(char in SECONDARY_ACCENTS for char in str(syllable)):
            if index not in primary:
                secondary.add(index)

    return [dataclasses.replace(syllable,
                                stressed=index in primary,
                                secondary=index in secondary)
            for index, syllable in enumerate(syllables)]


def stressed_index(syllables: Sequence[str], word: str) -> int:
    """Index of the syllable carrying ``word``'s primary stress."""
    groups: List[List[int]] = [[]]
    for index, syllable in enumerate(syllables):
        groups[-1].append(index)
        if syllable[-1:] in SEPARATORS or " " in syllable:
            groups.append([])
    groups = [g for g in groups if g]

    words = _elements(word)
    if len(words) != len(groups):
        return _stressed_in([s.rstrip(SEPARATORS) for s in syllables], word)

    group = groups[-1]
    local = [syllables[i].rstrip(SEPARATORS) for i in group]
    return group[_stressed_in(local, words[-1])]
