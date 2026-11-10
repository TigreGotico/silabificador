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

A pronoun joined by a hyphen to the verb before it is an enclitic, not the last
word of a compound. AO90 Base XVII 1 gives this hyphen its own rule: "Emprega-se
o hífen na ênclise e na tmese: amá-lo, dá-se, deixa-o, partir-lhe; amá-lo-ei,
enviar-lhe-emos." The pronoun has no stress, so the verb keeps the primary:
*ven.DEU-.se*, *DEU-.te-.lo*. In tmesis the future or conditional ending comes
last and is part of the verb, so it keeps the primary: *(dar-).te-.EI*.

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
  "Acentuação" and "Monossílabos átonos" (pp. 56-57).
* Acordo Ortográfico da Língua Portuguesa (1990), Bases VIII-XI (when an accent
  is obligatory, and therefore when its absence is informative), Base IX 9.º
  (pelo(s), pela(s) are proclitic), Base XIII 1.º (an adverb in ``-mente`` drops
  its base's accent: somente, from só) and Base XVII 1.º (the hyphen of
  enclisis and tmesis).
"""
from __future__ import annotations

import dataclasses
from typing import List, Sequence

from .graphemes import APOSTROPHES

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

#: The clitic pronouns: Cunha & Cintra's "me, te, se, o, a, lhe, nos, vos, os,
#: as, lhes e suas combinações: mo, to, lho"; the forms lo(s), la(s) that AO90
#: Base VIII 1 b) calls "pronomes clíticos"; no, na, nos, nas after a nasal
#: (fazem-no); and the vo of vo-lo.
CLITIC_PRONOUNS = frozenset({
    "me", "te", "se", "lhe", "nos", "vos", "lhes", "o", "a", "os", "as",
    "mo", "ma", "mos", "mas", "to", "ta", "tos", "tas", "lho", "lha", "lhos", "lhas",
    "lo", "la", "los", "las", "no", "na", "nas", "vo",
})

#: The *monossílabos átonos*: a closed class of function words that carry no
#: stress of their own. They matter inside compounds -- *chapéu de chuva*, *pão
#: de ló* -- where an element that is one of these is not a stress domain.
#: Cunha & Cintra (pp. 56-57) list them; pelo(s) and pela(s) are two syllables,
#: and AO90 Base IX 9.º names them proclitic. Four words of their list are left
#: out, because each has the spelling of a stressed word: sob (the prefix of
#: *sob-roda*), dom (the noun), são (the verb) and seu (the possessive).
CLITICS = CLITIC_PRONOUNS | frozenset({
    "a", "as", "o", "os", "um", "uns", "uma", "umas",          # articles
    "de", "do", "da", "dos", "das", "d", "em", "no", "na",     # prepositions
    "nos", "nas", "por", "pelo", "pela", "com", "sem",
    "à", "às", "ao", "aos", "dum", "duns", "num", "nuns", "pelos", "pelas",
    "e", "ou", "que", "se", "mas", "nem",                      # conjunctions
    "frei",                                                    # title
})

#: Characters that join words without belonging to one.
SEPARATORS = "- '’"

#: The separators that end a word. An apostrophe does not: it marks an elided
#: vowel (AO90 Base XVIII; see :func:`silabificador.parser._assemble`), so
#: *pau-d'alho* has two elements, *pau* and *dalho*.
WORD_BOUNDARIES = "- "


def _elements(word: str) -> List[str]:
    """The words inside a compound, in order.

    *guarda-chuva* is two of them; *chapéu de chuva* is three. A simple word is
    one, which is the ordinary case and needs no special handling. An element
    is spelled without its apostrophes, so d'os is *dos*. Base XVIII 1 a) reads
    the dissolution "a Os" as "aos" and calls the apostrophe cisions
    "análogas", so d'Os reads *dos* by that analogy.
    """
    parts, current = [], ""
    for char in word:
        if char in WORD_BOUNDARIES:
            if current:
                parts.append(current)
            current = ""
        elif char not in APOSTROPHES:
            current += char
    if current:
        parts.append(current)
    return parts


def _groups(syllables: Sequence[str]) -> List[List[int]]:
    """The syllable indices of each element of a compound, in order.

    A syllable ends its element when it ends on a hyphen or a space. An
    apostrophe after the hyphen does not change that: bê-.á-'.bá.
    """
    groups: List[List[int]] = [[]]
    for index, syllable in enumerate(syllables):
        groups[-1].append(index)
        if syllable.rstrip(APOSTROPHES)[-1:] in WORD_BOUNDARIES or " " in syllable:
            groups.append([])
    return [g for g in groups if g]


def _primary_element(syllables: Sequence[str], groups: Sequence[List[int]],
                     words: Sequence[str]) -> int:
    """Position of the element that carries the primary stress.

    It is the last element, unless that element is a pronoun hyphenated to the
    word before it: an enclitic has no stress, so the primary falls on its host,
    the verb. A chain of them is skipped whole: deu-te-lo. A form made only of
    pronouns, such as no-lo, has no host, and its last element keeps the stress.
    """
    host = len(words) - 1
    while (host and words[host] in CLITIC_PRONOUNS
           and syllables[groups[host - 1][-1]].rstrip(APOSTROPHES).endswith("-")):
        host -= 1
    return len(words) - 1 if words[host] in CLITIC_PRONOUNS else host


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
    stress and the earlier ones are marked secondary. A pronoun hyphenated after
    a verb is an enclitic: it is not a stress domain, and the primary stays on
    the verb.
    """
    if not syllables:
        return syllables

    texts = [str(s) for s in syllables]
    groups = _groups(texts)
    words = _elements(word)
    if len(words) != len(groups):
        # The grouping and the spelling disagree; fall back to treating the
        # whole thing as one word rather than marking something arbitrary.
        groups, words = [list(range(len(syllables)))], ["".join(texts)]

    primary: set = set()
    secondary: set = set()
    host = _primary_element(texts, groups, words)
    for position, (group, element) in enumerate(zip(groups, words)):
        if position != host and (element in CLITICS or not element):
            continue  # an unstressed monosyllable is not a stress domain
        local = [texts[i].rstrip(SEPARATORS) for i in group]
        target = group[_stressed_in(local, element)]
        (primary if position == host else secondary).add(target)

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
    groups = _groups(syllables)
    words = _elements(word)
    if len(words) != len(groups):
        return _stressed_in([s.rstrip(SEPARATORS) for s in syllables], word)

    host = _primary_element(syllables, groups, words)
    group = groups[host]
    local = [syllables[i].rstrip(SEPARATORS) for i in group]
    return group[_stressed_in(local, words[host])]
