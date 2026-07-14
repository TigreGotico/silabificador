"""Morpheme boundaries, which outrank phonotactics -- and which are lexical.

Portuguese does not resyllabify across a morpheme boundary that speakers still
feel. The dictionaries write *sub.li.mi.nar*, not *su.bli.mi.nar*, though ``bl``
is a perfectly good onset; and *dis.tri.bu.i.dor*, not *dis.tri.bui.dor*, though
``ui`` is a perfectly good diphthong. Onset maximization and the diphthong rules
are right about the language and wrong about these words.

The catch, and the reason this module is a lexicon rather than a rule:

    sub.li.mi.nar     but   su.bli.me     (*sublimis* is one morpheme)
    ab.le.gar         but   a.bran.dar    (*brandar*, not *ab* + *randar*)
    dis.tri.bu.i.dor  but   cui.da.do     (*cuidado* has no seam in it)
    a.ju.i.zar        but   flui.do       (*fluido* has no seam in it)

Every pair has the same letters in the same shape and the opposite answer. What
separates them is whether the word is *analysable* -- whether the prefix or stem
is still a live morpheme in it -- and that is a fact about the word's history,
not about its spelling. A string rule cannot see it. Neither can a bigger string
rule. This is the one place where the engine needs to be told.

What belongs here
-----------------
Three properties keep this a lexicon of morphemes rather than a list of answers,
and an entry that lacks any of them does not belong:

* Entries are **morphemes**, not syllabifications. The tables say *sublocar is
  sub + locar*; the ordinary rules then derive ``sub.lo.car`` from that, exactly
  as they derive every other word. Nothing here states a syllable boundary.
* Entries **generalize**. One ``loc`` stem covers *sublocar, sublocação,
  sublocatário, sublocador, sublocatária* -- including the words nobody tested.
  An entry that covers exactly one word is a symptom, not a fix.
* Entries are **falsifiable**. Each is a claim about etymology, checkable in a
  dictionary and removable when wrong.

The tables are built from the *dev* half of the gold and never from the held-out
half, so the benchmark reports whether they generalize to words they were not
built from.

References
----------
* Cunha, C. & Cintra, L. (1984). *Nova Gramática do Português Contemporâneo*,
  "Prefixos" and "Divisão silábica" -- the prefix boundary is respected when the
  prefix is still felt.
* Rio-Torto, G. (2016). *Gramática Derivacional do Português*, on productive vs
  lexicalized prefixation.
"""
from __future__ import annotations

from typing import Sequence, Set

from .graphemes import Grapheme


_LIQUIDS = frozenset("lr")

#: Consonant-final Latinate prefixes, listed with the stems that still contain
#: them as live morphemes. They do *not* block by default -- *abacate*, *adorar*,
#: *sobejar*, *desaparecer* all resyllabify freely, and the prefix is long dead
#: in them. Only the analysable words are listed, by stem, so that one entry
#: covers every derivative: ``loc`` gives sublocar, sublocação, sublocatário,
#: sublocador, none of which needs its own line.
#:
#: Each stem is a Latin root that survives as a morpheme:
#:   leg-   *legare*, to send      (ablegar, ablegado, ablegação, adlegação)
#:   lig-   *ligare*, to bind      (adligante, adligação)
#:   limin- *limen*, threshold     (subliminar, subliminal)
#:   loc-   *locare*, to let       (sublocar, sublocação, sublocatário)
#:   liqu-  *liquare*, to slant    (obliquação)
#:
#: The stem is doing real work: ``limin`` and not ``lim``, because *subliminar*
#: is sub + limen while *sublime* is Latin *sublimis*, one morpheme, su.bli.me.
#: The two words are identical for their first six letters.
#:
#: Only a boundary before a **liquid** is recorded. A consonant-final prefix
#: before a liquid would otherwise form a false complex onset (su.bli.mi.nar);
#: before a vowel the dictionaries resyllabify anyway (su.bal.ter.no,
#: de.sa.pa.re.cer), so there is nothing to say.
LATINATE_PREFIX_STEMS = {
    "ab": ("leg", "loc", "rog"),
    "ad": ("leg", "lig"),
    "ob": ("liqu", "rog"),
    "sub": ("limin", "loc", "lig", "lunar", "lacustre", "lamelar", "lenhoso",
            "licenc", "licenç"),
}

# Not implemented, deliberately: the *productive* prefixes (ciber-, hiper-,
# super-, inter-). Infopédia often keeps them intact -- ci.ber.as.sé.di.o,
# hi.per.es.pe.ci.a.li.zar -- but it does not do so consistently (it also writes
# hi.pe.ra.gu.do, hi.pe.ra.ti.vi.da.de, and a.dre.na.li.na beside ad.re.nal),
# and the Portal da Língua Portuguesa never does it at all: implementing the
# convention moves 177 of its words to wrong. The two authorities disagree, so
# the words are absent from the agreement gold and there is no fact here to
# encode. It is a house style, not a property of Portuguese, and the benchmark
# reports it as such.

#: Verb stems ending in a vowel, before a suffix that opens with ``i``. The
#: ``u``/``a`` belongs to the stem and the ``i`` to the suffix, so the seam
#: between them is a hiatus, however much it looks like a diphthong:
#: dis.tri.bu.i.dor (*distribuir*), des.tru.i.ção (*destruir*), a.ju.i.zar
#: (*juízo*), abs.tra.i.men.to (*abstrair*). Contrast *cuidado*, which has no
#: seam and keeps its diphthong.
#:
#: Matched anywhere in the word, so a prefixed derivative is covered for free:
#: *redistribuidor* needs no entry of its own.
#:
#: Short stems are excluded on purpose: ``us`` would seize *usina*, and ``flu``
#: would seize *fluido*, whose ``ui`` is a root diphthong with no seam in it. A
#: stem that cannot be pointed at an infinitive does not belong here.
VOWEL_FINAL_STEMS = (
    # -uir verbs (Latin -uere) and everything derived from them.
    "distribu", "contribu", "retribu", "atribu", "destru", "constru", "instru",
    "substitu", "constitu", "restitu", "institu", "prostitu",
    "polu", "dilu", "inclu", "conclu", "exclu", "diminu", "imbu", "possu",
    "continu",
    # -air verbs (Latin -ahere) and their nominalizations.
    "abstra", "distra", "subtra", "retra",
    # juízo, juiz and their family: pre.ju.í.zo, a.ju.i.zar, en.ju.i.zar.
    # Safe as a two-letter stem only because the seam test requires a following
    # ``i``, and no Portuguese word has ``jui`` without it.
    "ju",
)

#: The suffixes that open with the ``i`` of the seam above.
_VOCALIC_SUFFIXES = ("i",)


def _prefix_boundary(word: str) -> int:
    """Character offset of a prefix boundary, or 0 for none."""
    for prefix in sorted(LATINATE_PREFIX_STEMS, key=len, reverse=True):
        if not word.startswith(prefix):
            continue
        stem = word[len(prefix):]
        if stem[:1] in _LIQUIDS and any(stem.startswith(s)
                                        for s in LATINATE_PREFIX_STEMS[prefix]):
            return len(prefix)
        return 0

    return 0


def _seam_boundary(word: str) -> int:
    """Character offset of a stem/suffix seam that forces a hiatus, or 0."""
    for stem in VOWEL_FINAL_STEMS:
        start = word.find(stem)
        if start < 0:
            continue
        end = start + len(stem)
        if any(word[end:].startswith(s) for s in _VOCALIC_SUFFIXES) and len(word) > end + 1:
            return end
    return 0


def boundaries(word: str, units: Sequence[Grapheme]) -> Set[int]:
    """Grapheme indices that must begin a syllable, whatever phonotactics says."""
    offsets = {offset for offset in (_prefix_boundary(word), _seam_boundary(word)) if offset}
    if not offsets:
        return set()

    found: Set[int] = set()
    position = 0
    for index, unit in enumerate(units):
        if position in offsets:
            found.add(index)
        position += len(unit)
    return found
