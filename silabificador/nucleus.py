"""Layer 2 -- nucleus and glide resolution.

Given a run of adjacent vowel letters, how many syllables is it? *pai* is one,
*pa.ís* is two, and the letters are the same; *cui.da.do* is a diphthong while
*a.in.da* is a hiatus, and the shapes are the same. Deciding this by listing the
vowel pairs that "are" hiatuses cannot work, because the same pair goes both
ways: ``ui`` is a diphthong in *cui.da.do* and a hiatus in *ju.iz*.

The decision is not a property of the vowel pair. It is a property of the pair
*in its context*: what accent it carries, and what closes the syllable after it.
Each rule below is stated once, applies everywhere, and is named.

Rules, in order of application:

``ACCENT``
    A close vowel bearing an accent is a nucleus, never a glide. This is what
    the accent is *for*: it is the only thing that distinguishes ``pais``
    (parents, one syllable) from ``país`` (country, two).

``PRE-PALATAL``
    A close vowel before ``nh`` reverts to a nucleus: ra.i.nha, mo.i.nho,
    cam.pa.i.nha, mã.i.nha. The palatal nasal pulls it into its own syllable.
    AO90 Base X 2 states it for any vowel before the ``i``/``u``: the close
    vowel "constitui sílaba com a consoante seguinte, como é o caso de nh"
    (bainha, moinho, rainha). It therefore outranks NASAL-OFFGLIDE.

``NASAL-OFFGLIDE``
    A nasal nucleus licenses an offglide that a plain vowel does not, including
    the mid vowels: mãe, pão, põe, sa.guão. Without this, *mãe* would break.

``CONTRACTION-AO``
    The contractions *ao* and *aos* are one syllable: the ``o`` is an offglide.
    AO90 Base VII 1 Obs. admits the diphthong spelled ``ao`` "nas combinações
    da preposição a com as formas masculinas do artigo ou pronome demonstrativo
    o, ou seja, ao e aos", and in no other word. A hyphen or a space ends the
    word; an apostrophe does not, as in :func:`~silabificador.parser._assemble`.

``CLOSED-BY-LIQUID-OR-NASAL``
    A close vowel closed by ``l m n r z`` is a nucleus: a.in.da, ca.ir, ju.iz,
    ru.im, ra.ul. It is *not* a nucleus when closed by ``s``/``x``, which leave
    the diphthong standing: pais, mais, de.pois, a.zuis. The traditional rule
    (Cunha & Cintra) says the tonic ``i``/``u`` before ``l m n r z`` forms a
    hiatus; since stress is not modelled here, the coda that conditions the
    stress is used directly, and it picks out the same words.

``IDENTICAL-VOWELS``
    Two identical vowels are two nuclei: xi.i.ta, va.di.i.ce, du.un.vi.ro. AO90
    Base XX 4 separates consecutive vowels "mesmo que sejam iguais".

``FALLING-DIPHTHONG``
    Otherwise an unaccented ``i``/``u`` after a nucleus is an offglide, and a
    nucleus takes at most one: in *abaiucado* the run ``aiu`` is ``a`` + offglide
    ``i``, and the ``u`` must start a new syllable -- a.bai.u.ca.do.

Every other vowel is a nucleus, which is what makes *co.e.lho*, *su.a* and
*his.tó.ri.a* fall out for free: Portuguese has no mid or low glides, so a
rising sequence is simply a hiatus. No rule for it is needed, and none is given.

Rising sequences that *are* single syllables -- qua.dro, guer.ra, á.gua -- are
not exceptions to this. Their glide is spelled inside the onset (``qu``, ``gu``)
and layer 1 has already absorbed it, so it never reaches this layer. That is
also why triphthongs need no rule: by the time *quais* arrives here, it is a
plain nucleus plus one offglide.

Two more rules need to know which syllable is stressed, so they run in a second
pass, :func:`by_stress`, after the stress module has read the first one:

``STRESSED-IU-UI``
    A stressed ``iu`` or ``ui`` after another vowel is the diphthong, and the
    vowel before it stands alone: flu.iu, dis.tra.iu, pa.uis, not *flui.u*. AO90
    Base X 6: "Prescinde-se do acento agudo nos ditongos tónicos grafados iu e
    ui, quando precedidos de vogal: distraiu, instruiu, pauis"; Base XX 4 divides
    "flu-iu". The first pass reads *flui.u*, which makes the final ``u`` a
    stressed vowel in hiatus after a diphthong, and Base X 5 would then require
    an accent on it (*tuiuiú*). Only a final syllable, alone or before ``s``,
    qualifies: in a paroxytone, Base X 4 leaves the lone vowel unaccented after a
    diphthong, and *bai.u.ca* is right as it stands.

    "Final" means final in its own word. An enclitic pronoun joins the verb with
    a hyphen, Base XVII 1: "Emprega-se o hífen na ênclise e na tmese: amá-lo,
    dá-se, deixa-o, partir-lhe". Base X reads the verb before the hyphen as a
    word of its own: Base X 3 applies "as regras anteriores" to it and accents
    "atraí-lo(s) [de atrair-lo(s)]", where the ``r`` is lost and the ``i`` ends
    the verb. So the scan for a final syllable stops at a separator, and
    *saiu-se* is sa.iu-.se, as *saiu* is sa.iu.

``UNSTRESSED-IU``
    ``iu`` outside the stressed syllable is two nuclei: mi.u.de.za, ci.u.men.to,
    vi.ní.ci.us. Mateus & d'Andrade (1998) list the falling diphthong [iw] among
    the stressed diphthongs only (p. 18, ex. 5: *viu*). They give the unstressed
    ``iu`` of *miudeza* as a glide before a vowel, [ju] (p. 20, ex. 8b), and such
    glides, "even if they are perceived at the phonetic level as glides [...],
    are syllable nuclei at the base level" (p. 21). The rule does not extend to
    ``ui``: the same table lists a pre-stressed [uj] in *cuidado* (p. 18, ex. 5b).
"""
from __future__ import annotations

from itertools import takewhile
from typing import List, Sequence, Set

from .graphemes import APOSTROPHES, Grapheme
from .phonotactics import (AO_DIPHTHONG_WORDS, GLIDE_CAPABLE, HIATUS_BEFORE,
                           HIATUS_CODAS, NASAL_VOWELS, STRESS_MARKED)

#: One nucleus and its offglides, as indices into the grapheme list.
Group = List[int]


def _closes_syllable(units: Sequence[Grapheme], run_end: int) -> str:
    """The consonant that would close the syllable ending at ``run_end``, if any.

    A consonant only closes a syllable when it cannot be handed to the next one:
    in *bai.le* the ``l`` has a vowel after it, so it is an onset and the ``ai``
    survives as a diphthong; in *a.in.da* the ``n`` is followed by ``d``, so it
    is stuck in the coda, and the ``i`` is a nucleus. Same letters, opposite
    outcomes, one rule.

    A geminate is the exception, and not an arbitrary one: ``rr`` and ``ss``
    spell a *single* consonant, which the orthography splits across the boundary
    for reasons of spelling, not of sound. Nothing closes the syllable, so the
    diphthong stands -- bair.ro, bair.ris.mo, not *ba.ir.ro.
    """
    first = run_end + 1
    if first >= len(units) or not units[first].is_consonant:
        return ""

    following = units[first + 1] if first + 1 < len(units) else None
    if following is not None and following.text == units[first].text:
        return ""

    is_coda = following is None or not following.is_vowel
    return units[first].text if is_coda else ""


def resolve(units: Sequence[Grapheme], start: int, end: int) -> List[Group]:
    """Group the vowel run ``units[start:end + 1]`` into syllable nuclei."""
    groups: List[Group] = []
    i = start

    while i <= end:
        group = [i]
        nucleus = units[i].text

        if i < end:
            glide = units[i + 1].text
            is_last = (i + 1) == end
            after = _closes_syllable(units, end) if is_last else ""

            if is_last and glide in GLIDE_CAPABLE and _next_units(units, end) in HIATUS_BEFORE:
                pass  # PRE-PALATAL: ra.i.nha, mo.i.nho, mã.i.nha
            elif nucleus in NASAL_VOWELS and glide in "eoiu":
                group.append(i + 1)  # NASAL-OFFGLIDE: mãe, pão, põe
            elif nucleus + glide == "ao" and _word_at(units, i) in AO_DIPHTHONG_WORDS:
                group.append(i + 1)  # CONTRACTION-AO: ao, aos
            elif glide not in GLIDE_CAPABLE or glide in STRESS_MARKED:
                pass  # ACCENT / no mid or low glides: pa.ís, co.e.lho, su.a
            elif glide == nucleus:
                pass  # IDENTICAL-VOWELS: xi.i.ta, va.di.i.ce
            elif is_last and after in HIATUS_CODAS:
                pass  # CLOSED-BY-LIQUID-OR-NASAL: a.in.da, ca.ir, ju.iz
            else:
                group.append(i + 1)  # FALLING-DIPHTHONG: pai, sau.da.de, cui.da.do

        groups.append(group)
        i = group[-1] + 1

    return groups


def _word_at(units: Sequence[Grapheme], index: int) -> str:
    """The word that holds ``units[index]``, ended by a hyphen or a space."""
    def ends_word(unit: Grapheme) -> bool:
        return unit.is_separator and unit.text not in APOSTROPHES

    before = list(takewhile(lambda unit: not ends_word(unit), reversed(units[:index])))
    after = takewhile(lambda unit: not ends_word(unit), units[index:])
    return "".join(unit.text for unit in [*before[::-1], *after])


def _next_units(units: Sequence[Grapheme], run_end: int) -> str:
    nxt = run_end + 1
    return units[nxt].text if nxt < len(units) else ""


def by_stress(units: Sequence[Grapheme], groups: Sequence[Group],
              stressed: Sequence[bool], hard: Set[int]) -> List[Group]:
    """Regroup ``groups`` once ``stressed[k]`` says whether group ``k`` is stressed.

    ``hard`` holds the morpheme boundaries, which no group may straddle.
    """
    regrouped: List[Group] = []
    for group, is_stressed in zip(groups, stressed):
        head = group[0]
        previous = regrouped[-1] if regrouped else []
        glide = units[head - 1].text if head else ""
        if (is_stressed and len(group) == 1 and units[head].text in ("i", "u")
                and previous and previous[-1] == head - 1
                and glide in ("i", "u") and glide != units[head].text
                and head not in hard
                and all(unit.text == "s" for unit in takewhile(
                    lambda unit: not unit.is_separator, units[head + 1:]))):
            # STRESSED-IU-UI: flu.iu, dis.tra.iu, pa.uis
            if len(previous) > 1:
                previous.pop()
            else:
                regrouped.pop()
            regrouped.append([head - 1, head])
        elif (not is_stressed and len(group) == 2
                and units[head].text == "i" and units[group[1]].text == "u"):
            regrouped.extend([[head], [group[1]]])  # UNSTRESSED-IU: mi.u.de.za
        else:
            regrouped.append(list(group))
    return regrouped
