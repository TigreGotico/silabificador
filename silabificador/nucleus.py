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

``NASAL-OFFGLIDE``
    A nasal nucleus licenses an offglide that a plain vowel does not, including
    the mid vowels: mãe, pão, põe, sa.guão. Without this, *mãe* would break.

``PRE-PALATAL``
    A close vowel before ``nh`` reverts to a nucleus: ra.i.nha, mo.i.nho,
    cam.pa.i.nha. The palatal nasal pulls it into its own syllable.

``CLOSED-BY-LIQUID-OR-NASAL``
    A close vowel closed by ``l m n r z`` is a nucleus: a.in.da, ca.ir, ju.iz,
    ru.im, ra.ul. It is *not* a nucleus when closed by ``s``/``x``, which leave
    the diphthong standing: pais, mais, de.pois, a.zuis. The traditional rule
    (Cunha & Cintra) says the tonic ``i``/``u`` before ``l m n r z`` forms a
    hiatus; since stress is not modelled here, the coda that conditions the
    stress is used directly, and it picks out the same words.

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
"""
from __future__ import annotations

from typing import List, Sequence

from .graphemes import Grapheme
from .phonotactics import (GLIDE_CAPABLE, HIATUS_BEFORE, HIATUS_CODAS,
                           NASAL_VOWELS, STRESS_MARKED)

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

            if nucleus in NASAL_VOWELS and glide in "eoiu":
                group.append(i + 1)  # NASAL-OFFGLIDE: mãe, pão, põe
            elif glide not in GLIDE_CAPABLE or glide in STRESS_MARKED:
                pass  # ACCENT / no mid or low glides: pa.ís, co.e.lho, su.a
            elif is_last and after in HIATUS_CODAS:
                pass  # CLOSED-BY-LIQUID-OR-NASAL: a.in.da, ca.ir, ju.iz
            elif is_last and _next_units(units, end) in HIATUS_BEFORE:
                pass  # PRE-PALATAL: ra.i.nha, mo.i.nho
            else:
                group.append(i + 1)  # FALLING-DIPHTHONG: pai, sau.da.de, cui.da.do

        groups.append(group)
        i = group[-1] + 1

    return groups


def _next_units(units: Sequence[Grapheme], run_end: int) -> str:
    nxt = run_end + 1
    return units[nxt].text if nxt < len(units) else ""
