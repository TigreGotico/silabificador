"""Portuguese phonotactic inventories.

This module is *data*. It declares what Portuguese permits; it decides nothing.
The engine consults it, so a claim about the language can be checked, cited and
corrected in one place instead of being spread across the control flow.

References
----------
* Mateus, M. H. & d'Andrade, E. (2000). *The Phonology of Portuguese*. OUP.
  Ch. 3 (syllable structure, the onset inventory, the sonority scale).
* Cunha, C. & Cintra, L. (1984). *Nova Gramática do Português Contemporâneo*.
  §"Divisão silábica" -- the orthographic conventions the dictionaries follow.
* Acordo Ortográfico da Língua Portuguesa (1990), Base XX ("Da divisão
  silábica"), which is what the source dictionaries are codifying.
"""
from __future__ import annotations

from typing import FrozenSet

#: Letters that can head a syllable nucleus.
VOWELS: FrozenSet[str] = frozenset("aeiouáéíóúâêôàãõäëïöüy")

#: The only vowels that can surrender their nucleus and become glides. Portuguese
#: has no mid or low glides: in ``coelho`` the ``e`` cannot be a glide, so the
#: sequence is a hiatus (co.e.lho). ``y`` behaves as ``i`` in loans.
GLIDE_CAPABLE: FrozenSet[str] = frozenset("iuy")

#: An accent mark on a close vowel pins it as a nucleus, which is precisely how
#: Portuguese spelling disambiguates ``pais`` (pais) from ``país`` (pa.ís).
STRESS_MARKED: FrozenSet[str] = frozenset("áéíóúâêîôûàèìòù")

#: Nasal vowels. A nasal nucleus licenses an offglide that a plain vowel would
#: not: ``mãe``, ``pão``, ``põe`` are single syllables, while ``ma.e`` would be
#: a hiatus.
NASAL_VOWELS: FrozenSet[str] = frozenset("ãõ")

CONSONANTS: FrozenSet[str] = frozenset("bcdfghjklmnpqrstvwxyzçñ")

#: Two letters spelling one consonant. They never split, because there is no
#: boundary inside a single segment: fi.lho, chu.va, ba.nha.
CONSONANT_DIGRAPHS: FrozenSet[str] = frozenset({"ch", "lh", "nh"})

#: Loanword spellings that behave as single consonants: show, thriller, graffiti.
FOREIGN_DIGRAPHS: FrozenSet[str] = frozenset({"sh", "th", "ff", "ph"})

#: Consonants that can be followed by a liquid to form a complex onset. This is
#: the sonority sequencing principle: onset sonority must rise toward the
#: nucleus, and obstruent < liquid < vowel.
_OBSTRUENTS = "pbtdcgfv"
_LIQUIDS = "lr"

#: Every complex onset Portuguese licenses -- obstruent + liquid, and nothing
#: else. Membership here is the *whole* of the boundary rule for a consonant
#: cluster: a cluster's final units go to the onset if and only if they are in
#: this set. That single fact derives, with no further stipulation:
#:
#:   a.tle.ta   -- ``tl`` is a licit onset, so it does not split
#:   pac.to     -- ``ct`` is not, so ``c`` is forced into the coda
#:   car.ro     -- ``rr`` is not, so it splits; likewise ``ss``, ``sc``, ``xc``
#:   abs.tra.ir -- ``tr`` is licit, ``str`` is not; ``bs`` is left in the coda
#:
#: There is no companion list of clusters that *must* split. A cluster splits
#: precisely when it is not in this set, so stating which onsets exist states
#: both.
COMPLEX_ONSETS: FrozenSet[str] = frozenset(
    o + liquid for o in _OBSTRUENTS for liquid in _LIQUIDS
)

#: Codas are deliberately *not* an inventory. Portuguese spelling admits a coda
#: consonant wherever the following cluster is not a licit onset -- including
#: learned and loan clusters (ap.to, rit.mo, ab.sur.do, sof.twa.re) that no
#: closed native coda set would sanction. Constraining the onset is enough;
#: constraining the coda as well would only reject words the dictionaries spell.

#: A glide before ``nh`` reverts to a nucleus: ra.i.nha, mo.i.nho, cam.pa.i.nha.
#: The palatal nasal attracts the preceding close vowel into its own syllable.
HIATUS_BEFORE = frozenset({"nh"})

#: A close vowel closed by one of these consonants is a nucleus, not a glide:
#: a.in.da, ca.ir, ju.iz, ru.im, sa.ir. Crucially ``s`` and ``x`` are absent --
#: they leave the diphthong intact (pais, mais, de.pois, a.zuis), which is what
#: makes ``pais``/``país`` fall out of the rules rather than out of a list.
HIATUS_CODAS: FrozenSet[str] = frozenset("lmnrz")


def is_vowel(unit: str) -> bool:
    return unit in VOWELS


def is_complex_onset(cluster: str) -> bool:
    """Can ``cluster`` (2+ letters, digraphs already resolved) head a syllable?"""
    return cluster in COMPLEX_ONSETS
