"""Layer 1 -- orthography to grapheme units.

Portuguese spelling is not one letter per segment. ``lh`` is a single consonant;
so is the ``qu`` of *quero*, whose ``u`` spells nothing at all. A scanner that
walks letters therefore keeps tripping over sequences that are one unit, and the
usual fix -- rewriting ``ch``/``lh``/``nh``/``gu``/``qu`` to placeholder symbols
before scanning -- is context-blind: it collapses the ``gu`` of *agudo*, where
the ``u`` is a full vowel, exactly as it collapses the ``gu`` of *guerra*.

This layer tokenizes the real string instead, deciding each digraph in context,
so every later layer can assume one unit is one segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import takewhile
from typing import List

from .phonotactics import (CONSONANT_DIGRAPHS, FOREIGN_DIGRAPHS, FOREIGN_MARKS,
                           FOREIGN_TETRAGRAPHS, VOWELS)

#: Characters that hold a word together without being part of any syllable's
#: sound: the hyphen of *guarda-chuva*, the space of *ajudante de campo*, the
#: apostrophe of *pau-d'água*. They are carried through, never dropped.
SEPARATORS = frozenset("- '’")

#: The separators that mark an elided vowel rather than end a word: pau-d'alho,
#: Sant'Ana. The consonant before one is read with the vowel after it.
APOSTROPHES = frozenset("'’")

VOWEL = "vowel"
CONSONANT = "consonant"
SEPARATOR = "separator"


@dataclass(frozen=True)
class Grapheme:
    """One orthographic unit: ``text`` may be one letter or a digraph."""

    text: str
    kind: str

    @property
    def is_vowel(self) -> bool:
        return self.kind == VOWEL

    @property
    def is_consonant(self) -> bool:
        return self.kind == CONSONANT

    @property
    def is_separator(self) -> bool:
        return self.kind == SEPARATOR

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.text


def _u_is_mute(word: str, index: int) -> bool:
    """Is the ``u`` at ``index``, following q/g, spelling nothing of its own?

    After ``q`` or ``g``, a ``u`` that is immediately followed by another vowel
    never heads a syllable: it is either silent (*quero*, *guerra*) or a glide
    riding in the onset (*quadro*, *água*). Either way it belongs to the onset,
    so ``qu``/``gu`` is one consonant unit.

    A ``u`` followed by a consonant is a plain vowel and keeps its nucleus --
    which is the whole difference between *a.gu.do* and *guer.ra*, and the
    reason this decision cannot be made by string replacement.

    The pre-1990 trema marks the same ``u`` as pronounced (*agüentar*,
    *qüinqüenal*). It is still a glide in the onset, never a nucleus: AO90 Base
    XX 5 keeps ``gu`` and ``qu`` with the following vowel whether the ``u`` is
    pronounced or not.
    """
    if word[index] not in "uü":
        return False
    nxt = index + 1
    return nxt < len(word) and word[nxt] in VOWELS


def _foreign_vowel(word: str, index: int) -> str:
    """The loanword vowel spelling that starts at ``index``, or ``""``.

    Each spelling is one vowel in its donor language, and each is anchored on a
    context that no Portuguese word has, because the same letters are a
    Portuguese hiatus or a Portuguese diphthong elsewhere:

    * French ``eau`` [o], at the end of a word or before a final ``s``/``x``:
      bu.reau, rous.seau, bor.deaux. Portuguese has ``eau`` only at a prefix
      seam, before more of the word: an.te.au.ro.ra, te.le.au.la.
    * French ``eu`` [œ] before a final ``r``: bo.xeur, chauf.feur. No
      Portuguese word ends in ``eur``.
    * French and English ``ou`` [u], [aʊ], in a word that :data:`FOREIGN_MARKS`
      marks as foreign: gour.met, coun.try, play.ground. In a Portuguese word
      ``ou`` is already a diphthong, but a following ``n`` or ``r`` closes the
      syllable and AO90 Base X 2 reads the ``u`` as a nucleus (go.ur.met); that
      rule describes Portuguese stress, not the donor's spelling.

    A word ends at a separator, so the end of each element of a compound counts:
    beaux-.arts, as bor.deaux. AO90 Base XV 1 hyphenates compounds whose
    elements "mantêm acento próprio", each a word of its own.
    """
    rest = "".join(takewhile(lambda char: char not in SEPARATORS, word[index:]))
    if rest.startswith("eau") and rest[3:] in ("", "s", "x"):
        return "eau"
    if rest.startswith("eu") and rest[2:] in ("r", "rs"):
        return "eu"
    if rest.startswith("ou") and FOREIGN_MARKS.search(word):
        return "ou"
    return ""


def tokenize(word: str) -> List[Grapheme]:
    """Segment ``word`` (already lowercased) into grapheme units."""
    units: List[Grapheme] = []
    i = 0
    length = len(word)

    while i < length:
        char = word[i]
        pair = word[i:i + 2]

        foreign_vowel = _foreign_vowel(word, i) if char in VOWELS else ""

        if char in SEPARATORS:
            units.append(Grapheme(char, SEPARATOR))
            i += 1
        elif word[i:i + 4] in FOREIGN_TETRAGRAPHS:
            units.append(Grapheme(word[i:i + 4], CONSONANT))
            i += 4
        elif foreign_vowel:
            units.append(Grapheme(foreign_vowel, VOWEL))
            i += len(foreign_vowel)
        elif pair in CONSONANT_DIGRAPHS or pair in FOREIGN_DIGRAPHS:
            units.append(Grapheme(pair, CONSONANT))
            i += 2
        elif char in "qg" and i + 1 < length and _u_is_mute(word, i + 1):
            units.append(Grapheme(pair, CONSONANT))
            i += 2
        elif char in VOWELS:
            units.append(Grapheme(char, VOWEL))
            i += 1
        else:
            units.append(Grapheme(char, CONSONANT))
            i += 1

    return units
