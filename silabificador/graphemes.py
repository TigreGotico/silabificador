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
from typing import List

from .phonotactics import (CONSONANT_DIGRAPHS, FOREIGN_DIGRAPHS, VOWELS)

#: Characters that hold a word together without being part of any syllable's
#: sound: the hyphen of *guarda-chuva*, the space of *ajudante de campo*, the
#: apostrophe of *pau-d'água*. They are carried through, never dropped.
SEPARATORS = frozenset("- '’")

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
    """
    if word[index] != "u":
        return False
    nxt = index + 1
    return nxt < len(word) and word[nxt] in VOWELS


def tokenize(word: str) -> List[Grapheme]:
    """Segment ``word`` (already lowercased) into grapheme units."""
    units: List[Grapheme] = []
    i = 0
    length = len(word)

    while i < length:
        char = word[i]
        pair = word[i:i + 2]

        if char in SEPARATORS:
            units.append(Grapheme(char, SEPARATOR))
            i += 1
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
