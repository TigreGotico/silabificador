from typing import List

from silabificador.syl import analyze, stressed_index, syllabify
from silabificador.syllable import Syllable

__all__ = ["Syllabifier", "Syllable", "analyze", "stressed_index", "syllabify"]


class Syllabifier:
    """A Portuguese syllabifier.

    Rule-based and dependency-free; no model is loaded. Orthography is resolved
    into grapheme units, vowel runs into nuclei and glides, and consonants are
    placed by onset maximization bounded by the licit onset inventory, with
    transparent prefix boundaries overriding it.
    """

    def syllabify(self, word: str) -> List[str]:
        """Split a Portuguese word into syllable strings; see :func:`silabificador.syllabify`."""
        return syllabify(word)

    def analyze(self, word: str) -> List[Syllable]:
        """One :class:`Syllable` per syllable, with the stressed one marked; see :func:`silabificador.analyze`."""
        return analyze(word)

    def stressed_index(self, word: str) -> int:
        """Index of the syllable carrying the word's primary stress; see :func:`silabificador.stressed_index`."""
        return stressed_index(word)
