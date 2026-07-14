"""Public syllabification entry points.

The work is done in layers, each of which can be read, tested and corrected on
its own:

    :mod:`~silabificador.phonotactics`  what Portuguese licenses (data only)
    :mod:`~silabificador.graphemes`     orthography -> grapheme units  (layer 1)
    :mod:`~silabificador.nucleus`       nucleus and glide resolution   (layer 2)
    :mod:`~silabificador.morphology`    prefix boundaries
    :mod:`~silabificador.parser`        syllable assembly             (layer 3)
"""
from __future__ import annotations

from typing import List

from .parser import parse
from .syllable import Syllable


def analyze(word: str) -> List[Syllable]:
    """Syllabify ``word`` into :class:`~silabificador.syllable.Syllable` objects.

        >>> [(s.onset, s.nucleus, s.glide_off, s.coda) for s in analyze("prainha")]
        [('pr', 'a', '', ''), ('', 'i', '', ''), ('nh', 'a', '', '')]

    Use this when you need the constituents -- a phonemizer, a stress rule, a
    rhyme index. Use :func:`syllabify` when you only need the strings.
    """
    return parse(word.strip().lower())


def syllabify(word: str) -> List[str]:
    """Split a Portuguese word into syllables.

        >>> syllabify("computador")
        ['com', 'pu', 'ta', 'dor']
        >>> syllabify("guarda-chuva")
        ['guar', 'da-', 'chu', 'va']

    The syllables always concatenate back to ``word.strip().lower()``: a hyphen,
    space or apostrophe is kept on the syllable it follows rather than dropped.
    """
    return [str(syllable) for syllable in analyze(word)]
