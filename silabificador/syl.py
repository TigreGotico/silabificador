"""Public syllabification entry points.

The work is done in layers, each of which can be read, tested and corrected on
its own:

    :mod:`~silabificador.phonotactics`  what Portuguese licenses (data only)
    :mod:`~silabificador.graphemes`     orthography -> grapheme units  (layer 1)
    :mod:`~silabificador.nucleus`       nucleus and glide resolution   (layer 2)
    :mod:`~silabificador.morphology`    morpheme boundaries
    :mod:`~silabificador.parser`        syllable assembly              (layer 3)
    :mod:`~silabificador.stress`        which syllable bears the stress
"""
from __future__ import annotations

from typing import List

from . import stress as _stress
from .parser import parse
from .syllable import Syllable


def analyze(word: str) -> List[Syllable]:
    """Syllabify ``word`` into :class:`~silabificador.syllable.Syllable` objects.

        >>> [(s.onset, s.nucleus, s.stressed) for s in analyze("casa")]
        [('c', 'a', True), ('s', 'a', False)]

    Use this when you need the constituents -- a phonemizer, a stress rule, a
    rhyme index. Use :func:`syllabify` when you only need the strings.
    """
    cleaned = word.strip().lower()
    return _stress.assign(parse(cleaned), cleaned)


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


def stressed_index(word: str) -> int:
    """Index of the syllable carrying ``word``'s primary stress.

        >>> stressed_index("computador")   # com.pu.ta.DOR
        3
        >>> stressed_index("casa")         # CA.sa
        0
        >>> stressed_index("sílaba")       # SÍ.la.ba
        0
    """
    for index, syllable in enumerate(analyze(word)):
        if syllable.stressed:
            return index
    return 0
