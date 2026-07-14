"""The syllable as a structure, not a string.

A syllable is not an opaque chunk of letters. It is an onset, a nucleus with its
glides, and a coda -- and consumers of this library (phonemizers, stress rules,
rhyme and metre tools) need those constituents. Returning only ``"trans"`` makes
every one of them re-derive what was already known here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .graphemes import Grapheme


@dataclass(frozen=True)
class Syllable:
    """One syllable, decomposed.

    ``onset`` and ``coda`` may be empty; ``nucleus`` never is, except in the
    degenerate case of a word with no vowel at all (an initialism), where the
    whole thing is returned unanalysed.

    ``glide_on`` is the glide spelled inside the onset -- the ``u`` of *qua*dro
    and á*gua*. ``glide_off`` is the offglide of a falling diphthong: the ``i``
    of p*ai*, the ``u`` of s*au*dade.
    """

    onset: str = ""
    glide_on: str = ""
    nucleus: str = ""
    glide_off: str = ""
    coda: str = ""
    #: Hyphen, space or apostrophe held by this syllable. Carried, never dropped,
    #: so the word still spells itself: pau-d'água is pau-.d'á.gua.
    separator: str = ""
    #: The syllable as it is written. Kept verbatim rather than recomposed from
    #: the constituents above, because a separator can sit anywhere in it
    #: (``ra-d'``) and reassembling in onset-nucleus-coda order would silently
    #: reorder the letters.
    surface: str = ""

    @classmethod
    def of(cls, units: Sequence[Grapheme], nucleus: Optional[Sequence[int]] = None) -> "Syllable":
        """Build a syllable from its grapheme units.

        ``nucleus`` holds indices into the *word*, not into ``units``, so it is
        rebased here; when it is absent the units are treated as unanalysable.
        """
        text = "".join(unit.text for unit in units)
        if nucleus is None:
            return cls(onset=text, surface=text)

        base = nucleus[0]
        offset = next((i for i, u in enumerate(units) if u.is_vowel), 0)
        vowels = [offset + (index - base) for index in nucleus]

        head, tail = vowels[0], vowels[-1]
        onset_units = list(units[:head])
        coda_units = [u for u in units[tail + 1:] if not u.is_separator]

        # The glide of qu/gu is spelled inside the onset consonant itself, so it
        # is recovered from the digraph rather than from a separate unit.
        glide_on = ""
        if onset_units and onset_units[-1].text in ("qu", "gu"):
            glide_on = "u"

        return cls(
            onset="".join(u.text for u in onset_units),
            glide_on=glide_on,
            nucleus=units[head].text,
            glide_off="".join(units[i].text for i in vowels[1:]),
            coda="".join(u.text for u in coda_units),
            separator="".join(u.text for u in units if u.is_separator),
            surface=text,
        )

    def __str__(self) -> str:
        return self.surface

    def __len__(self) -> int:
        return len(self.surface)
