"""Layer 3 -- syllable assembly.

Every syllable has exactly one nucleus, so layer 2 has already fixed how many
syllables there are. All that remains is to decide where each consonant goes,
and there is one principle for that:

    **Onset maximization, bounded by the onset inventory.** A consonant cluster
    between two nuclei gives as much of its right edge to the following onset as
    Portuguese licenses as an onset; whatever is left over is stranded in the
    coda.

That is the entire boundary rule. Because :data:`~silabificador.phonotactics.
COMPLEX_ONSETS` says which clusters are licit, the rule derives on its own:

    a.tle.ta      ``tl`` is a licit onset, so nothing splits
    pac.to        ``ct`` is not, so ``c`` is stranded
    car.ro        ``rr`` is not, so it splits -- and so do ss, sc, sç, xc
    abs.tra.ir    ``tr`` is licit, ``str`` is not; ``bs`` is stranded
    cons.tru.ir   likewise

Codas are not checked against an inventory, because Portuguese spelling admits
whatever the onset rule strands -- ap.to, rit.mo, ab.sur.do, sof.twa.re. Onsets
are the constrained edge; the coda is the remainder.
"""
from __future__ import annotations

from typing import List, Sequence, Set

from . import morphology, nucleus
from .graphemes import Grapheme, tokenize
from .phonotactics import is_complex_onset
from .syllable import Syllable

#: Portuguese onsets hold at most two units (an obstruent and a liquid). ``qu``,
#: ``gu``, ``ch``, ``lh``, ``nh`` are single units by the time they get here.
MAX_ONSET = 2


def _onset_size(cluster: Sequence[Grapheme]) -> int:
    """How many of ``cluster``'s trailing units the next syllable may claim."""
    if not cluster:
        return 0
    if len(cluster) == 1:
        # A single consonant between nuclei is always an onset: ca.sa, never cas.a.
        return 1
    tail = cluster[-2].text + cluster[-1].text
    if is_complex_onset(tail):
        return MAX_ONSET
    return 1


def _split_runs(units: Sequence[Grapheme]) -> List[List[int]]:
    """Every maximal run of vowels, as index lists."""
    runs: List[List[int]] = []
    current: List[int] = []
    for index, unit in enumerate(units):
        if unit.is_vowel:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _split_group(group: nucleus.Group, hard: Set[int]) -> List[nucleus.Group]:
    """Break a nucleus group wherever a morpheme boundary falls inside it."""
    pieces: List[nucleus.Group] = []
    current = [group[0]]
    for index in group[1:]:
        if index in hard:
            pieces.append(current)
            current = [index]
        else:
            current.append(index)
    pieces.append(current)
    return pieces


def parse(word: str) -> List[Syllable]:
    """Syllabify a lowercased word into :class:`Syllable` objects."""
    units = tokenize(word)
    if not units:
        return []

    # Indices that a morpheme boundary forces to begin a syllable.
    hard = morphology.boundaries(word, units)

    groups: List[nucleus.Group] = []
    for run in _split_runs(units):
        groups.extend(nucleus.resolve(units, run[0], run[-1]))
    if hard:
        # A nucleus may not straddle a seam: the ``ui`` of *distribuidor* is two
        # nuclei because the boundary falls between them, not because ``ui`` is
        # ever a hiatus -- in *cuidado* it is not.
        groups = [piece for group in groups for piece in _split_group(group, hard)]

    if not groups:
        # No nucleus at all: an initialism, a stray consonant. There is no
        # syllable to find, and inventing one would only corrupt the word.
        return [Syllable.of(units)]

    syllables: List[Syllable] = []
    start = 0  # first unit not yet placed
    for position, group in enumerate(groups):
        if position + 1 == len(groups):
            end = len(units) - 1  # the last syllable takes every remaining coda
        else:
            nxt = groups[position + 1][0]
            between = range(group[-1] + 1, nxt)
            separators = [i for i in between if units[i].is_separator]
            cluster = [i for i in between if units[i].is_consonant]
            crossing = [i for i in hard if group[-1] < i <= nxt]

            if crossing:
                # A morpheme boundary outranks onset maximization: the next
                # syllable may not reach back across it. sub|liminar is
                # sub.li.mi.nar, not su.bli.mi.nar; ciber|assédio is
                # ci.ber.as.sé.di.o, not ci.be.ras.sé.di.o.
                end = min(crossing) - 1
            elif separators:
                # A separator is a hard boundary: no syllable spans it, and no
                # onset reaches back across it. In *ab-reagir* the ``b`` cannot
                # join the ``r`` to form the ``br`` onset it would otherwise make
                # -- the syllable closes on the hyphen: ab-.re.a.gir.
                end = separators[-1]
            elif not cluster:
                end = group[-1]
            else:
                onset = _onset_size([units[i] for i in cluster])
                end = cluster[-onset - 1] if onset < len(cluster) else group[-1]

        # A hyphen, space or apostrophe belongs to the syllable it follows, so
        # that guarda-chuva is guar.da-.chu.va and the word still spells itself.
        while end + 1 < len(units) and units[end + 1].is_separator:
            end += 1

        syllables.append(Syllable.of(units[start:end + 1], nucleus=group))
        start = end + 1

    return syllables
