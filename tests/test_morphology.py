"""Morpheme boundaries -- the lexical layer.

Every test here is a *minimal pair*: two words with the same letters in the same
shape and opposite syllabifications. Each pair is the justification for the
lexicon existing at all -- no phonotactic rule can separate them, because what
separates them is whether the word is still morphologically analysable.

If an entry is ever deleted, the left-hand column breaks and the right-hand
column keeps working. That is the test that a lexicon entry is doing real work
rather than memorizing an answer.
"""
import pytest

from silabificador import syllabify
from silabificador.morphology import LATINATE_PREFIX_STEMS, VOWEL_FINAL_STEMS


@pytest.mark.parametrize("analysable,expected,opaque,opaque_expected", [
    # sub + limen (threshold)      vs  Latin sublimis, one morpheme
    ("subliminar", ["sub", "li", "mi", "nar"], "sublime", ["su", "bli", "me"]),
    # ab + legare (to send away)   vs  brandar, with no prefix in it
    ("ablegar", ["ab", "le", "gar"], "abrandar", ["a", "bran", "dar"]),
    # sub + locare (to let)        vs  no seam
    ("sublocar", ["sub", "lo", "car"], "sobrar", ["so", "brar"]),
])
def test_prefix_boundary_only_where_the_prefix_is_alive(
        analysable, expected, opaque, opaque_expected):
    assert syllabify(analysable) == expected
    assert syllabify(opaque) == opaque_expected


@pytest.mark.parametrize("word,expected", [
    ("sublocatário", ["sub", "lo", "ca", "tá", "ri", "o"]),
    ("sublocação", ["sub", "lo", "ca", "ção"]),
    ("subliminal", ["sub", "li", "mi", "nal"]),
    ("ablegação", ["ab", "le", "ga", "ção"]),
])
def test_a_stem_entry_covers_every_derivative(word, expected):
    # One `loc` entry, not one entry per derived word.
    assert syllabify(word) == expected


@pytest.mark.parametrize("seam,expected,root,root_expected", [
    # distribu-ir + -idor          vs  cuidado, a root diphthong
    ("distribuidor", ["dis", "tri", "bu", "i", "dor"], "cuidado", ["cui", "da", "do"]),
    # destru-ir + -ição            vs  fluido, a root diphthong
    ("destruição", ["des", "tru", "i", "ção"], "fluido", ["flui", "do"]),
    # juízo                        vs  muito
    ("ajuizar", ["a", "ju", "i", "zar"], "muito", ["mui", "to"]),
])
def test_derivational_seam_forces_hiatus_but_a_root_diphthong_stands(
        seam, expected, root, root_expected):
    assert syllabify(seam) == expected
    assert syllabify(root) == root_expected


def test_seam_is_found_in_prefixed_derivatives_too():
    # The stem is matched anywhere, so this needs no entry of its own.
    assert syllabify("redistribuidor") == ["re", "dis", "tri", "bu", "i", "dor"]


def test_short_stems_do_not_seize_unrelated_words():
    # `us`/`flu` are deliberately absent: they would break these.
    assert syllabify("usina") == ["u", "si", "na"]
    assert syllabify("fluido") == ["flui", "do"]


def test_the_lexicon_stores_morphemes_not_answers():
    # Nothing in the tables is a syllable or a syllabified word: every entry is a
    # prefix or a stem, and the rules derive the split from it.
    for prefix, stems in LATINATE_PREFIX_STEMS.items():
        assert prefix.isalpha() and 2 <= len(prefix) <= 5
        assert all(stem.isalpha() for stem in stems)
    assert all(stem.isalpha() for stem in VOWEL_FINAL_STEMS)


def test_the_lexicon_stays_small():
    # A guard against the table growing back into a patch list. If a new entry
    # is needed, it should be because a morpheme was missing, not because a word
    # failed -- and either way it should be visible in review.
    entries = sum(len(stems) for stems in LATINATE_PREFIX_STEMS.values())
    assert entries + len(VOWEL_FINAL_STEMS) < 60
