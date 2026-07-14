"""Stress assignment."""
import pytest

from silabificador import analyze, stressed_index, syllabify


def marked(word):
    """The word with its stressed syllable in caps and secondaries in parens."""
    out = []
    for syllable in analyze(word):
        text = str(syllable)
        if syllable.stressed:
            out.append(text.upper())
        elif syllable.secondary:
            out.append(f"({text})")
        else:
            out.append(text)
    return ".".join(out)


@pytest.mark.parametrize("word,expected", [
    ("sílaba", 0),        # SÍ.la.ba -- proparoxytone, and only an accent can say so
    ("português", 2),     # por.tu.GUÊS
    ("avô", 1),
    ("público", 0),
])
def test_an_accent_names_the_stressed_syllable(word, expected):
    assert stressed_index(word) == expected


def test_the_accent_is_the_only_thing_separating_these():
    # Same letters, three words, three stresses. Nothing but the accent decides.
    assert stressed_index("sábia") == 0     # SÁ.bi.a  (wise)
    assert stressed_index("sabia") == 1     # sa.BI.a  (knew)
    assert stressed_index("sabiá") == 2     # sa.bi.Á  (a bird)


@pytest.mark.parametrize("word,expected", [
    ("irmã", 1),
    ("coração", 2),
    ("órfão", 0),         # the acute goes first, so the tilde stays silent
])
def test_a_tilde_carries_stress_when_no_other_accent_does(word, expected):
    assert stressed_index(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("comer", 1),         # -r
    ("rapaz", 1),         # -z
    ("papel", 1),         # -l
    ("jabuti", 2),        # -i
    ("bambu", 1),         # -u
    ("jardim", 1),        # -im
    ("batom", 1),         # -om
    ("comuns", 1),        # -uns
])
def test_ending_puts_the_stress_last(word, expected):
    assert stressed_index(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("casa", 0),
    ("computador", 3),    # ...-r, so oxytone after all
    ("homem", 0),         # -em is an inflection: it does not move the stress
    ("falam", 0),         # -am likewise
    ("jovens", 0),        # -ens likewise
    ("casas", 0),
])
def test_otherwise_the_stress_is_paroxytone(word, expected):
    assert stressed_index(word) == expected


def test_an_unaccented_word_is_never_proparoxytone():
    # It could not be: the spelling rules would have required an accent to say so.
    # This is what makes rule 3 a two-way choice rather than a guess.
    for word in ("casa", "comer", "jovens", "computador", "batom"):
        assert stressed_index(word) >= len(syllabify(word)) - 2


@pytest.mark.parametrize("word,expected", [
    ("guarda-chuva", "(guar).da-.CHU.va"),
    ("café-concerto", "ca.(fé-).con.CER.to"),
    ("ab-rogado", "(ab-).ro.GA.do"),
])
def test_a_compound_keeps_a_stress_on_every_element(word, expected):
    # Two words wearing one coat: each keeps its own stress, and the last element
    # takes the primary.
    assert marked(word) == expected


def test_clitics_inside_a_compound_carry_nothing():
    # "de" is a monossilabo atono: an article or preposition is not a stress
    # domain, so it is neither primary nor secondary.
    syllables = analyze("chapéu de chuva")
    de = [s for s in syllables if str(s).strip() == "de"][0]
    assert not de.stressed and not de.secondary
    assert marked("chapéu de chuva") == "cha.(péu ).de .CHU.va"


@pytest.mark.parametrize("word,expected", [
    ("sòmente", "(sò).MEN.te"),
    ("cafèzinho", "ca.(fè).ZI.nho"),
])
def test_the_pre_1990_grave_marks_a_secondary_stress(word, expected):
    # The only explicit signal of secondary stress Portuguese spelling ever had.
    assert marked(word) == expected


def test_the_grave_of_crasis_is_not_a_stress_mark():
    # "à" is a contracted preposition + article. Nothing about it is stressed
    # relative to another syllable -- and as a whole word it is simply itself.
    assert [s.secondary for s in analyze("à")] == [False]


def test_exactly_one_syllable_carries_the_primary_stress():
    for word in ("casa", "guarda-chuva", "chapéu de chuva", "a", "computador"):
        assert sum(1 for s in analyze(word) if s.stressed) == 1


def test_stress_does_not_alter_the_word():
    for word in ("guarda-chuva", "sòmente", "coração", "pau-d'água"):
        assert "".join(syllabify(word)) == word.lower()
