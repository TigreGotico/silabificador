"""Stress assignment."""
import pytest

from silabificador import analyze, stress, stressed_index, syllabify


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


@pytest.mark.parametrize("word,expected", [
    ("pau-d'alho", "(pau-).D'A.lho"),
    ("estrela-d'alva", "es.(tre).la-.D'AL.va"),
    ("copo d'água", "(co).po .D'Á.gua"),
    ("caiu d'água", "ca.(iu ).D'Á.gua"),
    ("caiu-o d'água", "ca.(iu-).o .D'Á.gua"),
    ("saiu d'aqui", "sa.(iu ).d'a.QUI"),
    ("saiu-se-lhe d'aqui", "sa.(iu-).se-.lhe .d'a.QUI"),
    ("sant'ana", "san.T'A.na"),
    ("d'os lusíadas", "d'os .lu.SÍ.a.das"),
    ("bê-á-'bá", "(bê-).(á-').BÁ"),
    ("três pa'anga", "(três ).pa'.AN.ga"),
    ("enguia-d’água-doce", "en.(gui).a-.(d’á).gua-.DO.ce"),
    ("d’os lusíadas", "d’os .lu.SÍ.a.das"),
    ("bê-á-’bá", "(bê-).(á-’).BÁ"),
])
def test_an_apostrophe_does_not_end_a_stress_domain(word, expected):
    # The elided d' is read with the word after it, so d'água is one element
    # and every element keeps its stress. d'Os reads dos by analogy with AO90
    # Base XVIII 1 a) (a Os = aos), an unstressed monosyllable. A hyphen still
    # ends its element when an apostrophe follows it, as in bê-á-'bá. The
    # apostrophe inside the one word pa'anga does not split it either.
    assert marked(word) == expected


def test_the_module_stressed_index_reads_the_same_elements():
    # pé-d'alho has two elements; the last, d'alho, is paroxytone (rule 3), and
    # the acute of pé belongs to the first element only.
    assert stress.stressed_index(["pé-", "d'a", "lho"], "pé-d'alho") == 1
    assert stress.stressed_index(["três ", "pa'", "an", "ga"], "três pa'anga") == 2


def test_clitics_inside_a_compound_carry_nothing():
    # "de" is a monossilabo atono: an article or preposition is not a stress
    # domain, so it is neither primary nor secondary.
    syllables = analyze("chapéu de chuva")
    de = [s for s in syllables if str(s).strip() == "de"][0]
    assert not de.stressed and not de.secondary
    assert marked("chapéu de chuva") == "cha.(péu ).de .CHU.va"


@pytest.mark.parametrize("word,expected", [
    ("por favor", "por .fa.VOR"),
    ("pelo rio", "pe.lo .RI.o"),
    ("pela estrada", "pe.la .es.TRA.da"),
    ("pão com manteiga", "(pão ).com .man.TEI.ga"),
    ("café sem açúcar", "ca.(fé ).sem .a.ÇÚ.car"),
])
def test_a_preposition_before_a_word_carries_nothing(word, expected):
    assert marked(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("à-vontade", "à-.von.TA.de"),
    ("às vezes", "às .VE.zes"),
    ("pelos campos", "pe.los .CAM.pos"),
    ("pelas ruas", "pe.las .RU.as"),
    ("num dia", "num .DI.a"),
    ("nuns dias", "nuns .DI.as"),
    ("dum lado", "dum .LA.do"),
    ("duns tempos", "duns .TEM.pos"),
    ("frei luís", "frei .lu.ÍS"),
])
def test_every_listed_unstressed_word_carries_nothing(word, expected):
    # Cunha & Cintra, Nova Gramática (1984), pp. 56-57, list as monossílabos
    # átonos "as combinações de preposição e artigo: à, ao, da, do, na, no, num,
    # etc." and "as formas de tratamento dom, frei, são, seu". AO90 Base IX 9
    # names pelo(s) and pela(s) "palavras proclíticas".
    assert marked(word) == expected


@pytest.mark.parametrize("word,elements", [("ao pé", 1), ("aos poucos", 2)])
def test_ao_carries_no_stress(word, elements):
    # Whatever the syllables of ao and aos, none of them is stressed.
    syllables = analyze(word)
    assert not any(s.stressed or s.secondary for s in syllables[:-elements])


@pytest.mark.parametrize("word,expected", [
    ("amá-lo", "a.MÁ-.lo"),
    ("dá-se", "DÁ-.se"),
    ("deixa-o", "DEI.xa-.o"),
    ("partir-lhe", "par.TIR-.lhe"),
    ("vendeu-se", "ven.DEU-.se"),
    ("deu-te-lo", "DEU-.te-.lo"),
    ("revelou-mo", "re.ve.LOU-.mo"),
    ("revelou-m'o", "re.ve.LOU-.m'o"),
    ("fazem-no", "FA.zem-.no"),
    ("fazemo-lo", "fa.ZE.mo-.lo"),
    ("mostrou-vo-los", "mos.TROU-.vo-.los"),
    ("eis-me", "EIS-.me"),
])
def test_an_enclitic_pronoun_leaves_the_primary_on_the_verb(word, expected):
    # AO90 Base XVII 1: "Emprega-se o hífen na ênclise e na tmese: amá-lo,
    # dá-se, deixa-o, partir-lhe". The pronoun is a monossílabo átono (Cunha &
    # Cintra pp. 56-57), so the verb keeps the stress it has alone.
    assert marked(word) == expected


@pytest.mark.parametrize("word", [
    "deu-me", "deu-te", "deu-se", "deu-lhe", "deu-nos", "deu-vos", "deu-lhes",
    "deu-o", "deu-a", "deu-os", "deu-as", "deu-mo", "deu-ma", "deu-mos", "deu-mas",
    "deu-to", "deu-ta", "deu-tos", "deu-tas", "deu-lho", "deu-lha", "deu-lhos",
    "deu-lhas", "dão-no", "dão-na", "dão-nos", "dão-nas",
])
def test_no_clitic_pronoun_takes_the_stress_from_its_verb(word):
    # Cunha & Cintra pp. 56-57: "os pronomes pessoais oblíquos me, te, se, o, a,
    # lhe, nos, vos, os, as, lhes e suas combinações: mo, to, lho, etc.". After a
    # verb that ends in a nasal, o and a are written no and na: fazem-no.
    verb, pronoun = word.split("-")
    assert marked(word) == f"{verb.upper()}-.{pronoun}"


@pytest.mark.parametrize("word,expected", [
    ("vê-lo", "VÊ-.lo"), ("vê-la", "VÊ-.la"), ("vê-los", "VÊ-.los"), ("vê-las", "VÊ-.las"),
])
def test_lo_and_la_are_clitic_pronouns(word, expected):
    # AO90 Base VIII 2 b): "as formas verbais oxítonas, quando conjugadas com os
    # pronomes clíticos -lo(s) ou -la(s)": vê-la(s).
    assert marked(word) == expected


def test_stressed_index_skips_an_enclitic_pronoun():
    assert stressed_index("vendeu-se") == 1
    assert stressed_index("deu-te-lo") == 0
    assert stress.stressed_index(["ven", "deu-", "se"], "vendeu-se") == 1
    assert stress.stressed_index(["re", "ve", "lou-", "m'o"], "revelou-m'o") == 2


@pytest.mark.parametrize("word,expected", [
    ("amá-lo-ei", "a.(má-).lo-.EI"),
    ("enviar-lhe-emos", "en.vi.(ar-).lhe-.E.mos"),
    ("dar-te-ei", "(dar-).te-.EI"),
    ("contribuir-lhe-ia", "con.tri.bu.(ir-).lhe-.I.a"),
    ("fá-lo-ás", "(fá-).lo-.ÁS"),
    ("dar-vo-la-ia", "(dar-).vo-.la-.I.a"),
])
def test_in_tmesis_the_verb_ending_keeps_the_primary(word, expected):
    # The future or conditional ending is part of the verb, and it comes last.
    # AO90 Base VIII 1 b) writes fá-lo(s)-ás: the acute on fá marks a "vogal
    # tónica", and the acute on ás marks the stressed ending.
    assert marked(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("guarda-chuva", "(guar).da-.CHU.va"),
    ("bem-te-vi", "(bem-).te-.VI"),
    ("arco-íris", "(ar).co-.Í.ris"),
    ("pé-de-meia", "(pé-).de-.MEI.a"),
    ("trinta-e-um", "(trin).ta-.e-.UM"),   # um is the numeral, not a pronoun
    ("no-lo", "no-.LO"),                    # no verb to lean on
    ("vitamina a", "vi.ta.(mi).na .A"),     # a space is not the hyphen of enclisis
])
def test_a_compound_without_an_enclitic_keeps_its_stress(word, expected):
    assert marked(word) == expected


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
    for word in ("casa", "guarda-chuva", "chapéu de chuva", "a", "computador",
                 "vendeu-se", "deu-te-lo", "no-lo", "à-vontade", "dar-te-ei", "a-"):
        assert sum(1 for s in analyze(word) if s.stressed) == 1


def test_stress_does_not_alter_the_word():
    for word in ("guarda-chuva", "sòmente", "coração", "pau-d'água"):
        assert "".join(syllabify(word)) == word.lower()
