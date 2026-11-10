"""Unit tests for the gold-set builder.

These exercise the tier, provenance and integrity logic on synthetic data; no
corpus is downloaded. Lexicon rows are
``(word, syllables, infopedia_url, michaelis_url)`` -- an Infopédia or Michaelis
row carries its dictionary URL and dot separators, a Portal row has no URL and
uses pipes. Wiktionary sources are word -> set of splits.
"""
import importlib.util
import sys
from pathlib import Path

# ``benchmark`` is a development harness, and the wheel does not ship it:
# packages.find includes only ``silabificador*``. So it cannot be imported by
# name once the checkout is off sys.path, which is what --import-mode=importlib
# gives us. Load the package from its own path and register it, so that its
# absolute imports keep resolving. Nothing is added to sys.path, so the
# INSTALLED silabificador stays the one under test.
_BENCHMARK = Path(__file__).resolve().parent.parent / "benchmark"
if "benchmark" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "benchmark",
        _BENCHMARK / "__init__.py",
        submodule_search_locations=[str(_BENCHMARK)],
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["benchmark"] = _module
    _spec.loader.exec_module(_module)

from benchmark import wiktionary  # noqa: E402
from benchmark.gold import accepts, build, fuse, lexicon_sources, summary  # noqa: E402

URL = "https://www.infopedia.pt/dicionarios/lingua-portuguesa/x"
MICHAELIS_URL = "https://michaelis.uol.com.br/moderno-espanhol/busca/portugues-espanhol/x/"


def info(word, syllables):
    return (word, syllables, URL, "")


def michaelis(word, syllables):
    return (word, syllables, "", MICHAELIS_URL)


def portal(word, syllables):
    return (word, syllables, "", "")


def split(text):
    return tuple(text.split("."))


def gold_of(rows=(), en=None, pt=None):
    raw = dict(lexicon_sources(rows))
    if en is not None:
        raw["enwiktionary"] = {w: {split(s) for s in ss} for w, ss in en.items()}
    if pt is not None:
        raw["ptwiktionary"] = {w: {split(s) for s in ss} for w, ss in pt.items()}
    return build(raw)


def test_two_human_sources_that_agree_are_gold():
    gold = gold_of([info("casa", "ca.sa")], en={"casa": ["ca.sa"]})
    assert gold.agreement == {"casa": frozenset({("ca", "sa")})}
    assert gold.support == {"casa": 2}


def test_the_portal_is_generated_and_never_gold():
    # Infopédia and the Portal agree, but the Portal is a rule system: one
    # human source is all the word has.
    gold = gold_of([info("casa", "ca.sa"), portal("casa", "ca|sa")])
    assert not gold.agreement
    assert gold.infopedia_only == {"casa": frozenset({("ca", "sa")})}
    assert gold.generated == {"casa": frozenset({("ca", "sa")})}


def test_every_split_two_sources_give_is_accepted():
    # en.wiktionary lists both splits of sublinhar; pt.wiktionary gives both too.
    gold = gold_of(en={"sublinhar": ["sub.li.nhar", "su.bli.nhar"]},
                   pt={"sublinhar": ["sub.li.nhar", "su.bli.nhar"]})
    assert gold.agreement["sublinhar"] == {split("sub.li.nhar"), split("su.bli.nhar")}
    assert accepts(gold.agreement["sublinhar"], ["sub", "li", "nhar"])
    assert accepts(gold.agreement["sublinhar"], ["su", "bli", "nhar"])
    assert not accepts(gold.agreement["sublinhar"], ["subl", "i", "nhar"])


def test_a_split_one_source_alone_gives_is_not_accepted_in_agreement():
    gold = gold_of([info("apto", "ap.to")], en={"apto": ["a.pto", "ap.to"]})
    assert gold.agreement["apto"] == {split("ap.to")}


def test_a_michaelis_row_is_a_human_source_not_the_portal():
    gold = gold_of([michaelis("azia", "a.zi.a")])
    assert gold.michaelis_only == {"azia": {split("a.zi.a")}}
    assert not gold.generated and not gold.wiktionary_only


def test_michaelis_and_infopedia_that_agree_are_gold():
    gold = gold_of([info("calva", "cal.va"), michaelis("calva", "cal.va"),
                    portal("calva", "cal|va")])
    assert gold.agreement == {"calva": {split("cal.va")}}
    assert gold.support == {"calva": 2}


def test_four_sources_record_support_four():
    gold = gold_of([info("casa", "ca.sa"), michaelis("casa", "ca.sa")],
                   en={"casa": ["ca.sa"]}, pt={"casa": ["ca.sa"]})
    assert gold.support["casa"] == 4
    assert dict(gold.tiers())["agreement, 4 sources"] == {"casa": {split("ca.sa")}}


def test_the_summary_counts_michaelis_only_words():
    gold = gold_of([michaelis("azia", "a.zi.a")])
    assert "michaelis-only        1   one human source" in summary(gold)


def test_three_sources_record_support_three():
    gold = gold_of([info("casa", "ca.sa")], en={"casa": ["ca.sa"]}, pt={"casa": ["ca.sa"]})
    assert gold.support["casa"] == 3


def test_human_sources_that_share_no_split_are_a_conflict_not_gold():
    gold = gold_of([info("ablactar", "ab.lac.tar")], pt={"ablactar": ["a.blac.tar"]})
    assert not gold.agreement
    assert gold.conflict == {"ablactar": {split("ab.lac.tar"), split("a.blac.tar")}}


def test_single_source_words_are_kept_but_segregated():
    gold = gold_of([info("aachenense", "aa.che.nen.se")], en={"xisto": ["xis.to"]})
    assert gold.infopedia_only == {"aachenense": {split("aa.che.nen.se")}}
    assert gold.wiktionary_only == {"xisto": {split("xis.to")}}
    assert not gold.agreement


def test_splits_that_do_not_reconstruct_the_word_are_dropped():
    # The Portal deletes the hyphen; Infopédia drops a syllable outright. A
    # source that cannot spell the headword is not evidence about the headword.
    gold = gold_of([
        portal("a-propósito", "a|pro|pó|si|to"),
        info("acidentadamente", "a.ci.den.ta.men.te"),
    ])
    assert not gold.infopedia_only and not gold.generated
    assert gold.dropped == {"infopedia": 1, "michaelis": 0, "portal": 1}


def test_a_headword_printed_without_division_is_dropped():
    # Michaelis prints these entries with no syllable division (its <es> element
    # reads "bacon", "arroz-doce"); pt.wiktionary has "metodismo" undivided.
    gold = gold_of([michaelis("bacon", "bacon"), michaelis("arroz-doce", "arroz-doce")],
                   pt={"metodismo": ["metodismo"]})
    assert not gold.michaelis_only and not gold.wiktionary_only
    assert gold.dropped["michaelis"] == 2
    assert gold.dropped["ptwiktionary"] == 1


def test_a_monosyllable_and_a_hyphen_divided_compound_are_kept():
    gold = gold_of([michaelis("pão", "pão"), michaelis("quais", "quais"),
                    info("bem-te-vi", "bem-te-vi"), info("a-histórico", "a-his.tó.ri.co")])
    assert gold.michaelis_only == {"pão": {("pão",)}, "quais": {("quais",)}}
    assert gold.infopedia_only == {"bem-te-vi": {("bem-te-vi",)},
                                   "a-histórico": {("a-his", "tó", "ri", "co")}}
    assert gold.dropped == {"infopedia": 0, "michaelis": 0, "portal": 0}


def test_a_dictionary_contradicting_itself_is_dropped():
    gold = gold_of([info("casa", "ca.sa"), info("casa", "cas.a")])
    assert gold.dropped["infopedia"] == 1
    assert not gold.infopedia_only


def test_a_wiktionary_listing_alternatives_keeps_both():
    gold = gold_of(en={"apto": ["a.pto", "ap.to"]})
    assert gold.wiktionary_only == {"apto": {split("a.pto"), split("ap.to")}}
    assert gold.dropped["enwiktionary"] == 0


def test_headwords_are_case_folded():
    gold = gold_of([info("Adonai", "A.do.nai")], en={"adonai": ["a.do.nai"]})
    assert gold.agreement == {"adonai": {split("a.do.nai")}}


def test_portal_regional_duplicates_collapse():
    # The orthographic split is region-invariant; the lexicon repeats it once
    # per phonetic region.
    gold = gold_of([portal("casa", "ca|sa") for _ in range(6)])
    assert gold.generated == {"casa": {split("ca.sa")}}
    assert gold.dropped["portal"] == 0


def test_fuse_merges_across_a_separator():
    # The dictionaries mark hyphenation points and never break at a hyphen, so
    # their token holds two nuclei. Fusing puts both notations in one shape.
    assert fuse(("a-", "his", "tó", "ri", "co")) == ("a-his", "tó", "ri", "co")
    assert fuse(("ca", "sa")) == ("ca", "sa")


def test_hyphen_notation_and_engine_notation_are_accepted_alike():
    gold = gold_of([info("ab-rogar", "ab-ro.gar")], en={"ab-rogar": ["ab-ro.gar"]})
    assert accepts(gold.agreement["ab-rogar"], ["ab-", "ro", "gar"])


PAGE = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"><page><title>{title}</title>
<ns>{ns}</ns><revision><text>{text}</text></revision></page></mediawiki>"""


def test_ptwiktionary_reads_the_portuguese_section_only(tmp_path):
    text = ("={{-gl-}}=\n{{paroxítona|ga|le|go}}\n"
            "={{-pt-}}=\n{{paroxítona|sub|li|nhar}} {{oxítona|su|bli|nhar|id=pt}}\n")
    dump = tmp_path / "dump.xml"
    dump.write_text(PAGE.format(title="sublinhar", ns=0, text=text), encoding="utf-8")
    assert wiktionary.load_ptwiktionary(str(dump)) == {
        "sublinhar": {("sub", "li", "nhar"), ("su", "bli", "nhar")}}


def test_ptwiktionary_skips_other_namespaces(tmp_path):
    dump = tmp_path / "dump.xml"
    dump.write_text(PAGE.format(title="Ajuda:casa", ns=12, text="={{-pt-}}=\n{{paroxítona|ca|sa}}"),
                    encoding="utf-8")
    assert wiktionary.load_ptwiktionary(str(dump)) == {}


def test_enwiktionary_keeps_every_listed_alternative(tmp_path):
    path = tmp_path / "pt.jsonl"
    path.write_text('{"word": "Apto", "hyphenations": [{"parts": ["a", "pto"]}, {"parts": ["ap", "to"]}]}\n'
                    '{"word": "casa"}\n', encoding="utf-8")
    assert wiktionary.load_enwiktionary(str(path)) == {"apto": {("a", "pto"), ("ap", "to")}}
