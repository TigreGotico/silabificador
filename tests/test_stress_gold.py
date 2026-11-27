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

from benchmark.stress_gold import from_rows  # noqa: E402

INFOPEDIA = "https://www.infopedia.pt/dicionarios/lingua-portuguesa/x"


def test_no_portal_row_enters_the_stress_gold():
    # Each word has a stress mark that aligns with its spelling, so only the
    # provenance filter can keep it out.
    rows = [
        ("casa", "pt-PT", "ˈkazɐ", "", INFOPEDIA),
        ("mesa", "pt-PT-x-lisboa", "ˈmezɐ", "m e z ɐ", ""),
        ("bola", "pt-PT", "ˈbɔlɐ", "b ɔ l ɐ", ""),
        ("gato", "pt-BR", "ˈgatu", "", ""),
    ]
    assert from_rows(rows) == {"casa": 0}


def test_an_infopedia_row_with_phones_is_not_used():
    assert from_rows([("casa", "pt-PT", "ˈkazɐ", "k a z ɐ", INFOPEDIA)]) == {}
