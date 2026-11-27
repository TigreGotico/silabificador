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

import json  # noqa: E402

from benchmark.report import RESULTS, render_markdown  # noqa: E402

PAGE = Path(__file__).resolve().parent.parent / "docs" / "accuracy.md"


def test_accuracy_page_matches_committed_results():
    with open(RESULTS, encoding="utf-8") as fh:
        results = json.load(fh)
    assert PAGE.read_text(encoding="utf-8") == render_markdown(results)
