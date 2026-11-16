"""Hyphenation from the two Wiktionaries, read from their public dumps.

Both are edited by people, independently of each other and of the dictionaries:

* **en.wiktionary** writes ``{{hyph|pt|sub|li|nhar||su|bli|nhar}}`` under each
  Portuguese entry. Kaikki (Wiktextract) publishes it as ``hyphenations``, one
  item per alternative, so a word can carry more than one accepted split.
* **pt.wiktionary** writes the split inside the stress template of the
  Portuguese section: ``{{paroxítona|ex|cer|to}}``. Wiktextract drops these
  templates, so they are read from the Wikimedia XML dump directly.

Nothing here is scraped page by page. Each loader reads one file that its
project publishes for download.
"""
from __future__ import annotations

import bz2
import gzip
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, Set, Tuple

Split = Tuple[str, ...]

ENWIKT_URL = "https://kaikki.org/dictionary/Portuguese/kaikki.org-dictionary-Portuguese.jsonl.gz"
PTWIKT_URL = "https://dumps.wikimedia.org/ptwiktionary/latest/ptwiktionary-latest-pages-articles.xml.bz2"

#: The stress templates of pt.wiktionary. Their positional parameters are the
#: syllables; named ones (``id=``) are not.
_STRESS_TEMPLATE = re.compile(
    r"\{\{\s*(?:oxítona|paroxítona|proparoxítona(?:\s+aparente|\s+eventual)?)\s*\|([^{}]*)\}\}")
#: A level-1 language header: ``={{-pt-}}=``.
_LANGUAGE_HEADER = re.compile(r"^=\s*\{\{-([a-z-]+)-\}\}\s*=\s*$", re.M)


def download(url: str, cache_dir: str) -> str:
    """Fetch ``url`` into ``cache_dir`` once, and return the local path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        request = urllib.request.Request(url, headers={"User-Agent": "silabificador-benchmark"})
        with urllib.request.urlopen(request) as response, open(path + ".part", "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        os.replace(path + ".part", path)
    return path


def _add(table: Dict[str, Set[Split]], word: str, parts) -> None:
    split = tuple(p.strip().lower() for p in parts if p.strip())
    if split:
        table.setdefault(word.lower(), set()).add(split)


def load_enwiktionary(path: str) -> Dict[str, Set[Split]]:
    """Word -> every split en.wiktionary accepts, from a Kaikki JSONL(.gz) file."""
    table: Dict[str, Set[Split]] = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            for item in entry.get("hyphenations", ()):
                _add(table, entry["word"], item.get("parts", ()))
    return table


def portuguese_sections(text: str) -> Iterator[str]:
    """The body of every ``={{-pt-}}=`` section in a page's wikitext."""
    headers = list(_LANGUAGE_HEADER.finditer(text))
    for index, header in enumerate(headers):
        if header.group(1) == "pt":
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            yield text[header.end():end]


def splits_in(section: str) -> Iterator[Tuple[str, ...]]:
    """Every syllable list written in a stress template of ``section``."""
    for match in _STRESS_TEMPLATE.finditer(section):
        yield tuple(p for p in match.group(1).split("|") if "=" not in p)


def load_ptwiktionary(path: str) -> Dict[str, Set[Split]]:
    """Word -> every split pt.wiktionary gives, from a pages-articles XML(.bz2) dump."""
    table: Dict[str, Set[Split]] = {}
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as handle:
        title, namespace = "", ""
        for _, element in ET.iterparse(handle, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = element.text or ""
            elif tag == "ns":
                namespace = element.text or ""
            elif tag == "text" and namespace == "0":
                for section in portuguese_sections(element.text or ""):
                    for parts in splits_in(section):
                        _add(table, title, parts)
            elif tag == "page":
                element.clear()
    return table
