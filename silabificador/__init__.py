import os
from typing import List, Optional, Tuple, Any
from silabificador.syl import syllabify


class Syllabifier:
    """
    A Portuguese syllabifier that splits words into syllables using hand-crafted
    phonotactic rules (onset/nucleus/coda, sonority sequencing, diphthong/hiatus
    handling). Rule-based and dependency-free; no model is loaded.
    """

    def syllabify(self, word: str) -> List[str]:
        """
        Split a Portuguese word into syllables.

        Args:
            word (str): The input word to be syllabified.

        Returns:
            List[str]: A list of syllables as strings.
        """
        return syllabify(word)


if __name__ == "__main__":
    s = Syllabifier()
    print(s.syllabify("computador"))
    # ['com', 'pu', 'ta', 'dor']
