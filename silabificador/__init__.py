import os
from typing import List, Optional, Tuple, Any
from silabificador.syl import syllabify


class Syllabifier:
    """
    A Portuguese syllabifier that uses a trained Brill tagger model to split words into syllables.
    """

    def syllabify(self, word: str) -> List[str]:
        """
        Split a Portuguese word into syllables using IOB tagging.

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
