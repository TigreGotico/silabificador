import os
from typing import List, Optional, Tuple, Any
import joblib

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class Syllabifier:
    """
    A Portuguese syllabifier that uses a trained Brill tagger model to split words into syllables.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the Syllabifier.

        Args:
            model_path (Optional[str]): Path to the pre-trained syllabification model.
        """
        model_path = model_path or os.path.join(_MODELS_DIR, "lbx_brill_syllabifier.pkl")
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        """
        Load the trained Brill tagger model from the specified path.

        Args:
            model_path (str): Path to the `.pkl` file containing the trained model.

        Returns:
            Any: A Brill tagger model used for IOB tagging of syllables.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Syllabifier model not found: '{model_path}'")
        return joblib.load(model_path)

    def syllabify(self, word: str) -> List[str]:
        """
        Split a Portuguese word into syllables using IOB tagging.

        Args:
            word (str): The input word to be syllabified.

        Returns:
            List[str]: A list of syllables as strings.
        """
        tags: List[Tuple[str, str]] = self.model.tag(list(word))
        syllables: List[str] = []
        current = ""

        for char, tag in tags:
            if tag == "B" and current:
                syllables.append(current)
                current = char
            else:
                current += char

        if current:
            syllables.append(current)

        return syllables


if __name__ == "__main__":
    s = Syllabifier()
    print(s.syllabify("computador"))
    # ['com', 'pu', 'ta', 'dor']
