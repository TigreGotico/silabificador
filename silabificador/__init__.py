import os
import joblib

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class Syllabifier:
    def __init__(self, region="lbx"):
        self.region = region
        self.model = self._load_model(region)

    @staticmethod
    def _load_model(region):
        model_path = os.path.join(_MODELS_DIR, f"{region}_brill_syllabifier.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Syllabifier model not found for region '{region}'")
        return joblib.load(model_path)

    def syllabify(self, word: str):
        # Predict IOB tags
        tags = self.model.tag(list(word))
        syllables = []
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