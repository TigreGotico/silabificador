import random

import joblib
from datasets import load_dataset
from joblib import Parallel, delayed
from nltk.tag import UnigramTagger, BigramTagger, brill, brill_trainer, DefaultTagger
from sklearn.model_selection import train_test_split


# --------- Syllabifier Features ---------

def validate_entry(entry, region: str = ""):
    if entry["region_code"] != region:
        return False
    word = entry["word"]
    syllables = entry["syllables"].split("|")
    if "".join(syllables) != word.replace("-", ""):
        print(f"BAD ENTRY: word doesn't match syllables - {entry}")
        return False
    return True


def prepare_syllabifier_dataset(entries, region: str = ""):
    X, y = [], []
    for entry in entries:
        if not validate_entry(entry, region):
            continue
        word = entry["word"]
        syllables = entry["syllables"].split("|")
        labels = ["I"] * len(word)
        idx = 0
        try:
            for syl in syllables:
                labels[idx] = "B"
                idx += len(syl)
        except IndexError:
            print(f"BAD ENTRY: {entry}")
            continue
        if len(labels) != len(word):
            raise RuntimeError("no, can't happen")
        X.append([word[i] for i in range(len(word))])
        y.append(labels)
    return X, y


# --------- Loading from Hugging Face ---------
def load_dataset_from_huggingface():
    dataset = load_dataset("Jarbas/portuguese_phonetic_lexicon", split="train")
    return [entry for entry in dataset if entry["syllables"] and entry["phones"]]


# --------- Saving and Loading Models ---------
def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


# --------- NLTK Taggers ---------
def prepare_nltk_train_data(X, y):
    train_data = []
    for syllables, labels in zip(X, y):
        sentence = list(zip(syllables, labels))
        train_data.append(sentence)
    return train_data


def train_ngram_and_brill(train_data):
    # Ngram chain
    default_tagger = DefaultTagger('UNK')
    unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)
    bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)

    # Brill tagger on top
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger=bigram_tagger, templates=brill.fntbl37())
    brill_tagger = trainer.train(train_data, max_rules=100, min_score=3)
    return unigram_tagger, bigram_tagger, brill_tagger


# --------- Main Training Routine ---------
def train_for_region(region, entries):
    # Syllabifier
    X_syll, y_syll = prepare_syllabifier_dataset(entries, region)
    X_syll_train, X_syll_test, y_syll_train, y_syll_test = train_test_split(X_syll, y_syll, test_size=0.1)

    # Prepare NLTK training data
    train_data = prepare_nltk_train_data(X_syll_train, y_syll_train)

    # Train Ngram + Brill tagger
    unigram_tagger, bigram_tagger, brill_tagger = train_ngram_and_brill(train_data)

    # Save model
    print(f"[*] Model saved: {region}_brill_syllabifier.pkl")

    # Evaluate the model
    test_data = prepare_nltk_train_data(X_syll_test, y_syll_test)

    unigram_acc = unigram_tagger.accuracy(test_data)
    bigram_acc = bigram_tagger.accuracy(test_data)
    brill_acc = brill_tagger.accuracy(test_data)
    print(region, unigram_acc, bigram_acc, brill_acc)

    print(f"[{region} - Syllabifier] Evaluation accuracy: {brill_acc:.4f}")

    # Save models
    save_model(unigram_tagger, f"{region}_unigram_syllabifier.pkl")
    save_model(bigram_tagger, f"{region}_bigram_syllabifier.pkl")
    save_model(brill_tagger, f"{region}_brill_syllabifier.pkl")

    return {
        "region": region,
        "unigram": unigram_acc,
        "bigram": bigram_acc,
        "brill": brill_acc
    }


def train_models():
    entries = load_dataset_from_huggingface()
    random.shuffle(entries)

    regions = ["lbx", "lbn", "lda", "rjx", "rjo", "spx", "spo", "mpx", "map", "dli"]
    print("[*] Training taggers for each region...\n")

    # Parallel execution
    results = Parallel(n_jobs=-1)(
        delayed(train_for_region)(region, entries) for region in regions
    )

    # --------- Visualization ---------
    print("\n\nAccuracy Table:")
    print(f"{'Region':<8} | {'Unigram':>10} | {'Bigram':>10} | {'Brill':>10}")
    print("-" * 40)
    for res in results:
        if res:
            print(
                f"{res['region']:<8} | {res['unigram'] * 100:>9.2f} | {res['bigram'] * 100:>9.2f} | {res['brill'] * 100:>9.2f}"
            )


if __name__ == "__main__":
    train_models()
    # Accuracy Table:
    # Region        Affix |    Unigram |     Bigram |    Trigram | Brill (bi) | Brill (tri)
    # ----------------------------------------
    # lbx           0.00      82.59     85.17     84.22     98.47     97.57
    # lbn           0.00      82.73     85.36     85.00     98.51     97.35
    # lda           0.00      82.81     85.39     84.41     98.55     97.77
    # rjx           0.00      82.51     85.35     84.07     98.45     97.55
    # rjo           0.00      82.76     85.49     84.94     98.67     97.50
    # spx           0.00      82.73     85.32     84.77     98.51     97.37
    # spo           0.00      82.83     85.22     84.91     98.33     97.27
    # mpx           0.00      82.86     85.52     84.51     98.54     97.60
    # map           0.00      82.89     85.50     84.37     98.58     97.75
    # dli           0.00      82.67     85.10     84.77     98.47     97.37
