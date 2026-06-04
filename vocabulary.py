"""
vocabulary.py
Caption tokenization and vocabulary for the image captioning project.

The Vocabulary class:
  * tokenizes raw caption strings,
  * builds a word <-> index mapping from the training captions
    (rare words below a frequency threshold are dropped and later mapped
    to <unk>),
  * converts captions into lists of integer indices for the model.

Run this file directly to build the vocabulary from the dataset and save it:
    python vocabulary.py
This writes `vocab.pkl` into the outputs folder so that dataset.py, train.py
and evaluate.py all share the exact same vocabulary.
"""

import re
import pickle
from collections import Counter

import pandas as pd

from config import (
    VOCAB_FREQ_THRESHOLD,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
    CAPTIONS_FILE,
    OUTPUT_DIR,
)


class Vocabulary:
    """Maps words to integer indices and back."""

    def __init__(self, freq_threshold: int = VOCAB_FREQ_THRESHOLD):
        self.freq_threshold = freq_threshold

        # The four special tokens always get the lowest, fixed indices.
        # Keeping <pad> at index 0 lets us tell the loss function to ignore it.
        self.itos = {0: PAD_TOKEN, 1: START_TOKEN, 2: END_TOKEN, 3: UNK_TOKEN}
        self.stoi = {token: index for index, token in self.itos.items()}

    def __len__(self) -> int:
        return len(self.itos)

    # -- convenient access to the special-token indices ---------------------
    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.stoi[START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.stoi[END_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    # -- tokenization -------------------------------------------------------
    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase the text and split it into alphanumeric word tokens.

        Punctuation is discarded. Example:
            "A girl, going into a wooden building." -> ['a', 'girl', 'going',
            'into', 'a', 'wooden', 'building']
        """
        return re.findall(r"\w+", text.lower())

    # -- building the vocabulary -------------------------------------------
    def build_vocabulary(self, sentence_list) -> None:
        """Populate the mappings from an iterable of caption strings.

        Only words occurring at least `freq_threshold` times are added; the
        rest stay out of the vocabulary and become <unk> at numericalize time.
        """
        frequencies = Counter()
        for sentence in sentence_list:
            frequencies.update(self.tokenize(sentence))

        index = len(self.itos)  # start after the special tokens (i.e. at 4)
        # Sort for a deterministic vocabulary (same indices on every run).
        for word in sorted(frequencies):
            if frequencies[word] >= self.freq_threshold:
                self.stoi[word] = index
                self.itos[index] = word
                index += 1

    # -- using the vocabulary ----------------------------------------------
    def numericalize(self, text: str) -> list[int]:
        """Convert a caption string into a list of token indices.

        Out-of-vocabulary words map to <unk>. Does NOT add <start>/<end>;
        the Dataset is responsible for wrapping sequences with those.
        """
        return [self.stoi.get(token, self.unk_idx) for token in self.tokenize(text)]

    def decode(self, indices) -> str:
        """Turn a list of indices back into a readable string.

        Special tokens are skipped, which is handy for printing generated
        captions during evaluation.
        """
        special = {self.pad_idx, self.start_idx, self.end_idx}
        words = [self.itos[i] for i in indices if i not in special]
        return " ".join(words)

    # -- persistence --------------------------------------------------------
    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_and_save_vocabulary(save_path=None) -> Vocabulary:
    """Read the captions file, build the vocabulary, save it, and report."""
    if save_path is None:
        save_path = OUTPUT_DIR / "vocab.pkl"

    captions_df = pd.read_csv(CAPTIONS_FILE)

    vocab = Vocabulary()
    vocab.build_vocabulary(captions_df["caption"].tolist())
    vocab.save(save_path)

    print(f"Vocabulary built from {len(captions_df)} captions.")
    print(f"Vocabulary size (incl. 4 special tokens): {len(vocab)}")
    print(f"Saved to: {save_path}")

    # Small sanity check on the first caption.
    example = captions_df["caption"].iloc[0]
    encoded = [vocab.start_idx] + vocab.numericalize(example) + [vocab.end_idx]
    print("\nExample")
    print("  caption :", example)
    print("  encoded :", encoded)
    print("  decoded :", vocab.decode(encoded))

    return vocab


if __name__ == "__main__":
    build_and_save_vocabulary()
