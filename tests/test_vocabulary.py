"""
tests/test_vocabulary.py
Unit tests for the Vocabulary class.

Run from the project root with:
    pytest

These tests use a tiny in-memory set of captions, so they run instantly and
do NOT require the Flickr8k dataset to be downloaded.
"""

import sys
from pathlib import Path

# Allow importing project modules (config.py, vocabulary.py) from the repo root
# when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vocabulary import Vocabulary

# Flickr8k-style sample captions used across the tests.
SAMPLE_CAPTIONS = [
    "A child in a pink dress is climbing up a set of stairs .",
    "A girl going into a wooden building .",
    "A little girl climbing into a wooden playhouse .",
    "A little girl climbing the stairs to her playhouse .",
    "A little girl in a pink dress going into a wooden cabin .",
]


def make_vocab(threshold: int = 2) -> Vocabulary:
    vocab = Vocabulary(freq_threshold=threshold)
    vocab.build_vocabulary(SAMPLE_CAPTIONS)
    return vocab


def test_special_tokens_have_fixed_indices():
    vocab = Vocabulary()
    assert vocab.pad_idx == 0
    assert vocab.start_idx == 1
    assert vocab.end_idx == 2
    assert vocab.unk_idx == 3
    # Before building, only the four special tokens exist.
    assert len(vocab) == 4


def test_tokenize_lowercases_and_strips_punctuation():
    tokens = Vocabulary.tokenize("A girl, going INTO a building.")
    assert tokens == ["a", "girl", "going", "into", "a", "building"]


def test_frequency_threshold_keeps_and_drops_words():
    vocab = make_vocab(threshold=2)
    assert "wooden" in vocab.stoi      # occurs 3x -> kept
    assert "cabin" not in vocab.stoi   # occurs 1x -> dropped


def test_numericalize_maps_oov_to_unk():
    vocab = make_vocab(threshold=2)
    ids = vocab.numericalize("a purple dress")  # 'purple' is out of vocabulary
    assert vocab.unk_idx in ids
    # A frequent word like 'a' should map to its own index, not <unk>.
    assert ids[0] == vocab.stoi["a"]


def test_numericalize_does_not_add_start_or_end():
    vocab = make_vocab(threshold=2)
    ids = vocab.numericalize("a girl")
    assert vocab.start_idx not in ids
    assert vocab.end_idx not in ids


def test_decode_skips_special_tokens_but_keeps_unk():
    vocab = make_vocab(threshold=2)
    ids = [vocab.start_idx] + vocab.numericalize("a wooden building") + [vocab.end_idx]
    decoded = vocab.decode(ids)
    assert "<start>" not in decoded
    assert "<end>" not in decoded
    assert "wooden" in decoded
    assert "<unk>" in decoded  # 'building' occurs once -> dropped -> <unk>


def test_build_vocabulary_is_deterministic():
    assert make_vocab().stoi == make_vocab().stoi


def test_len_grows_after_building():
    vocab = make_vocab(threshold=2)
    assert len(vocab) > 4  # special tokens plus real words


def test_save_and_load_roundtrip(tmp_path):
    vocab = make_vocab()
    path = tmp_path / "vocab.pkl"
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.stoi == vocab.stoi
    assert loaded.itos == vocab.itos
    assert len(loaded) == len(vocab)
