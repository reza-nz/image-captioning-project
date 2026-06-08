"""
tests/test_evaluate.py
Unit tests for the evaluation utilities (references, BLEU, generation).

Run from the project root with:
    pytest

The BLEU and reference tests are pure and instant. The generation test uses a
few tiny images and a small model with pretrained=False, so still no download.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from vocabulary import Vocabulary
from dataset import build_transform
from model import CaptioningModel
from evaluate import build_references, compute_bleu, align, generate_hypotheses


# --------------------------------------------------------------------------
# build_references
# --------------------------------------------------------------------------
def test_build_references_groups_captions_by_image():
    df = pd.DataFrame([
        {"image": "a.jpg", "caption": "a dog runs"},
        {"image": "a.jpg", "caption": "a puppy plays"},
        {"image": "b.jpg", "caption": "a cat sleeps"},
    ])
    references = build_references(df)

    assert set(references.keys()) == {"a.jpg", "b.jpg"}
    assert len(references["a.jpg"]) == 2          # two captions for a.jpg
    assert references["a.jpg"][0] == ["a", "dog", "runs"]  # tokenized
    assert len(references["b.jpg"]) == 1


# --------------------------------------------------------------------------
# compute_bleu
# --------------------------------------------------------------------------
def test_bleu_is_perfect_when_hypothesis_matches_reference():
    references = [[["a", "small", "dog", "runs", "in", "the", "park"]]]
    hypotheses = [["a", "small", "dog", "runs", "in", "the", "park"]]
    scores = compute_bleu(references, hypotheses)
    # An exact match should score essentially 1.0 across all n.
    for n in range(1, 5):
        assert scores[f"BLEU-{n}"] > 0.99


def test_bleu_is_low_when_nothing_overlaps():
    references = [[["a", "small", "dog", "runs", "in", "the", "park"]]]
    hypotheses = [["completely", "different", "unrelated", "words", "here"]]
    scores = compute_bleu(references, hypotheses)
    # No shared unigrams -> BLEU-1 should be very low.
    assert scores["BLEU-1"] < 0.1


def test_bleu_returns_all_four_scores():
    references = [[["a", "dog", "runs", "fast"]]]
    hypotheses = [["a", "dog", "walks", "slow"]]
    scores = compute_bleu(references, hypotheses)
    assert set(scores.keys()) == {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"}


# --------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------
def test_align_keeps_references_and_hypotheses_in_step():
    references = {
        "a.jpg": [["a", "dog"]],
        "b.jpg": [["a", "cat"]],
    }
    hypotheses = {
        "b.jpg": ["a", "cat"],   # deliberately only b.jpg, to test selection
    }
    refs, hyps = align(references, hypotheses)
    assert len(refs) == 1
    assert len(hyps) == 1
    assert refs[0] == [["a", "cat"]]
    assert hyps[0] == ["a", "cat"]


# --------------------------------------------------------------------------
# generate_hypotheses (end-to-end on fake images)
# --------------------------------------------------------------------------
def test_generate_hypotheses_produces_token_lists():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        images_dir = Path(tmp) / "Images"
        images_dir.mkdir(parents=True)
        names = []
        for i in range(3):
            name = f"img_{i}.jpg"
            Image.new("RGB", (200, 200), color=(i * 40, 100, 150)).save(images_dir / name)
            names.append(name)

        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary(["a small dog is running in the park"])

        model = CaptioningModel(vocab_size=len(vocab), embed_size=32,
                                hidden_size=64, dropout=0.0, pretrained=False)

        hypotheses = generate_hypotheses(model, names, images_dir,
                                         build_transform(), vocab,
                                         device="cpu", max_length=10)

        assert set(hypotheses.keys()) == set(names)
        for tokens in hypotheses.values():
            assert isinstance(tokens, list)
            assert all(isinstance(t, str) for t in tokens)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print("=== test_evaluate ===\n")
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed  {failed} failed")
    if failed:
        sys.exit(1)
