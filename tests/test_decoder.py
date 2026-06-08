"""
tests/test_decoder.py
Unit tests for the DecoderRNN.

Run from the project root with:
    pytest

All tests use random tensors and a tiny fake vocabulary, so they need NO
dataset download and run in well under a second.
"""

import sys
from pathlib import Path

# Allow importing project modules (config.py, decoder.py, encoder.py) from the
# repo root when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from decoder import DecoderRNN


VOCAB_SIZE = 50
EMBED_SIZE = 32
HIDDEN_SIZE = 64


def make_decoder(num_layers: int = 1):
    return DecoderRNN(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE,
                      hidden_size=HIDDEN_SIZE, num_layers=num_layers,
                      dropout=0.0)


def make_features(batch_size: int = 4):
    """Fake encoder output: [B, EMBED_SIZE]."""
    return torch.randn(batch_size, EMBED_SIZE)


def make_captions(batch_size: int = 4, length: int = 10):
    """Fake captions of word indices: [B, length]."""
    return torch.randint(0, VOCAB_SIZE, (batch_size, length))


def test_forward_output_shape():
    decoder = make_decoder()
    features = make_features(4)
    captions = make_captions(4, length=10)
    outputs = decoder(features, captions)
    # One score vector over the vocabulary for each of the T timesteps.
    assert outputs.shape == (4, 10, VOCAB_SIZE)


def test_forward_output_length_matches_caption_length():
    decoder = make_decoder()
    features = make_features(2)
    captions = make_captions(2, length=7)
    outputs = decoder(features, captions)
    # Output length T must equal the caption length so the loss lines up.
    assert outputs.shape[1] == captions.shape[1]


def test_linear_maps_to_vocabulary_size():
    decoder = make_decoder()
    assert decoder.linear.out_features == VOCAB_SIZE
    assert decoder.embedding.num_embeddings == VOCAB_SIZE


def test_forward_is_differentiable():
    decoder = make_decoder()
    features = make_features(3)
    captions = make_captions(3, length=6)
    outputs = decoder(features, captions)
    outputs.sum().backward()
    # Gradients should reach both the embedding and the output layer.
    assert decoder.embedding.weight.grad is not None
    assert decoder.linear.weight.grad is not None


def test_sample_output_shape_and_dtype():
    decoder = make_decoder()
    decoder.eval()
    features = make_features(4)
    generated = decoder.sample(features, max_length=15)
    assert generated.shape == (4, 15)
    assert generated.dtype == torch.long


def test_sample_indices_are_within_vocabulary():
    decoder = make_decoder()
    decoder.eval()
    generated = decoder.sample(make_features(4), max_length=15)
    assert generated.min().item() >= 0
    assert generated.max().item() < VOCAB_SIZE


def test_works_with_multiple_lstm_layers():
    decoder = make_decoder(num_layers=2)
    outputs = decoder(make_features(2), make_captions(2, length=8))
    assert outputs.shape == (2, 8, VOCAB_SIZE)


def test_encoder_and_decoder_connect():
    """Integration check: the encoder's output feeds straight into the decoder."""
    from encoder import EncoderCNN

    encoder = EncoderCNN(embed_size=EMBED_SIZE, pretrained=False)
    decoder = make_decoder()
    encoder.eval()
    decoder.eval()

    images = torch.randn(2, 3, 224, 224)
    captions = make_captions(2, length=9)

    features = encoder(images)          # [2, EMBED_SIZE]
    outputs = decoder(features, captions)
    assert outputs.shape == (2, 9, VOCAB_SIZE)


if __name__ == "__main__":
    # Allow running without pytest: discover and run every test_* function.
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print("=== test_decoder ===\n")
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
