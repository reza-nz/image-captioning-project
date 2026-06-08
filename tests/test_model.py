"""
tests/test_model.py
Unit tests for the full CaptioningModel (encoder + decoder).

Run from the project root with:
    pytest

All tests build the model with pretrained=False, so no download is needed.
Building a ResNet backbone is a little slow, so the tests share ONE model
instance (built lazily on first use) instead of creating a new one each time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import CaptioningModel
from encoder import EncoderCNN
from decoder import DecoderRNN
from vocabulary import Vocabulary


EMBED_SIZE = 32
HIDDEN_SIZE = 64

# A tiny real vocabulary so generate_captions has something to decode with.
# Using the actual Vocabulary class keeps the test honest.
SAMPLE_CAPTIONS = [
    "a small dog is running in the park",
    "a girl plays with a red ball",
    "two men are walking on the beach",
]
VOCAB = Vocabulary(freq_threshold=1)
VOCAB.build_vocabulary(SAMPLE_CAPTIONS)
VOCAB_SIZE = len(VOCAB)


_shared_model = None


def get_model():
    """Build the model once and reuse it (ResNet construction is slow)."""
    global _shared_model
    if _shared_model is None:
        _shared_model = CaptioningModel(
            vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE, dropout=0.0, pretrained=False,
        )
        _shared_model.eval()
    return _shared_model


def make_images(batch_size: int = 4):
    return torch.randn(batch_size, 3, 224, 224)


def make_captions(batch_size: int = 4, length: int = 10):
    return torch.randint(0, VOCAB_SIZE, (batch_size, length))


def test_model_holds_an_encoder_and_a_decoder():
    model = get_model()
    assert isinstance(model.encoder, EncoderCNN)
    assert isinstance(model.decoder, DecoderRNN)


def test_forward_output_shape():
    model = get_model()
    outputs = model(make_images(4), make_captions(4, length=10))
    assert outputs.shape == (4, 10, VOCAB_SIZE)


def test_generate_output_shape_and_dtype():
    model = get_model()
    generated = model.generate(make_images(3), max_length=15)
    assert generated.shape == (3, 15)
    assert generated.dtype == torch.long


def test_generate_indices_are_within_vocabulary():
    model = get_model()
    generated = model.generate(make_images(3), max_length=15)
    assert generated.min().item() >= 0
    assert generated.max().item() < VOCAB_SIZE


def test_generate_captions_returns_list_of_strings():
    model = get_model()
    captions = model.generate_captions(make_images(2), VOCAB, max_length=12)
    assert isinstance(captions, list)
    assert len(captions) == 2
    assert all(isinstance(c, str) for c in captions)
    # Special tokens must never appear in the readable output.
    for caption in captions:
        assert "<start>" not in caption
        assert "<end>" not in caption
        assert "<pad>" not in caption


def test_only_the_projection_and_decoder_are_trainable():
    model = get_model()
    # The frozen ResNet backbone must have no trainable parameters...
    assert all(not p.requires_grad for p in model.encoder.backbone.parameters())
    # ...while the decoder is fully trainable.
    assert all(p.requires_grad for p in model.decoder.parameters())


def test_forward_is_differentiable_for_trainable_parts():
    model = get_model()
    model.zero_grad()

    outputs = model(make_images(2), make_captions(2, length=8))
    outputs.sum().backward()

    # Gradients should reach the decoder and the encoder's projection layer...
    assert model.decoder.linear.weight.grad is not None
    assert model.encoder.project.weight.grad is not None
    # ...but not the frozen backbone.
    first_backbone_param = next(model.encoder.backbone.parameters())
    assert first_backbone_param.grad is None


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print("=== test_model ===\n")
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
