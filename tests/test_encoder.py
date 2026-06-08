"""
tests/test_encoder.py
Unit tests for the EncoderCNN.

Run from the project root with:
    pytest

All tests build the encoder with pretrained=False, so they need NO internet
download and run on random tensors in a second or two.
"""

import sys
from pathlib import Path

# Allow importing project modules (config.py, encoder.py) from the repo root
# when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from encoder import EncoderCNN


def make_image_batch(batch_size: int = 4):
    """A batch of random, ResNet-shaped images: [B, 3, 224, 224]."""
    return torch.randn(batch_size, 3, 224, 224)


def test_encoder_output_shape():
    encoder = EncoderCNN(embed_size=256, pretrained=False)
    encoder.eval()
    features = encoder(make_image_batch(4))
    assert features.shape == (4, 256)


def test_encoder_respects_custom_embed_size():
    encoder = EncoderCNN(embed_size=128, pretrained=False)
    encoder.eval()
    features = encoder(make_image_batch(2))
    assert features.shape == (2, 128)


def test_output_is_float_tensor():
    encoder = EncoderCNN(embed_size=64, pretrained=False)
    encoder.eval()
    features = encoder(make_image_batch(2))
    assert features.dtype == torch.float32


def test_backbone_is_frozen_by_default():
    encoder = EncoderCNN(embed_size=64, pretrained=False)
    # Every backbone parameter must be frozen...
    assert all(not p.requires_grad for p in encoder.backbone.parameters())
    # ...while the projection layer must stay trainable.
    assert all(p.requires_grad for p in encoder.project.parameters())


def test_backbone_is_trainable_when_not_frozen():
    encoder = EncoderCNN(embed_size=64, freeze_cnn=False, pretrained=False)
    assert all(p.requires_grad for p in encoder.backbone.parameters())


def test_encoder_handles_single_image_in_eval_mode():
    encoder = EncoderCNN(embed_size=64, pretrained=False)
    encoder.eval()  # eval mode -> BatchNorm uses running stats, batch of 1 ok
    features = encoder(make_image_batch(1))
    assert features.shape == (1, 64)


def test_projection_receives_gradients_but_frozen_backbone_does_not():
    encoder = EncoderCNN(embed_size=32, pretrained=False)
    encoder.eval()

    features = encoder(make_image_batch(2))
    loss = features.sum()
    loss.backward()

    # The trainable projection layer should have received a gradient...
    assert encoder.project.weight.grad is not None
    # ...while the frozen backbone should not have any.
    first_backbone_param = next(encoder.backbone.parameters())
    assert first_backbone_param.grad is None


if __name__ == "__main__":
    # Allow running without pytest: discover and run every test_* function.
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print("=== test_encoder ===\n")
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
