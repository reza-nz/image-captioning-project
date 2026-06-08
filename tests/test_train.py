"""
tests/test_train.py
Unit tests for the training loop.

Run from the project root with:
    pytest

Uses a handful of tiny generated images + captions and a small model with
pretrained=False, so it needs NO Flickr8k download and runs quickly.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial

from vocabulary import Vocabulary
from dataset import FlickrDataset, collate_fn, build_transform
from model import CaptioningModel
from train import train_one_epoch, evaluate_loss


EMBED_SIZE = 32
HIDDEN_SIZE = 64


def make_fake_dataset(tmp_dir, n_images=8, captions_per_image=5):
    """Create tiny JPEGs and a matching captions DataFrame."""
    from PIL import Image

    images_dir = Path(tmp_dir) / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_images):
        name = f"img_{i:04d}.jpg"
        Image.new("RGB", (200, 200),
                  color=(i * 20 % 255, 90, 160)).save(images_dir / name)
        for j in range(captions_per_image):
            rows.append({"image": name,
                         "caption": "a small dog is running in the green park"})
    return pd.DataFrame(rows), images_dir


def make_loader(tmp_dir, batch_size=4):
    df, images_dir = make_fake_dataset(tmp_dir)
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(df["caption"].tolist())

    dataset = FlickrDataset(df, images_dir, vocab, build_transform())
    collate = partial(collate_fn, pad_idx=vocab.pad_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)
    return loader, vocab


def make_model(vocab):
    return CaptioningModel(vocab_size=len(vocab), embed_size=EMBED_SIZE,
                           hidden_size=HIDDEN_SIZE, dropout=0.0, pretrained=False)


def test_train_one_epoch_returns_finite_loss():
    with tempfile.TemporaryDirectory() as tmp:
        loader, vocab = make_loader(tmp)
        model = make_model(vocab)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)

        loss = train_one_epoch(model, loader, criterion, optimizer, device="cpu")
        assert isinstance(loss, float)
        assert loss > 0
        assert torch.isfinite(torch.tensor(loss))


def test_training_updates_the_trainable_weights():
    with tempfile.TemporaryDirectory() as tmp:
        loader, vocab = make_loader(tmp)
        model = make_model(vocab)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)

        before = model.decoder.linear.weight.detach().clone()
        train_one_epoch(model, loader, criterion, optimizer, device="cpu")
        after = model.decoder.linear.weight.detach().clone()

        # The trainable decoder weights must have moved after one epoch.
        assert not torch.allclose(before, after)


def test_evaluate_loss_returns_finite_and_does_not_update_weights():
    with tempfile.TemporaryDirectory() as tmp:
        loader, vocab = make_loader(tmp)
        model = make_model(vocab)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

        before = model.decoder.linear.weight.detach().clone()
        loss = evaluate_loss(model, loader, criterion, device="cpu")
        after = model.decoder.linear.weight.detach().clone()

        assert isinstance(loss, float)
        assert loss > 0
        # Evaluation must NOT change any weights.
        assert torch.allclose(before, after)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print("=== test_train ===\n")
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
