"""
tests/test_dataset.py
Run with: pytest    (or: python tests/test_dataset.py)

Uses tiny generated images and captions in a temp folder, so it runs in a
second and needs NO Flickr8k download.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from vocabulary import Vocabulary
from dataset import FlickrDataset, collate_fn, split_by_image, build_transform


# --------------------------------------------------------------------------
# Fake data helpers
# --------------------------------------------------------------------------
def make_fake_dataset(tmp_dir, n_images=10, captions_per_image=5):
    """Create n_images tiny JPEGs and a matching captions DataFrame."""
    from PIL import Image

    images_dir = Path(tmp_dir) / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_images):
        name = f"img_{i:04d}.jpg"
        # Varying size on purpose, to confirm the transform standardises them.
        Image.new("RGB", (200 + i * 5, 180 + i * 3),
                  color=(i * 20 % 255, 90, 160)).save(images_dir / name)
        for j in range(captions_per_image):
            rows.append({"image": name,
                         "caption": f"a small dog is running in the green park {i} {j}"})

    return pd.DataFrame(rows), images_dir


def make_vocab(df):
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(df["caption"].tolist())
    return vocab


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_dataset_length_matches_rows():
    with tempfile.TemporaryDirectory() as tmp:
        df, images_dir = make_fake_dataset(tmp, n_images=4, captions_per_image=5)
        ds = FlickrDataset(df, images_dir, make_vocab(df), build_transform())
        assert len(ds) == 20  # 4 images * 5 captions


def test_getitem_returns_correct_image_shape():
    with tempfile.TemporaryDirectory() as tmp:
        df, images_dir = make_fake_dataset(tmp, n_images=3)
        ds = FlickrDataset(df, images_dir, make_vocab(df), build_transform())
        image, caption = ds[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32


def test_getitem_caption_wrapped_with_start_and_end():
    with tempfile.TemporaryDirectory() as tmp:
        df, images_dir = make_fake_dataset(tmp, n_images=3)
        vocab = make_vocab(df)
        ds = FlickrDataset(df, images_dir, vocab, build_transform())
        _, caption = ds[0]
        assert caption[0].item() == vocab.start_idx
        assert caption[-1].item() == vocab.end_idx
        assert caption.dtype == torch.long


def test_caption_respects_max_length():
    with tempfile.TemporaryDirectory() as tmp:
        df, images_dir = make_fake_dataset(tmp, n_images=2)
        vocab = make_vocab(df)
        ds = FlickrDataset(df, images_dir, vocab, build_transform(),
                           max_caption_length=5)
        _, caption = ds[0]
        assert len(caption) <= 5
        # still properly wrapped even after truncation
        assert caption[0].item() == vocab.start_idx
        assert caption[-1].item() == vocab.end_idx


def test_collate_fn_shapes_and_padding():
    with tempfile.TemporaryDirectory() as tmp:
        df, images_dir = make_fake_dataset(tmp, n_images=4)
        vocab = make_vocab(df)
        ds = FlickrDataset(df, images_dir, vocab, build_transform())

        batch = [ds[i] for i in range(4)]
        images, captions, lengths = collate_fn(batch, pad_idx=vocab.pad_idx)

        assert images.shape == (4, 3, 224, 224)
        assert captions.shape[0] == 4                 # batch dimension
        assert captions.shape[1] == int(lengths.max())  # padded to longest
        assert len(lengths) == 4


def test_collate_fn_pads_shorter_with_pad_idx():
    with tempfile.TemporaryDirectory() as tmp:
        df, images_dir = make_fake_dataset(tmp, n_images=4)
        vocab = make_vocab(df)
        ds = FlickrDataset(df, images_dir, vocab, build_transform())

        # Two captions of deliberately different lengths.
        short = (ds[0][0], torch.tensor([vocab.start_idx, 5, vocab.end_idx]))
        long = (ds[1][0], torch.tensor([vocab.start_idx, 5, 6, 7, 8, vocab.end_idx]))
        _, captions, lengths = collate_fn([short, long], pad_idx=vocab.pad_idx)

        assert captions.shape == (2, 6)               # padded to the longer one
        # The short caption's tail should be padding.
        assert captions[0, -1].item() == vocab.pad_idx
        assert lengths.tolist() == [3, 6]


def test_split_by_image_no_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        df, _ = make_fake_dataset(tmp, n_images=100, captions_per_image=5)
        train_df, val_df, test_df = split_by_image(df, 0.8, 0.1, seed=42)

        train_imgs = set(train_df["image"])
        val_imgs = set(val_df["image"])
        test_imgs = set(test_df["image"])

        assert train_imgs.isdisjoint(val_imgs)
        assert train_imgs.isdisjoint(test_imgs)
        assert val_imgs.isdisjoint(test_imgs)
        # every image landed somewhere
        assert len(train_imgs | val_imgs | test_imgs) == 100


def test_split_fractions_are_roughly_right():
    with tempfile.TemporaryDirectory() as tmp:
        df, _ = make_fake_dataset(tmp, n_images=100, captions_per_image=5)
        train_df, val_df, test_df = split_by_image(df, 0.8, 0.1, seed=42)

        # 100 images -> 80/10/10 -> 400/50/50 caption rows
        assert len(train_df) == 400
        assert len(val_df) == 50
        assert len(test_df) == 50


def test_split_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        df, _ = make_fake_dataset(tmp, n_images=50, captions_per_image=5)
        a = split_by_image(df, 0.8, 0.1, seed=7)[0]["image"].tolist()
        b = split_by_image(df, 0.8, 0.1, seed=7)[0]["image"].tolist()
        assert a == b


if __name__ == "__main__":
    # Allow running without pytest: discover and run every test_* function.
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print("=== test_dataset ===\n")
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
