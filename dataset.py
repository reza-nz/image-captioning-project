"""
dataset.py
PyTorch Dataset, collate function and data loaders for the captioning project.

Pieces:
  * build_transform()  - the ResNet-compatible image transform (from Section A).
  * FlickrDataset      - returns (image_tensor, caption_indices) for one sample.
  * collate_fn         - pads captions in a batch to equal length.
  * split_by_image     - train/val/test split done BY IMAGE (no leakage).
  * get_data_loaders   - ties everything together into three DataLoaders.

Run directly to sanity-check the pipeline on the real dataset:
    python dataset.py
"""

from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image

from config import (
    RESIZE_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_CAPTION_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    OUTPUT_DIR,
    get_captions_file,
    get_images_dir,
)
from vocabulary import Vocabulary


def build_transform():
    """The image transform: resize -> center crop 224 -> tensor -> normalize."""
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FlickrDataset(Dataset):
    """One sample = one caption (so each image appears `captions_per_image` times).

    __getitem__ returns:
        image_tensor    : FloatTensor [3, IMAGE_SIZE, IMAGE_SIZE]
        caption_indices : LongTensor  [seq_len]  (wrapped with <start>/<end>)
    """

    def __init__(self, df, images_dir, vocab, transform,
                 max_caption_length: int = MAX_CAPTION_LENGTH):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform
        self.max_caption_length = max_caption_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row["image"]
        caption = row["caption"]

        # Load and transform the image.
        with Image.open(self.images_dir / image_name) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        # Trim word tokens so that <start> + words + <end> fits the length cap.
        word_indices = self.vocab.numericalize(caption)[: self.max_caption_length - 2]
        caption_indices = [self.vocab.start_idx] + word_indices + [self.vocab.end_idx]

        return image_tensor, torch.tensor(caption_indices, dtype=torch.long)


def collate_fn(batch, pad_idx: int):
    """Collate a list of (image, caption) samples into padded batch tensors.

    Returns:
        images   : FloatTensor [B, 3, H, W]
        captions : LongTensor  [B, T]  (T = longest caption in this batch)
        lengths  : LongTensor  [B]     (true length of each caption, pre-padding)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, padded_captions, lengths


def split_by_image(df, train_frac: float = TRAIN_FRAC,
                   val_frac: float = VAL_FRAC, seed: int = RANDOM_SEED):
    """Split the captions DataFrame into train/val/test BY IMAGE.

    Splitting on unique image names (not rows) guarantees that all five
    captions of an image stay together in one split, so the model is never
    evaluated on an image it trained on.
    """
    # .unique() can return a pandas extension array; shuffling that directly is
    # unreliable (may duplicate entries), so convert to a plain object array first.
    unique_images = np.array(df["image"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)

    n = len(unique_images)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_images = set(unique_images[:n_train])
    val_images = set(unique_images[n_train:n_train + n_val])
    test_images = set(unique_images[n_train + n_val:])

    train_df = df[df["image"].isin(train_images)].reset_index(drop=True)
    val_df = df[df["image"].isin(val_images)].reset_index(drop=True)
    test_df = df[df["image"].isin(test_images)].reset_index(drop=True)
    return train_df, val_df, test_df


def get_data_loaders(vocab=None, batch_size: int = BATCH_SIZE,
                     num_workers: int = NUM_WORKERS):
    """Build train/val/test DataLoaders (and the vocabulary they share).

    If `vocab` is None, loads outputs/vocab.pkl when present, otherwise builds
    and saves a fresh vocabulary from the captions.
    """
    captions_df = pd.read_csv(get_captions_file())

    if vocab is None:
        vocab_path = OUTPUT_DIR / "vocab.pkl"
        if vocab_path.exists():
            vocab = Vocabulary.load(vocab_path)
        else:
            vocab = Vocabulary()
            vocab.build_vocabulary(captions_df["caption"].tolist())
            vocab.save(vocab_path)

    transform = build_transform()
    images_dir = get_images_dir()
    train_df, val_df, test_df = split_by_image(captions_df)

    datasets = {
        "train": FlickrDataset(train_df, images_dir, vocab, transform),
        "val": FlickrDataset(val_df, images_dir, vocab, transform),
        "test": FlickrDataset(test_df, images_dir, vocab, transform),
    }

    collate = partial(collate_fn, pad_idx=vocab.pad_idx)

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate)

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        batch_size=4, num_workers=0
    )
    print("Vocabulary size:", len(vocab))
    print("Batches  ->  train:", len(train_loader),
          "| val:", len(val_loader), "| test:", len(test_loader))

    images, captions, lengths = next(iter(train_loader))
    print("Image batch shape  :", tuple(images.shape))
    print("Caption batch shape:", tuple(captions.shape))
    print("Caption lengths    :", lengths.tolist())
