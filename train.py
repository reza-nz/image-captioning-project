"""
train.py
Training loop for the image captioning model.

What it does:
  * builds the train/val/test DataLoaders and the shared vocabulary,
  * builds the CaptioningModel (encoder + decoder),
  * trains with teacher forcing using cross-entropy loss (padding ignored),
  * evaluates on the validation set after every epoch,
  * saves checkpoints (always "last.pt", and "best.pt" when val loss improves).

The loss alignment:
    model(images, captions) returns scores of shape [B, T, vocab_size], where
    each step predicts the NEXT token. The target is simply `captions` itself
    (<start> ... <end>), and <pad> positions are ignored via ignore_index.

Run:
    python train.py
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    CHECKPOINT_DIR,
)
from dataset import get_data_loaders
from model import CaptioningModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full training pass over `loader`. Returns the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, captions, lengths in tqdm(loader, desc="train", leave=False):
        # A batch of size 1 breaks BatchNorm in train mode, so skip it.
        if images.size(0) == 1:
            continue

        images = images.to(device)
        captions = captions.to(device)

        outputs = model(images, captions)  # [B, T, vocab_size]

        # Flatten batch and time so cross-entropy compares one token at a time.
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),  # [B*T, vocab_size]
            captions.reshape(-1),                   # [B*T]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    """Compute the average loss over `loader` without updating weights."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)

        outputs = model(images, captions)
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            captions.reshape(-1),
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, path, epoch, val_loss):
    """Save the model weights plus enough info to rebuild it for evaluation.

    The architecture sizes are read straight from the model, so the checkpoint
    always matches the weights it stores (no reliance on the config constants
    being unchanged).
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "vocab_size": model.decoder.embedding.num_embeddings,
            "embed_size": model.decoder.embedding.embedding_dim,
            "hidden_size": model.decoder.lstm.hidden_size,
            "num_layers": model.decoder.lstm.num_layers,
        },
        path,
    )


def main():
    device = DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, vocab = get_data_loaders()
    print(f"Vocabulary size: {len(vocab)}")

    model = CaptioningModel(vocab_size=len(vocab)).to(device)

    # Ignore <pad> positions so padding never contributes to the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Only the trainable parameters (the frozen ResNet backbone is excluded).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        save_checkpoint(model, CHECKPOINT_DIR / "last.pt", epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, CHECKPOINT_DIR / "best.pt", epoch, val_loss)
            print(f"  -> new best model saved (val loss {val_loss:.4f})")

    print(f"\nTraining done. Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
