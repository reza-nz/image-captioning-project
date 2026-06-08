"""
model.py
The full image captioning model: EncoderCNN + DecoderRNN in one module.

This is the single object train.py and evaluate.py work with. It wires the two
parts together so the rest of the code never has to touch them separately:

    images --[EncoderCNN]--> feature vector --[DecoderRNN]--> word scores

Methods:
  * forward(images, captions) : teacher-forced pass for TRAINING. Returns word
                                scores of shape [B, T, vocab_size].
  * generate(images)          : greedy inference. Returns token indices
                                [B, max_length] (no ground-truth caption needed).
  * generate_captions(images, vocab) : convenience wrapper that turns those
                                indices into readable strings, cutting each one
                                at the first <end> token.

Run directly for a quick shape sanity check (no dataset / download needed):
    python model.py
"""

import torch
import torch.nn as nn

from config import EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
from encoder import EncoderCNN
from decoder import DecoderRNN


class CaptioningModel(nn.Module):
    """Encoder + decoder wrapped into a single image captioning model."""

    def __init__(self, vocab_size: int, embed_size: int = EMBED_SIZE,
                 hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT, freeze_cnn: bool = True,
                 pretrained: bool = True):
        super().__init__()

        self.encoder = EncoderCNN(embed_size=embed_size, freeze_cnn=freeze_cnn,
                                  pretrained=pretrained)
        self.decoder = DecoderRNN(vocab_size=vocab_size, embed_size=embed_size,
                                  hidden_size=hidden_size, num_layers=num_layers,
                                  dropout=dropout)

    def forward(self, images, captions):
        """Teacher-forced forward pass used during training.

        images   : FloatTensor [B, 3, 224, 224]
        captions : LongTensor  [B, T]            (with <start> ... <end>)
        returns  : FloatTensor [B, T, vocab_size]
        """
        features = self.encoder(images)            # [B, embed_size]
        outputs = self.decoder(features, captions)  # [B, T, vocab_size]
        return outputs

    @torch.no_grad()
    def generate(self, images, max_length: int = 20):
        """Greedy-decode captions for a batch of images (inference).

        images  : FloatTensor [B, 3, 224, 224]
        returns : LongTensor  [B, max_length]  (token indices)
        """
        features = self.encoder(images)
        return self.decoder.sample(features, max_length=max_length)

    @torch.no_grad()
    def generate_captions(self, images, vocab, max_length: int = 20):
        """Generate readable caption strings for a batch of images.

        Each generated sequence is cut at the first <end> token and then turned
        into a string (special tokens are skipped). Handy for evaluate.py.

        returns : list[str], one caption per image
        """
        token_indices = self.generate(images, max_length=max_length)

        captions = []
        for indices in token_indices:
            captions.append(self._decode_until_end(indices, vocab))
        return captions

    @staticmethod
    def _decode_until_end(indices, vocab) -> str:
        """Cut a sequence of indices at the first <end>, then decode to text."""
        kept_indices = []
        for index in indices.tolist():
            if index == vocab.end_idx:
                break
            kept_indices.append(index)
        return vocab.decode(kept_indices)


if __name__ == "__main__":
    # Quick sanity check with random images and fake captions (pretrained=False
    # so no ResNet weights are downloaded).
    vocab_size = 100
    batch_size = 4
    caption_length = 12

    model = CaptioningModel(vocab_size=vocab_size, pretrained=False)
    model.eval()

    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, caption_length))

    outputs = model(images, captions)
    print("Images shape :", tuple(images.shape))
    print("Captions shape:", tuple(captions.shape))
    print("Output shape :", tuple(outputs.shape),
          f"(expected ({batch_size}, {caption_length}, {vocab_size}))")

    generated = model.generate(images, max_length=15)
    print("Generated shape:", tuple(generated.shape),
          f"(expected ({batch_size}, 15))")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {frozen:,}")
