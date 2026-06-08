"""
encoder.py
CNN encoder for the image captioning project.

The EncoderCNN turns each input image into a single feature vector of size
EMBED_SIZE. The LSTM decoder will later use this vector as its first input
(the idea from "Show and Tell", Vinyals et al. 2015: feed the image to the
RNN as if it were the very first word of the caption).

Design:
  * Use a pretrained ResNet (ImageNet weights) as a feature extractor.
  * Drop ResNet's final classification layer (the 1000-class ImageNet head),
    so the backbone outputs a 2048-dim feature vector per image.
  * Add a trainable linear layer that projects those 2048 features down to
    EMBED_SIZE, so the image vector lives in the same space as the word
    embeddings the decoder uses.
  * A BatchNorm on the projected features keeps training stable.

By default the convolutional backbone is FROZEN (its weights are not updated),
so only the small projection layer is trained. This is fast and works well on
a small dataset like Flickr8k. Pass freeze_cnn=False to fine-tune the backbone.

Run directly for a quick shape sanity check (downloads ResNet weights on the
first run only):
    python encoder.py
"""

import torch
import torch.nn as nn
from torchvision import models

from config import EMBED_SIZE


class EncoderCNN(nn.Module):
    """Encode an image into a single EMBED_SIZE feature vector."""

    def __init__(self, embed_size: int = EMBED_SIZE,
                 freeze_cnn: bool = True, pretrained: bool = True):
        super().__init__()

        # Load a ResNet-50. With pretrained=True we get the ImageNet weights,
        # which already give us a strong general-purpose image feature
        # extractor. (Tests use pretrained=False so they need no download.)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # A ResNet ends with:  ... -> avgpool -> fc (1000 ImageNet classes).
        # We keep every layer EXCEPT the final fc, so the backbone now outputs
        # a 2048-dim feature vector per image (taken right after avgpool).
        backbone_layers = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*backbone_layers)
        backbone_output_size = resnet.fc.in_features  # 2048 for ResNet-50

        # Optionally freeze the backbone so that only the projection layer is
        # trained. A frozen layer still runs in the forward pass, it just does
        # not receive weight updates during training.
        if freeze_cnn:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Project the 2048-dim CNN features down to EMBED_SIZE.
        self.project = nn.Linear(backbone_output_size, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        """images: FloatTensor [B, 3, 224, 224]  ->  features: [B, embed_size]."""
        features = self.backbone(images)             # [B, 2048, 1, 1]
        features = features.flatten(start_dim=1)     # [B, 2048]
        features = self.project(features)            # [B, embed_size]
        features = self.batch_norm(features)         # [B, embed_size]
        return features


if __name__ == "__main__":
    # Quick sanity check on a batch of random "images".
    encoder = EncoderCNN(embed_size=EMBED_SIZE)
    encoder.eval()  # eval mode so BatchNorm uses running stats

    dummy_images = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = encoder(dummy_images)

    print("Input shape :", tuple(dummy_images.shape))
    print("Output shape:", tuple(output.shape), "(expected (4,", EMBED_SIZE, "))")

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {frozen:,}")
