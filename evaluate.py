"""
evaluate.py
Generate captions for the test images and measure quality with BLEU.

What it does:
  * loads the trained model from a checkpoint and the shared vocabulary,
  * recreates the SAME test split used during training (split_by_image),
  * generates one caption per test image,
  * compares each generated caption against that image's five reference
    captions using corpus BLEU-1 through BLEU-4,
  * prints the scores and a few example image/caption pairs.

BLEU briefly: it measures how many n-grams (1 to 4 words) of the generated
caption also appear in the reference captions, with a penalty for captions that
are too short. Higher is better; 1.0 (or 100) is a perfect match.

Run (after training has produced a checkpoint):
    python evaluate.py
"""

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from config import (
    DEVICE,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    MAX_CAPTION_LENGTH,
    get_captions_file,
    get_images_dir,
)
from dataset import build_transform, split_by_image
from vocabulary import Vocabulary
from model import CaptioningModel


def build_references(captions_df):
    """Group the reference captions per image.

    Returns: dict mapping image name -> list of tokenized reference captions,
    e.g. {"123.jpg": [["a", "dog", "runs"], ["a", "puppy", "plays"], ...]}.
    """
    references = {}
    for image_name, group in captions_df.groupby("image"):
        references[image_name] = [
            Vocabulary.tokenize(caption) for caption in group["caption"]
        ]
    return references


@torch.no_grad()
def generate_hypotheses(model, image_names, images_dir, transform, vocab,
                        device, max_length=MAX_CAPTION_LENGTH):
    """Generate one caption per image.

    Returns: dict mapping image name -> tokenized generated caption.
    """
    model.eval()
    hypotheses = {}

    for image_name in tqdm(image_names, desc="generating", leave=False):
        with Image.open(images_dir / image_name) as image:
            image = image.convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

        caption = model.generate_captions(image_tensor, vocab,
                                          max_length=max_length)[0]
        hypotheses[image_name] = Vocabulary.tokenize(caption)

    return hypotheses


def align(references, hypotheses):
    """Line up references and hypotheses into two parallel lists for BLEU.

    Only images present in `hypotheses` are used, and the order is shared, so
    list_of_references[i] are the references for list_of_hypotheses[i].
    """
    list_of_references = []
    list_of_hypotheses = []
    for image_name, hypothesis_tokens in hypotheses.items():
        list_of_references.append(references[image_name])
        list_of_hypotheses.append(hypothesis_tokens)
    return list_of_references, list_of_hypotheses


def compute_bleu(list_of_references, list_of_hypotheses):
    """Compute corpus BLEU-1..4. Returns a dict like {"BLEU-1": 0.61, ...}.

    A smoothing function is used so that short captions missing some higher
    n-grams do not collapse the score straight to zero.
    """
    smoothing = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n)  # uniform weights over 1..n-grams
        scores[f"BLEU-{n}"] = corpus_bleu(
            list_of_references, list_of_hypotheses,
            weights=weights, smoothing_function=smoothing,
        )
    return scores


def load_model_from_checkpoint(path, device):
    """Rebuild the model from a checkpoint saved by train.py.

    pretrained=False because the checkpoint already contains all weights
    (including the backbone), so no ResNet download is needed here.
    """
    checkpoint = torch.load(path, map_location=device)
    model = CaptioningModel(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    device = DEVICE
    print(f"Using device: {device}")

    vocab = Vocabulary.load(OUTPUT_DIR / "vocab.pkl")
    model = load_model_from_checkpoint(CHECKPOINT_DIR / "best.pt", device)

    # Recreate the exact same test split used during training.
    captions_df = pd.read_csv(get_captions_file())
    _, _, test_df = split_by_image(captions_df)

    references = build_references(test_df)
    image_names = list(dict.fromkeys(test_df["image"].tolist()))  # unique, ordered

    transform = build_transform()
    hypotheses = generate_hypotheses(model, image_names, get_images_dir(),
                                     transform, vocab, device)

    list_of_references, list_of_hypotheses = align(references, hypotheses)
    scores = compute_bleu(list_of_references, list_of_hypotheses)

    print("\n===== BLEU scores on the test set =====")
    for name, value in scores.items():
        print(f"  {name}: {value:.4f}  ({value * 100:.2f})")

    # Show a few qualitative examples.
    print("\n===== Example captions =====")
    for image_name in image_names[:5]:
        generated = " ".join(hypotheses[image_name])
        reference = " ".join(references[image_name][0])
        print(f"\nImage    : {image_name}")
        print(f"Generated: {generated}")
        print(f"Reference: {reference}")


if __name__ == "__main__":
    main()
