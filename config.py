"""
config.py
Central configuration for the image captioning project.

All paths, hyperparameters and environment settings live here so that the rest
of the code (dataset, model, training, evaluation) stays clean and the project
runs identically on a laptop, GitHub Codespaces, Google Colab or Kaggle.

Static settings (tokens, hyperparameters, thresholds) are plain constants.
Dataset paths are resolved LAZILY via get_data_dir() / get_images_dir() /
get_captions_file(), so simply importing this module never triggers the ~1GB
download — only calling those functions does. This keeps imports and unit
tests fast.

    from config import EMBED_SIZE, DEVICE, get_captions_file
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Dataset location (resolved lazily — importing config does NOT download)
# ---------------------------------------------------------------------------
# The Flickr8k dataset lives on Kaggle. `kagglehub` downloads it once, caches
# it locally, and returns the same path on every environment. If you are on
# Kaggle and have attached the dataset directly, set DATA_DIR_OVERRIDE to
# "/kaggle/input/flickr8k" to skip the download.
DATA_DIR_OVERRIDE = None  # e.g. "/kaggle/input/flickr8k"

_data_dir_cache = None


def get_data_dir() -> Path:
    """Return the dataset root, downloading via kagglehub on the first call.

    The result is cached, so repeated calls are free. Importing config does
    not call this — the download happens the first time you actually need the
    data (e.g. when building the vocab or the dataset).
    """
    global _data_dir_cache
    if _data_dir_cache is None:
        if DATA_DIR_OVERRIDE is not None:
            _data_dir_cache = Path(DATA_DIR_OVERRIDE)
        else:
            import kagglehub
            _data_dir_cache = Path(kagglehub.dataset_download("adityajn105/flickr8k"))
    return _data_dir_cache


def get_images_dir() -> Path:
    return get_data_dir() / "Images"


def get_captions_file() -> Path:
    return get_data_dir() / "captions.txt"


# ---------------------------------------------------------------------------
# Output location (checkpoints, saved vocab, generated captions)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() # Apple Silicon GPU support (e.g. on M1/2/3... Macs)
    else "cpu"
)
# ---------------------------------------------------------------------------
# Image preprocessing (matches the ResNet-compatible transform from Section A)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
VOCAB_FREQ_THRESHOLD = 5   # words appearing fewer times than this become <unk>
MAX_CAPTION_LENGTH = 50    # captions longer than this are truncated

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
NUM_WORKERS = 2
RANDOM_SEED = 42

# Train / val / test split done BY IMAGE (so no image leaks across splits).
# Fractions must sum to 1.0.
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
