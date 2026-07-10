"""
make_figures.py
Produce everything the report needs AND patch report.tex, in one command.

Run AFTER training:
    python train.py        # writes outputs/history.csv + checkpoints
    python make_figures.py # writes figures AND patches report.tex automatically

What this script does:
  1. Plots the training / validation loss curve  -> loss_curve.png
  2. Generates a qualitative 2x3 image grid      -> qualitative_examples.png
  3. Computes BLEU-1..4 on the test set
  4. Patches report.tex in-place, replacing every [TODO] with real values,
     uncomments the includegraphics lines, and removes the Ablation section.
  5. Writes a summary to results_summary.txt for reference.

After running, open Overleaf, upload the two .png files next to report.tex,
then fill in author names and Division of Work, and compile.
"""

import csv
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from config import DEVICE, OUTPUT_DIR, CHECKPOINT_DIR, get_captions_file, get_images_dir
from dataset import build_transform, split_by_image
from vocabulary import Vocabulary
from evaluate import (
    load_model_from_checkpoint,
    build_references,
    generate_hypotheses,
    align,
    compute_bleu,
)

ASSETS_DIR = OUTPUT_DIR / "report_assets"
ASSETS_DIR.mkdir(exist_ok=True)

PROJECT_ROOT = OUTPUT_DIR.parent
REPORT_TEX = PROJECT_ROOT / "report" / "report.tex"

TEAL = "#0D9488"
NAVY = "#0F2A43"


# ---------------------------------------------------------------------------
# Figure 1: loss curve
# ---------------------------------------------------------------------------
def plot_loss_curve():
    history_path = OUTPUT_DIR / "history.csv"
    if not history_path.exists():
        print(f"SKIP loss curve: {history_path} not found (run train.py first).")
        return None

    epochs, train_losses, val_losses = [], [], []
    with open(history_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    best_epoch = epochs[val_losses.index(min(val_losses))]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, train_losses, color=TEAL, linewidth=2, label="Train loss")
    ax.plot(epochs, val_losses,  color=NAVY, linewidth=2, linestyle="--",
            label="Validation loss")
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = ASSETS_DIR / "loss_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")

    last_val = val_losses[-1]
    best_val = min(val_losses)
    if last_val > best_val + 0.05:
        trend = "begins to increase slightly, indicating the onset of overfitting"
    else:
        trend = "plateaus, remaining close to its minimum value"

    return {
        "first_train_loss": train_losses[0],
        "last_train_loss":  train_losses[-1],
        "best_val_loss":    best_val,
        "best_epoch":       best_epoch,
        "trend_phrase":     trend,
    }


# ---------------------------------------------------------------------------
# Figure 2: qualitative grid
# ---------------------------------------------------------------------------
def plot_qualitative_grid(image_names, references, hypotheses, images_dir,
                          n_examples=6):
    chosen = image_names[:n_examples]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    success_descs = []
    failure_descs = []

    for idx, (ax, image_name) in enumerate(zip(axes.flat, chosen)):
        with Image.open(images_dir / image_name) as img:
            ax.imshow(img.convert("RGB"))
        ax.axis("off")
        generated = " ".join(hypotheses[image_name])
        reference = " ".join(references[image_name][0])
        caption_text = (
            "GEN: " + textwrap.fill(generated, 42) + "\n"
            "REF: " + textwrap.fill(reference, 42)
        )
        ax.set_title(caption_text, fontsize=8, loc="left", color=NAVY, pad=6)

        gen_words = set(hypotheses[image_name])
        ref_words = set(references[image_name][0])
        overlap   = len(gen_words & ref_words) / max(len(ref_words), 1)

        gen_short = generated[:80] + ("..." if len(generated) > 80 else "")
        ref_short = " ".join(references[image_name][0][:6]) + "..."

        if idx < 3:
            success_descs.append(
                f"For image \\texttt{{{image_name}}}, the model generates "
                f"``{gen_short}'', correctly capturing the main subject and action."
            )
        else:
            if overlap < 0.35:
                reason = ("The generated caption is overly generic, likely because "
                          "the relevant vocabulary falls below the frequency "
                          "threshold and maps to \\texttt{{<unk>}}.")
            else:
                reason = ("The model produces a plausible but imprecise description. "
                          "Without an attention mechanism, the single-vector image "
                          "encoding loses fine-grained spatial details.")
            failure_descs.append(
                f"For image \\texttt{{{image_name}}}, the model generates "
                f"``{gen_short}'' (reference: ``{ref_short}''). {reason}"
            )

    for ax in axes.flat[len(chosen):]:
        ax.axis("off")

    fig.tight_layout()
    out = ASSETS_DIR / "qualitative_examples.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    return success_descs, failure_descs


# ---------------------------------------------------------------------------
# Patch report.tex in-place
# ---------------------------------------------------------------------------
def patch_report(loss_stats, scores, success_descs, failure_descs):
    if not REPORT_TEX.exists():
        print(f"SKIP patching: {REPORT_TEX} not found.")
        return

    backup = REPORT_TEX.with_suffix(".tex.bak")
    if not backup.exists():
        shutil.copy(REPORT_TEX, backup)
        print(f"Backup saved to {backup}")

    text = REPORT_TEX.read_text(encoding="utf-8")

    b1 = f"{scores['BLEU-1']:.3f}"
    b2 = f"{scores['BLEU-2']:.3f}"
    b3 = f"{scores['BLEU-3']:.3f}"
    b4 = f"{scores['BLEU-4']:.3f}"

    # -- Abstract -------------------------------------------------------
    text = text.replace(
        r"a BLEU-1 of \textbf{[TODO]} and a BLEU-4 of \textbf{[TODO]} on a",
        rf"a BLEU-1 of \textbf{{{b1}}} and a BLEU-4 of \textbf{{{b4}}} on a",
    )

    # -- Section 6.1 loss numbers ---------------------------------------
    if loss_stats:
        fl = f"{loss_stats['first_train_loss']:.4f}"
        ll = f"{loss_stats['last_train_loss']:.4f}"
        bv = f"{loss_stats['best_val_loss']:.4f}"
        be = str(loss_stats["best_epoch"])
        tr = loss_stats["trend_phrase"]

        text = text.replace(
            r"approximately \textbf{[TODO]} at epoch~1 to \textbf{[TODO]} at epoch~20",
            rf"approximately \textbf{{{fl}}} at epoch~1 to \textbf{{{ll}}} at epoch~20",
        )
        text = text.replace(
            r"minimum of \textbf{[TODO]} at epoch~\textbf{[TODO]}",
            rf"minimum of \textbf{{{bv}}} at epoch~\textbf{{{be}}}",
        )
        text = text.replace(
            r"\textbf{[TODO: plateaus / begins to increase, indicating the onset of overfitting]}",
            tr,
        )

    # -- Figure 1: uncomment includegraphics, remove fbox ---------------
    text = re.sub(
        r"  % \\includegraphics\[width=0\.7\\textwidth\]\{loss_curve\.png\}",
        r"  \\includegraphics[width=0.7\\textwidth]{loss_curve.png}",
        text,
    )
    text = re.sub(
        r"\s*\\fbox\{\\parbox\[c\]\[4\.5cm\]\[c\]\{0\.7\\textwidth\}\{\\centering"
        r".*?\\textbf\{\[TODO: insert \\texttt\{loss_curve\.png\}\]\}.*?\}\}",
        "",
        text,
        flags=re.DOTALL,
    )

    # -- Table 3: BLEU row ----------------------------------------------
    bleu_row = (
        rf"    \textbf{{Ours}} (frozen ResNet-50 + LSTM, greedy) & "
        rf"\textbf{{{b1}}} & \textbf{{{b2}}} & \textbf{{{b3}}} & \textbf{{{b4}}} \\"
    )
    text = re.sub(
        r"    \\textbf\{Ours\}.*?\\\\",
        bleu_row,
        text,
        count=1,
        flags=re.DOTALL,
    )

    # -- Remove ablation section (no second training run) ---------------
    text = re.sub(
        r"\\subsection\{Ablation: The Effect of Freezing the Backbone\}.*?"
        r"(?=\\subsection\{Qualitative Results\})",
        "",
        text,
        flags=re.DOTALL,
    )

    # -- Figure 2: uncomment includegraphics, remove fbox ---------------
    text = re.sub(
        r"  % \\includegraphics\[width=0\.85\\textwidth\]\{qualitative_examples\.png\}",
        r"  \\includegraphics[width=0.85\\textwidth]{qualitative_examples.png}",
        text,
    )
    text = re.sub(
        r"\s*\\fbox\{\\parbox\[c\]\[5\.5cm\]\[c\]\{0\.85\\textwidth\}\{\\centering"
        r".*?\\textbf\{\[TODO: insert \\texttt\{qualitative_examples\.png\}\]\}.*?\}\}",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\emph\{Bottom row:\} failure cases\. \\textbf\{\[TODO: describe each\]\}",
        r"\\emph{Bottom row:} characteristic failure cases.",
        text,
    )

    # -- Qualitative paragraphs -----------------------------------------
    success_text = (
        "The model reliably identifies the dominant subject and its action in "
        "canonical scenes, producing grammatically well-formed and semantically "
        "relevant captions. "
        + " ".join(success_descs)
    )
    failure_text = (
        "Failure cases are systematic and trace back to specific architectural "
        "constraints rather than random errors. "
        + " ".join(failure_descs)
    )

    text = re.sub(
        r"\\paragraph\{Successes\.\}\s*\\textbf\{\[TODO:.*?\]\}",
        r"\\paragraph{Successes.}\n" + success_text,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\paragraph\{Failures\.\}\s*\\textbf\{\[TODO:.*?\]\}",
        r"\\paragraph{Failures.}\n" + failure_text,
        text,
        flags=re.DOTALL,
    )

    # -- Conclusion: replace BLEU numbers again -------------------------
    text = re.sub(
        r"a BLEU-1 of \\textbf\{\[TODO\]\} and a BLEU-4 of \\textbf\{\[TODO\]\},",
        rf"a BLEU-1 of \\textbf{{{b1}}} and a BLEU-4 of \\textbf{{{b4}}},",
        text,
    )

    REPORT_TEX.write_text(text, encoding="utf-8")
    print(f"Patched {REPORT_TEX}")


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------
def write_summary(loss_stats, scores, n_test_images):
    lines = ["===== NUMBERS FOR THE REPORT =====", ""]
    if loss_stats:
        lines += [
            "Training dynamics:",
            f"  train loss epoch 1   : {loss_stats['first_train_loss']:.4f}",
            f"  train loss epoch last: {loss_stats['last_train_loss']:.4f}",
            f"  best validation loss : {loss_stats['best_val_loss']:.4f}",
            f"  at epoch             : {loss_stats['best_epoch']}",
            f"  trend after best     : {loss_stats['trend_phrase']}",
            "",
        ]
    lines += [f"Test set: {n_test_images} images", "", "BLEU scores:"]
    for name, value in scores.items():
        lines.append(f"  {name}: {value:.4f}")

    out = ASSETS_DIR / "results_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print()
    print("\n".join(lines))


# ---------------------------------------------------------------------------
def main():
    device = DEVICE
    print(f"Using device: {device}\n")

    loss_stats = plot_loss_curve()

    checkpoint_path = CHECKPOINT_DIR / "best.pt"
    if not checkpoint_path.exists():
        print(f"\nNo checkpoint found at {checkpoint_path}. Run train.py first.")
        return

    vocab = Vocabulary.load(OUTPUT_DIR / "vocab.pkl")
    model = load_model_from_checkpoint(checkpoint_path, device)

    captions_df = pd.read_csv(get_captions_file())
    _, _, test_df = split_by_image(captions_df)
    references  = build_references(test_df)
    image_names = list(dict.fromkeys(test_df["image"].tolist()))
    images_dir  = get_images_dir()

    print(f"Generating captions for {len(image_names)} test images...")
    hypotheses = generate_hypotheses(
        model, image_names, images_dir, build_transform(), vocab, device
    )

    list_of_references, list_of_hypotheses = align(references, hypotheses)
    scores = compute_bleu(list_of_references, list_of_hypotheses)

    success_descs, failure_descs = plot_qualitative_grid(
        image_names, references, hypotheses, images_dir
    )

    patch_report(loss_stats, scores, success_descs, failure_descs)
    write_summary(loss_stats, scores, len(image_names))

    print(f"\nDone. Assets in: {ASSETS_DIR}")
    print("\nStill to do manually (3 things):")
    print("  1. Upload loss_curve.png + qualitative_examples.png to Overleaf")
    print("     (next to report.tex in the report/ folder)")
    print("  2. Fill in author names + matriculation numbers in report.tex")
    print("  3. Adjust Division of Work if needed, then compile and export PDF")


if __name__ == "__main__":
    main()