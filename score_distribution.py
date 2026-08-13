"""Summarize overlapping LR-patch Laplace scores for the Test4k dataset."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from utils.laplace import laplacian_map, rgb_to_gray


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_DIR = Path("/home/tyzheng/Datasets_pt/val/Test4k")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spatial_redundancy_plots"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the distribution of mean Laplace scores for overlapping "
            "x4 LR patches in Test4k."
        )
    )
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-path", type=Path, default=None, help="Existing histogram CSV to plot without recomputing scores.")
    parser.add_argument("--output-figure", type=Path, default=None, help="SVG output path; defaults to the computed or input CSV stem.")
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0, help="0 uses all images.")
    return parser.parse_args()


def iter_lr_images(val_dir, scale=4):
    """Yield CHW x4-downscaled images from Test4k validation shards."""
    shard_paths = sorted(val_dir.glob("*.pt"))
    if not shard_paths:
        raise ValueError(f"No validation .pt shards found in {val_dir}")

    lr_key = f"lr_x{scale}"
    for shard_path in shard_paths:
        packed = torch.load(shard_path, weights_only=False)
        if not isinstance(packed, dict) or lr_key not in packed:
            raise ValueError(f"{shard_path} is not a supported validation shard")
        for lr in packed[lr_key]:
            if lr.ndim != 3 or lr.shape[0] not in (1, 3):
                raise ValueError(
                    f"Expected a CHW grayscale/RGB LR image, got {tuple(lr.shape)} "
                    f"in {shard_path}"
                )
            yield lr


def patch_laplace_scores(image, patch_size, stride):
    """Match laplace_psnr.py: mean absolute 8-neighbor Laplace response."""
    image = image.unsqueeze(0).to(dtype=torch.float32)
    response = laplacian_map(rgb_to_gray(image))
    if response.shape[-2] < patch_size or response.shape[-1] < patch_size:
        return torch.empty(0, dtype=torch.float32)
    patches = F.unfold(response, kernel_size=patch_size, stride=stride)
    return patches.mean(dim=1).squeeze(0)


def collect_scores(val_dir, patch_size, overlap, max_images):
    stride = patch_size - overlap
    score_parts = []
    images_seen = 0
    for image in iter_lr_images(val_dir):
        if max_images and images_seen >= max_images:
            break
        images_seen += 1
        scores = patch_laplace_scores(image, patch_size, stride)
        if scores.numel():
            score_parts.append(scores.cpu())

    if not score_parts:
        raise ValueError(f"No {patch_size}x{patch_size} LR patches found in {val_dir}")
    return torch.cat(score_parts).numpy(), images_seen


def histogram(scores, bin_width):
    max_score = float(scores.max())
    upper = max(bin_width, np.ceil(max_score / bin_width) * bin_width)
    edges = np.arange(0.0, upper + bin_width * 0.5, bin_width)
    counts, edges = np.histogram(scores, bins=edges)
    return counts, edges


def save_csv(counts, edges, output_path):
    total = int(counts.sum())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("laplace_min", "laplace_max", "laplace_center", "patch_count", "patch_ratio"))
        for index, count in enumerate(counts):
            writer.writerow((
                edges[index], edges[index + 1], 0.5 * (edges[index] + edges[index + 1]),
                int(count), count / total,
            ))


def load_histogram_csv(csv_path):
    """Load the histogram format emitted by save_csv."""
    required_columns = {"laplace_min", "laplace_max", "patch_count"}
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path} has no histogram rows")

    lower_edges = np.asarray([float(row["laplace_min"]) for row in rows])
    upper_edges = np.asarray([float(row["laplace_max"]) for row in rows])
    counts = np.asarray([int(row["patch_count"]) for row in rows])
    edges = np.concatenate((lower_edges[:1], upper_edges))
    if counts.sum() <= 0 or np.any(counts < 0) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{csv_path} contains invalid histogram intervals or counts")
    return counts, edges


def save_plot(counts, edges, output_path):
    plt.rcParams["font.family"] = "Arial"
    centers = 0.5 * (edges[:-1] + edges[1:])
    proportions = counts / counts.sum() * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.bar(centers, proportions, width=np.diff(edges), align="center", color="#5a89e6", edgecolor="none")
    ax.set_xlabel("Mean Laplace score")
    ax.set_ylabel("Patch proportion (%)")
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(axis="y", color="gray", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    args = parse_args()
    if args.csv_path is not None:
        if not args.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
        counts, edges = load_histogram_csv(args.csv_path)
        output_path = args.output_figure or args.csv_path.with_suffix(".svg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot(counts, edges, output_path)
        print(f"Read {int(counts.sum()):,} patches from {args.csv_path}")
        print(f"Saved histogram plot to {output_path}")
        return

    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if not 0 <= args.overlap < args.patch_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < patch-size")
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")

    scores, image_count = collect_scores(
        args.val_dir, args.patch_size, args.overlap, args.max_images
    )
    counts, edges = histogram(scores, args.bin_width)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "laplace_score_distribution"
    np.save(stem.with_suffix(".npy"), scores)
    save_csv(counts, edges, stem.with_suffix(".csv"))
    output_path = args.output_figure or stem.with_suffix(".svg")
    save_plot(counts, edges, output_path)

    print(f"Analyzed {len(scores):,} patches from {image_count:,} x4 LR images.")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}], mean: {scores.mean():.4f}")
    print(f"Saved raw scores to {stem.with_suffix('.npy')}")
    print(f"Saved histogram data to {stem.with_suffix('.csv')}")
    print(f"Saved histogram plot to {output_path}")


if __name__ == "__main__":
    main()
