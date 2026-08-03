import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import DPSR
from utils.laplace import (
    laplacian_map as shared_laplacian_map,
    rgb_to_gray,
    rgb_to_studio_y,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_DIR = Path("/home/tyzheng/Datasets_pt/val/Set5")
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "DPSR_x2_0803_1031.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spatial_redundancy_plots"
SCALE = 2
MIN_LAPLACIAN = 1
MAX_LAPLACIAN = 24
LAPLACIAN_STEP = 0.5
NUM_BINS = round((MAX_LAPLACIAN - MIN_LAPLACIAN) / LAPLACIAN_STEP)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate x2 super-resolution on Set5 blocks grouped by "
            "their LR Laplacian scores."
        )
    )
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--in-channels", type=int, default=3, choices=(1, 3))
    parser.add_argument("--channel-nums", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument(
        "--subnet-channels",
        type=int,
        default=16,
        help="Feature channels of the explicit full-depth DPSR subnet.",
    )
    parser.add_argument("--block-size", type=int, default=24, help="LR block width/height.")
    parser.add_argument("--max-images", type=int, default=0, help="0 uses the full dataset.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--trend-degree",
        type=int,
        default=3,
        choices=(1, 2, 3, 4, 5),
        help="Degree of the weighted Chebyshev trend curve.",
    )
    parser.add_argument(
        "--plot-range",
        nargs=2,
        type=float,
        default=(MIN_LAPLACIAN, MAX_LAPLACIAN),
        metavar=("MIN", "MAX"),
        help="Inclusive Laplacian range for the overview plot.",
    )
    parser.add_argument(
        "--detail-plot-range",
        nargs=2,
        type=float,
        default=(8.0, MAX_LAPLACIAN),
        metavar=("MIN", "MAX"),
        help="Inclusive Laplacian range for the DPSR-only detail plot.",
    )
    return parser.parse_args()


def to_y_channel(image):
    """Convert an NCHW tensor in [0, 255] to one studio-range Y channel."""
    return rgb_to_studio_y(image)


def rgb_to_model_y(image):
    """Match the one-channel conversion used by the validation dataset."""
    if image.shape[0] == 1:
        return image
    if image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}")
    image = image.to(torch.float32)
    y = (
        image[0] * 65.481
        + image[1] * 128.553
        + image[2] * 24.966
        + 16.0
    ) / 255.0
    return y.unsqueeze(0).round().clamp(0, 255).to(torch.uint8)


def iter_validation_pairs(val_dir, scale, in_channels):
    shard_files = sorted(val_dir.glob("*.pt"))
    if not shard_files:
        raise ValueError(f"No validation .pt shards found in {val_dir}")

    lr_key = f"lr_x{scale}"
    for shard_path in shard_files:
        packed = torch.load(shard_path, weights_only=False)
        if not isinstance(packed, dict) or "hr" not in packed or lr_key not in packed:
            raise ValueError(f"{shard_path} is not a supported validation shard")
        if len(packed["hr"]) != len(packed[lr_key]):
            raise ValueError(f"{shard_path} has mismatched HR/LR sample counts")

        for lr, hr in zip(packed[lr_key], packed["hr"]):
            if in_channels == 1:
                lr = rgb_to_model_y(lr)
                hr = rgb_to_model_y(hr)
            yield lr.unsqueeze(0), hr.unsqueeze(0)


def laplacian_map(image_y):
    """Return the shared absolute eight-neighbor Laplace response."""
    return shared_laplacian_map(image_y)


def block_means(image, block_size):
    blocks = F.unfold(image, kernel_size=block_size, stride=block_size)
    return blocks.mean(dim=1).reshape(-1)


def block_mse(prediction, target, block_size):
    return block_means((prediction - target).square(), block_size)


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        subnet_channels=args.subnet_channels,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model


def collect_block_statistics(dpsr_model, loader, args, device):
    score_parts = []
    dpsr_full_mse_parts = []
    dpsr_subnet_mse_parts = []
    bicubic_mse_parts = []

    with torch.no_grad():
        for image_index, (lr, hr) in enumerate(loader):
            if args.max_images and image_index >= args.max_images:
                break

            lr = lr.to(device=device, dtype=torch.float32)
            hr = hr.to(device=device, dtype=torch.float32)
            lr_height = min(lr.shape[-2], hr.shape[-2] // args.scale)
            lr_width = min(lr.shape[-1], hr.shape[-1] // args.scale)
            lr_height = (lr_height // args.block_size) * args.block_size
            lr_width = (lr_width // args.block_size) * args.block_size
            if lr_height == 0 or lr_width == 0:
                continue

            lr = lr[..., :lr_height, :lr_width]
            hr = hr[..., : lr_height * args.scale, : lr_width * args.scale]
            bicubic = F.interpolate(
                lr, scale_factor=args.scale, mode="bicubic", align_corners=False
            ).round().clamp(0, 255)
            model_input = lr / 255.0
            dpsr_full = bicubic + dpsr_model(
                model_input, channels=args.channel_nums
            ) * 255.0
            dpsr_subnet = bicubic + dpsr_model(
                model_input, channels=args.subnet_channels
            ) * 255.0
            dpsr_full = dpsr_full.round().clamp(0, 255)
            dpsr_subnet = dpsr_subnet.round().clamp(0, 255)

            lr_y = to_y_channel(lr)
            hr_y = to_y_channel(hr)
            bicubic_y = to_y_channel(bicubic)
            dpsr_full_y = to_y_channel(dpsr_full)
            dpsr_subnet_y = to_y_channel(dpsr_subnet)
            hr_block_size = args.block_size * args.scale

            score_parts.append(
                block_means(laplacian_map(rgb_to_gray(lr)), args.block_size).cpu()
            )
            dpsr_full_mse_parts.append(
                block_mse(dpsr_full_y, hr_y, hr_block_size).cpu()
            )
            dpsr_subnet_mse_parts.append(
                block_mse(dpsr_subnet_y, hr_y, hr_block_size).cpu()
            )
            bicubic_mse_parts.append(
                block_mse(bicubic_y, hr_y, hr_block_size).cpu()
            )

    if not score_parts:
        raise ValueError(
            f"No complete {args.block_size}x{args.block_size} LR blocks found in {args.val_dir}"
        )
    return (
        torch.cat(score_parts).numpy(),
        torch.cat(dpsr_full_mse_parts).numpy(),
        torch.cat(dpsr_subnet_mse_parts).numpy(),
        torch.cat(bicubic_mse_parts).numpy(),
    )


def bin_statistics(
    scores,
    dpsr_full_mse,
    dpsr_subnet_mse,
    bicubic_mse,
    bins,
    max_laplacian=None,
):
    if max_laplacian is None:
        max_laplacian = float(np.percentile(scores, 99.0))
    if max_laplacian <= 0:
        raise ValueError("Maximum Laplacian magnitude must be positive.")

    valid = scores <= max_laplacian
    scores = scores[valid]
    dpsr_full_mse = dpsr_full_mse[valid]
    dpsr_subnet_mse = dpsr_subnet_mse[valid]
    bicubic_mse = bicubic_mse[valid]
    edges = np.linspace(0.0, max_laplacian, bins + 1)
    bin_ids = np.digitize(scores, edges[1:-1])
    counts = np.bincount(bin_ids, minlength=bins)

    def mean_psnr_by_bin(mse):
        block_psnr = 10.0 * np.log10((255.0**2) / np.maximum(mse, 1e-12))
        psnr_sums = np.bincount(bin_ids, weights=block_psnr, minlength=bins)
        return np.divide(
            psnr_sums,
            counts,
            out=np.full(bins, np.nan, dtype=np.float64),
            where=counts > 0,
        )

    return {
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
        "counts": counts,
        "dpsr_full_psnr": mean_psnr_by_bin(dpsr_full_mse),
        "dpsr_subnet_psnr": mean_psnr_by_bin(dpsr_subnet_mse),
        "bicubic_psnr": mean_psnr_by_bin(bicubic_mse),
    }


def save_csv(stats, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "laplacian_min",
                "laplacian_max",
                "laplacian_center",
                "block_count",
                "dpsr_full_psnr_db",
                "dpsr_subnet_psnr_db",
                "bicubic_psnr_db",
            ]
        )
        for index, center in enumerate(stats["centers"]):
            writer.writerow(
                [
                    stats["edges"][index],
                    stats["edges"][index + 1],
                    center,
                    int(stats["counts"][index]),
                    stats["dpsr_full_psnr"][index],
                    stats["dpsr_subnet_psnr"][index],
                    stats["bicubic_psnr"][index],
                ]
            )


def fit_trend_curve(centers, values, counts, degree, points=256):
    valid = (
        np.isfinite(centers)
        & np.isfinite(values)
        & np.isfinite(counts)
        & (counts > 0)
    )
    x = centers[valid]
    y = values[valid]
    weights = np.sqrt(counts[valid])
    if x.size < 2:
        raise ValueError("At least two populated Laplacian intervals are required.")
    degree = min(int(degree), x.size - 1)
    trend = np.polynomial.Chebyshev.fit(x, y, degree, w=weights)
    trend_x = np.linspace(x.min(), x.max(), points)
    return trend_x, trend(trend_x)


def save_line_plot(stats, output_path, plot_range, trend_degree, include_bicubic):
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = "Arial"
    centers = stats["centers"]
    counts = stats["counts"]
    dpsr_full = stats["dpsr_full_psnr"]
    dpsr_subnet = stats["dpsr_subnet_psnr"]
    bicubic = stats["bicubic_psnr"]
    plot_min, plot_max = plot_range

    valid = (
        (centers >= plot_min)
        & (centers <= plot_max)
        & np.isfinite(dpsr_full)
        & np.isfinite(dpsr_subnet)
    )
    if include_bicubic:
        valid &= np.isfinite(bicubic)
    if valid.sum() < 2:
        raise ValueError(
            "At least two populated Laplacian intervals are required in the plot range."
        )

    x = centers[valid]
    weights = counts[valid]
    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)

    def plot_trend(values, color, label):
        trend_x, trend_y = fit_trend_curve(x, values[valid], weights, trend_degree)
        ax.plot(trend_x, trend_y, color=color, linewidth=3, label=label)

    if include_bicubic:
        plot_trend(bicubic, "#5a89e6", "Bicubic")
    plot_trend(dpsr_subnet, "#d9534f", f"DPSR({stats['subnet_channels']}ch)")
    plot_trend(dpsr_full, "#8ddb51", f"DPSR({stats['full_channels']}ch)")
    ax.set_xlim(plot_min, plot_max)
    ax.set_xlabel("Laplacian Magnitude", fontsize=22, labelpad=4, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=22, labelpad=4, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(color="gray", linewidth=1.0, alpha=0.65, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        borderaxespad=0,
        frameon=False,
        prop={"size": 13, "weight": "bold"},
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)


def validate_plot_range(plot_range, option_name):
    plot_min, plot_max = plot_range
    if not MIN_LAPLACIAN <= plot_min < plot_max <= MAX_LAPLACIAN:
        raise ValueError(
            f"{option_name} must satisfy {MIN_LAPLACIAN} <= MIN < MAX <= {MAX_LAPLACIAN}"
        )


def main():
    args = parse_args()
    args.scale = SCALE
    if not 0 < args.subnet_channels < args.channel_nums:
        raise ValueError(
            f"--subnet-channels must be in [1, {args.channel_nums - 1}]"
        )
    args.plot_range = tuple(args.plot_range)
    args.detail_plot_range = tuple(args.detail_plot_range)
    validate_plot_range(args.plot_range, "--plot-range")
    validate_plot_range(args.detail_plot_range, "--detail-plot-range")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dpsr_model = load_model(args, device)
    loader = iter_validation_pairs(args.val_dir, SCALE, args.in_channels)
    scores, dpsr_full_mse, dpsr_subnet_mse, bicubic_mse = collect_block_statistics(
        dpsr_model, loader, args, device
    )
    stats = bin_statistics(
        scores,
        dpsr_full_mse,
        dpsr_subnet_mse,
        bicubic_mse,
        bins=NUM_BINS,
        max_laplacian=MAX_LAPLACIAN,
    )
    stats["full_channels"] = args.channel_nums
    stats["subnet_channels"] = args.subnet_channels

    output_stem = args.output_dir / "laplace_psnr"
    csv_path = output_stem.with_suffix(".csv")
    overview_plot_path = output_stem.with_suffix(".svg")
    detail_plot_path = args.output_dir / "laplace_psnr_detail.svg"
    save_csv(stats, csv_path)
    save_line_plot(
        stats,
        overview_plot_path,
        plot_range=args.plot_range,
        trend_degree=args.trend_degree,
        include_bicubic=True,
    )
    save_line_plot(
        stats,
        detail_plot_path,
        plot_range=args.detail_plot_range,
        trend_degree=args.trend_degree,
        include_bicubic=False,
    )

    included_blocks = int(stats["counts"].sum())
    print(
        f"Analyzed {len(scores):,} blocks on {device}; "
        f"{included_blocks:,} have Laplacian scores in [0.0, {MAX_LAPLACIAN:.1f}]."
    )
    print(
        f"Compared full DPSR({args.channel_nums}ch) with "
        f"full-depth subnet DPSR({args.subnet_channels}ch)."
    )
    print(f"Saved {NUM_BINS} PSNR intervals to {csv_path}")
    print(f"Saved overview plot to {overview_plot_path}")
    print(f"Saved DPSR detail plot to {detail_plot_path}")


if __name__ == "__main__":
    main()
