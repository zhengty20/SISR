import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "spatial_redundancy_plots" / "laplace_psnr.csv"
REQUIRED_COLUMNS = (
    "laplacian_center",
    "dpsr_full_psnr_db",
    "dpsr_subnet_psnr_db",
    "bicubic_psnr_db",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Laplacian-binned PSNR values from a CSV file."
    )
    parser.add_argument("csv_path", type=Path, nargs="?", default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the SVG file (defaults to the CSV directory).",
    )
    return parser.parse_args()


def load_statistics(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")
    return {
        column: np.asarray([float(row[column]) for row in rows])
        for column in REQUIRED_COLUMNS
    }


def save_line_plot(stats, output_path):
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = "Arial"
    centers = stats["laplacian_center"]
    bicubic = stats["bicubic_psnr_db"]
    dpsr_full = stats["dpsr_full_psnr_db"]
    dpsr_subnet = stats["dpsr_subnet_psnr_db"]
    valid = (
        np.isfinite(centers)
        & np.isfinite(bicubic)
        & np.isfinite(dpsr_full)
        & np.isfinite(dpsr_subnet)
    )
    if valid.sum() < 2:
        raise ValueError("At least two complete PSNR rows are required.")
    x = centers[valid]
    fig, ax = plt.subplots(figsize=(4.9, 4.2), constrained_layout=True)
    ax.plot(x, bicubic[valid], color="#5a89e6", linewidth=3, label="Bicubic")
    ax.plot(x, dpsr_subnet[valid], color="#d9534f", linewidth=3, label="DPSR(subnet)")
    ax.plot(x, dpsr_full[valid], color="#8ddb51", linewidth=3, label="DPSR(full)")
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Laplacian Magnitude", fontsize=22, labelpad=4, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=22, labelpad=4, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(color="gray", linewidth=1.0, alpha=0.65, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, prop={"size": 16, "weight": "bold"})
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = load_statistics(args.csv_path)
    plot_path = output_dir / "laplace_psnr.svg"
    save_line_plot(stats, plot_path)
    print(f"Read {len(stats['laplacian_center'])} PSNR intervals from {args.csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
